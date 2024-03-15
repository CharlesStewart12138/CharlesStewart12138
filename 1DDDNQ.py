import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from adabound import AdaBound
from torch import nn
import collections
import random
#定义一个经验回放池，他允许强化学习算法在学习中实现经验回放
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)                          #初始化一个用于储存经验的队列，指定最大长度，避免过度占用内存，经验池样本量不变
    def add(self, state, action, reward, next_state, done):                       #添加数据，如果队列已满，旧的经验会被移除，给新经验留出空间
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):                                                 #随机采样，返回的数据包含了动作、状态和奖励，以及结束标志
        transitions = random.sample(self.buffer, batch_size)                      # list, len=32
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    def size(self):
        return len(self.buffer)
#构建深度神经网络结构，目标网络和训练网络共用该结构。这里为了后续的动作选择策略更偏向“探索”而引入噪声网络。网络层采用卷积层
#引入带噪声的卷积层，并且引入噪声的动态改变机制
class NoisyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, std_init=0.4):
        super(NoisyConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 确保 kernel_size 是一个用于一致处理的元组
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.std_init = std_init
        # 权重参数
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.weight_sigma = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.register_buffer('weight_epsilon', torch.empty(out_channels, in_channels, *self.kernel_size))
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_sigma = nn.Parameter(torch.empty(out_channels))
        self.register_buffer('bias_epsilon', torch.empty(out_channels))
        # 用于噪声水平控制的参数
        self.noise_level = 1.0

        self.reset_parameters()
        self.reset_noise()
    #参数初始化
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_channels * np.prod(self.kernel_size))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_channels * np.prod(self.kernel_size)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_channels))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_channels)
        epsilon_out = self._scale_noise(self.out_channels)
        epsilon_in = epsilon_in.view(self.in_channels, 1, 1)
        epsilon_out = epsilon_out.view(self.out_channels, 1, 1)
        epsilon_w = epsilon_out * epsilon_in
        self.weight_epsilon.copy_(epsilon_w.view_as(self.weight_epsilon))
        # 偏置噪声
        epsilon_bias = self._scale_noise(self.out_channels)
        self.bias_epsilon.copy_(epsilon_bias)
    #调整噪声水平
    def adjust_noise_level(self, noise_level):
        self.noise_level = noise_level
    #生成噪声
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    #向前传播
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not x.is_floating_point():
            x = x.float()
        device = next(self.parameters()).device
        x = x.to(device)
        if self.training:
            weight = self.weight_mu + self.noise_level * self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.noise_level * self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.conv1d(x, weight, bias, padding=(self.kernel_size[0] // 2,))
#用噪声卷积层代替原有的线性全连接层
class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, input_length):
        super(Net, self).__init__()
        # 辅助函数，用于计算卷积或池化层之后的输出长度
        def compute_conv1d_output_size(L_in, padding, kernel_size, stride):
            return (L_in + 2*padding - kernel_size) // stride + 1
        padding = 0
        stride = 1
        pool_kernel_size = 2
        pool_stride = 2
        # 定义卷积层
        self.conv1 = NoisyConv1d(n_states, 16, kernel_size=3)
        self.conv2 = NoisyConv1d(16, 32, kernel_size=3)
        self.conv3 = NoisyConv1d(32, 64, kernel_size=3)
        self.conv4 = NoisyConv1d(64, 32, kernel_size=3)
        self.conv5 = NoisyConv1d(32, 16, kernel_size=3)
        # 定义池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 逐层计算输出维度
        L_out = compute_conv1d_output_size(input_length, padding, 3, stride)
        L_out = compute_conv1d_output_size(L_out, padding, pool_kernel_size, pool_stride)  
        L_out = compute_conv1d_output_size(L_out, padding, 3, stride)
        L_out = compute_conv1d_output_size(L_out, padding, pool_kernel_size, pool_stride)  
        L_out = compute_conv1d_output_size(L_out, padding, 3, stride)
        L_out = compute_conv1d_output_size(L_out, padding, 3, stride)
        L_out = compute_conv1d_output_size(L_out, padding, 3, stride)
        # 丢弃层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层的输入维度现在是基于计算结果
        self.fc_input_dim = 16 * L_out
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, n_hiddens[0])
        self.fc_layers = nn.ModuleList()
        for i in range(1, len(n_hiddens)):
            self.fc_layers.append(nn.Linear(n_hiddens[i-1], n_hiddens[i]))
        self.fc_final = nn.Linear(n_hiddens[-1], n_actions)
    def forward(self, x):
        # 重塑输入以匹配NoisyConv1d的期望输入
        x = x.unsqueeze(0).unsqueeze(-1)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        print(f"Before entering fully connected layer, shape: {x.shape}")  
        x = x.view(-1, self.fc_input_dim)  
        x = F.relu(self.fc1(x))
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        x = self.fc_final(x)
        x = self.dropout(x)
        return x
    # 重置所有噪声层的噪声
    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, NoisyConv1d):
                module.reset_noise()
    #定义Double-DQN神经网络，其中优化器采用性能更优的ababound优化算法
class Double_DQN:
    def __init__(self, n_states, n_hiddens, n_actions,                                       #初始化
                 learning_rate, gamma, epsilon, 
                 target_update, device):
        self.n_states = n_states                                                             #定义参数
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0

        self.q_net = Net(self.n_states, self.n_hiddens, self.n_actions).to(device)           # 确保网络移到了正确的设备上
        self.target_q_net = Net(self.n_states, self.n_hiddens, self.n_actions).to(device)  
        #导入adabound优化器
        self.optimizer = AdaBound(self.q_net.parameters(), lr=self.learning_rate)  
    #定义动作选择策略——注意！不同的改进算法往往要在这里进行改进，例如不同的强化学习策略等等
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).unsqueeze(-1).to(self.device)       # 调整形状并移动到设备
        with torch.no_grad():
            actions_value = self.q_net(state)
        action_probabilities = F.softmax(actions_value, dim=-1).cpu().numpy()[0]             # 确保操作在CPU上进行，以便使用numpy
        action = np.random.choice(np.arange(self.n_actions), p=action_probabilities)
        return action
    #获取每个状态对应的最大的state_value
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).unsqueeze(-1).to(self.device)       # 调整形状并移动到设备
        with torch.no_grad():
            max_q = self.q_net(state).max().item()
        return max_q
    #网络训练
    def update(self, transitions_dict):
        # 加载数据并确保它们在正确的设备上
        states = torch.tensor(transitions_dict['states'], dtype=torch.float).unsqueeze(-1).to(self.device)
        actions = torch.tensor(transitions_dict['actions'], dtype=torch.int64).view(-1,1).to(self.device)
        rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float).unsqueeze(-1).to(self.device)
        dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1,1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        #定义损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
         # 梯度清零和梯度翻转，更新训练参数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        # 更新目标网络参数
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
