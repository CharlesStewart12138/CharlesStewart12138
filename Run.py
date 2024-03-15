import torch
import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from parsers import args
from RL import ReplayBuffer, Double_DQN
from gym.envs.registration import register
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 注册环境
register(
    id='CustomEnv-v0',                              # 'CustomEnv-v0'是本工程环境唯一的标识符，后续调用均使用本ID调用
    entry_point='environment:CustomEnv',            # 指定环境类的路径
)
#加载环境
env = gym.make('CustomEnv-v0')
n_states = env.observation_space.shape[0]           # 状态数
n_actions = 20                                      # 动作是连续的[-2,2]，将其离散成11个动作
act_low = 1
act_high = 20                                       # 定义离散动作的上界和下界
print(n_states)
#确定连续动作
def dis_to_con(discrete_action, n_actions):
    return act_low + (act_high-act_low) * (discrete_action/(n_actions-1))
# 实例化经验池
replay_buffer = ReplayBuffer(args.capacity)
# 示例参数，根据需要调整
#n_states = 6
n_hiddens = 128  # 这里简化为单一隐藏层大小，而不是列表
n_actions = 20
input_length = 4  # 简化的输入长度示例
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
target_update = 10
# 实例化Double_DQN类
agent = Double_DQN(n_states=n_states, 
                        n_hiddens=n_hiddens, 
                        n_actions=n_actions, 
                        learning_rate=learning_rate, 
                        gamma=gamma, epsilon=epsilon, 
                        target_update=target_update, 
                        device=device, 
                        input_length=input_length
                       )
def state_prepare():
    def mock_env_reset():
        return env.reset()
    # 转换为torch张量并重塑的过程
    def process_and_reshape():
        tensors = []
        for _ in range(4):  # 运行四遍
            state = mock_env_reset()  # 模拟env.reset()的返回
            tensor = torch.from_numpy(state).float()  # 转换为torch张量
            tensors.append(tensor.squeeze(0))  # 移除多余的维度并添加到列表中
        tensor_4x4 = torch.stack(tensors)  # 将四个[4]张量堆叠成一个[4,4]张量
        return tensor_4x4
    # 拓展[4,4]张量到[4,4,4]
    def expand_tensor(tensor_4x4):
        expanded_tensor = tensor_4x4.unsqueeze(0).repeat(4, 1, 1)  # 在第0维度增加维度并重复4次
        return expanded_tensor
    # 主流程
    tensor_4x4 = process_and_reshape()
    expanded_tensor = expand_tensor(tensor_4x4)
    state = expanded_tensor
    return state
return_list = []  # 记录每次迭代的return，即链上的reward之和
max_q_value = 0  # 最大state_value
max_q_value_list = []  # 保存所有最大的state_value

for i in range(10):
    done = False  # 初始，未到达终点
    state = state_prepare()
    episode_return = 0

    with tqdm(total=10, desc='Iteration %d' % i) as pbar:
        while True:
            print(type(state),state.shape)
            action = agent.take_action(state)
            max_q_value = agent.max_q_value(state) * 0.005 + \
                        max_q_value * 0.995
            max_q_value_list.append(max_q_value)
            action_continuous = dis_to_con(action, n_actions)

            result = env.step(action_continuous)
            print(result)

            next_state, reward, done, _ = env.step(action_continuous)
            
            replay_buffer.add(state, action, reward, next_state, done)
            if isinstance(next_state, np.ndarray) and next_state.shape == (4,):
                # 将 next_state 转换为 torch.Tensor 并调整形状至 [1, 4]
                next_state = torch.from_numpy(next_state).float().view(1, 4)
                # 重复数据以形成 [4, 4, 4] 张量
                next_state = next_state.repeat(4, 4)
                next_state = next_state.view(4, 4, 4)
            elif isinstance(next_state, torch.Tensor) and next_state.shape == (4, 4, 4):
                # 如果 next_state 已经是 [4, 4, 4] 张量，则直接使用
                next_state = next_state
            else:
                # 其他情况，根据实际需求进行处理
                raise ValueError("Unexpected state shape")

            state = next_state
            episode_return += reward
            print(state)
            if replay_buffer.size() > args.min_size:
                s, a, r, ns, d = replay_buffer.sample(args.batch_size)
                transitions_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                agent.update(transitions_dict)
            if done is True: break
        return_list.append(episode_return)
        pbar.set_postfix({
            'step':
            agent.count,
            'return':
            '%.3f' % np.mean(return_list[-10:])
        })
        pbar.update(1)
