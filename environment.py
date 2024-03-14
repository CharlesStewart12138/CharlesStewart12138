import gym
from gym import spaces
import numpy as np
#定义适用于最优p搜索的环境类：
class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(2)                                               # 由于p只能是整数，所以动作空间假设只有两个：+1和-1
        self.observation_space = spaces.Box(low=np.array([0, -np.inf, -np.inf, -np.inf]),
                                            high=np.array([10, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)                                # 状态由p, F, C, t组成，这里假设保真度，计算复杂度，演化时间和参数p构成一个态
        self.state = None                                                                    # 状态初始化
        self.current_step = 0                                                                # 添加步数计数器
        self.reset()
    # 设置参数的初始值：为了调试代码方便，这里使用了随机值，实际问题中需要将其考虑在内
    def reset(self):
        F = np.random.uniform(1, 10)
        C = np.random.uniform(1, 10)
        t = np.random.uniform(0, 5)
        p = np.random.uniform(0, 10)  
        self.state = np.array([p, F, C, t])
        self.current_step = 0  
        return self.state
    # 定义动作
    def step(self, action):
        self.current_step += 1  
        p, F, C, t = self.state
        delta = -1 if action == 0 else 1
        p = np.clip(p + delta, 0, 10)                                                        # 根据动作修改参数p的值
        F = self.F_change(p, F, C, t)                                                        # 参数F的改变，下同；
        C = self.C_change(p, F, C, t)
        t = self.t_change(p, F, C, t)
        self.state = np.array([p, F, C, t])                                                  # 更新状态
        reward = self.calculate_reward(p, F, C, t)                                           # 设置奖励
        done = self.is_done(p, F, C, t)                                                      # 定义结束回合的条件
        return self.state, reward, done, {}                                                  # 返回新的状态信息 
    # 定义F，C，t三个参数的改变策略，这样每一个函数都返回一个数学意义上的函数值，这里只是示例，实际工程中要进行替换
    def F_change(self, p, F, C, t):
        return p + F + t + C
    def C_change(self, p, F, C, t):
        return p + F + t + C
    def t_change(self, p, F, C, t):
        return t
    # 定义奖励函数：这个奖励函数的值也只是一个例子，要根据实际把这个替换。
    def calculate_reward(self, p, F, C, t):
        return F * C * np.exp(t)
    # 定义回合截止函数
    def is_done(self, p, F, C, t):
        if self.current_step >= 500:  
            return True
        elif C > 100:  
            return True
        elif F <= 0.01:  
            return True
        else:
            return False
