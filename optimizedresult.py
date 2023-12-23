import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import X, Y, Z
from qiskit.quantum_info import state_fidelity
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 设置字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 定义哈密顿量的构建函数
def build_hamiltonian(m, g, num_qubits):
    X_matrix = Operator(X).data
    Y_matrix = Operator(Y).data
    Z_matrix = Operator(Z).data

    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_qubits - 1):
        xi_xi1 = np.kron(np.eye(2**i), np.kron(-1 * (np.kron(X_matrix, X_matrix) + np.kron(Y_matrix, Y_matrix)), np.eye(2**(num_qubits-i-2))))
        zi_zi1 = np.kron(np.eye(2**i), np.kron(g * np.kron(Z_matrix, Z_matrix), np.eye(2**(num_qubits-i-2))))
        hamiltonian += xi_xi1 + zi_zi1

    for i in range(num_qubits):
        zi = np.kron(np.eye(2**i), np.kron(m * Z_matrix, np.eye(2**(num_qubits-i-1))))
        hamiltonian += zi

    return hamiltonian

# 定义高斯脉冲函数
def gaussian_pulse(t, A, mu, sigma, f, phi):
    return A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * f * t + phi)

# 时间演化函数（含量子电路构建）
def time_evolution(hamiltonian, initial_state, steps, interval, num_qubits, action, noise_strength=0.02):
    states = []
    fidelities = []  # 用于存储每个步骤的保真度
    backend = Aer.get_backend('statevector_simulator')
    evolution_circuit = QuantumCircuit(num_qubits)

    for i, state in enumerate(initial_state):
        if state == '1':
            evolution_circuit.x(i)

    for step in range(steps):
        U = scipy.linalg.expm(-1j * hamiltonian * interval)
        evolution_operator = Operator(U)

        temp_circuit = evolution_circuit.copy()
        temp_circuit.append(evolution_operator, range(num_qubits))

        # 应用高斯脉冲
        for qubit in range(num_qubits):
            # 确保 action 数组长度正确
            A, mu, sigma, f, phi = action[qubit*5:(qubit+1)*5]
            pulse = gaussian_pulse(step, A, mu, sigma, f, phi)
            temp_circuit.rx(pulse, qubit)

            # 添加噪声：对每个量子比特应用小的随机旋转
            temp_circuit.rx(noise_strength * np.random.randn(), qubit)
            temp_circuit.ry(noise_strength * np.random.randn(), qubit)
            temp_circuit.rz(noise_strength * np.random.randn(), qubit)

        transpiled_circuit = transpile(temp_circuit, backend)
        job = execute(transpiled_circuit, backend)
        result = job.result()
        state = Statevector(result.get_statevector())
        states.append(state)
        evolution_circuit = temp_circuit

        # 记录保真度
        fidelity = state_fidelity(Statevector.from_label('0' * num_qubits), state)
        fidelities.append(fidelity)

    return states, fidelities


# 绘图函数
def plot_graphs(states_list, hamiltonians, masses, couplings, num_qubits, dt):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    # 绘制哈密顿量演化图
    for hamiltonian, m, g in zip(hamiltonians, masses, couplings):
        energies = np.linalg.eigvalsh(hamiltonian)
        time_points = np.arange(len(energies))
        e_interp = interp1d(time_points, energies, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_energies = e_interp(smooth_time_points)
        axs[0].plot(smooth_time_points, smooth_energies, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[0].scatter(time_points, energies, s=10)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Energy')
    axs[0].set_title('Hamiltonian Evolution')
    axs[0].legend()

    # 绘制粒子数密度演化图
    for states, m, g in zip(states_list, masses, couplings):
        densities = [np.mean([(-1)**l * state.data[2**l] + 1 for l in range(num_qubits)]) / (2 * num_qubits) for state in states]
        time_points = np.arange(len(densities))
        d_interp = interp1d(time_points, densities, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_densities = d_interp(smooth_time_points)
        axs[1].plot(smooth_time_points, smooth_densities, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[1].scatter(time_points, densities, s=10)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Particle Density')
    axs[1].set_title('Particle Density Evolution')
    axs[1].legend()

    # 绘制真空持续性振幅演化图
    for hamiltonian, states, m, g in zip(hamiltonians, states_list, masses, couplings):
        vacuum_state = Statevector.from_label('0' * num_qubits)
        amplitudes = [np.abs(vacuum_state.evolve(Operator(scipy.linalg.expm(-1j * hamiltonian * dt * step))).data @ state.data) for step, state in enumerate(states)]
        time_points = np.arange(len(amplitudes))
        a_interp = interp1d(time_points, amplitudes, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_amplitudes = a_interp(smooth_time_points)
        axs[2].plot(smooth_time_points, smooth_amplitudes, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[2].scatter(time_points, amplitudes, s=10)
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Vacuum Persistence Amplitude')
    axs[2].set_title('Vacuum Persistence Amplitude Evolution')
    axs[2].legend()

    # 绘制能量演化图
    for hamiltonian, states, m, g in zip(hamiltonians, states_list, masses, couplings):
        energies = [np.real(np.conj(state.data) @ hamiltonian @ state.data) for state in states]
        time_points = np.arange(len(energies))
        e_interp = interp1d(time_points, energies, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_energies = e_interp(smooth_time_points)
        axs[3].plot(smooth_time_points, smooth_energies, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[3].scatter(time_points, energies, s=10)
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Energy')
    axs[3].set_title('Energy Evolution')
    axs[3].legend()

    # 绘制保真度的演化图
    for states, m, g in zip(states_list, masses, couplings):
        time_points = np.arange(len(states))
        initial_state = states[0]
        fidelities = [state_fidelity(initial_state, state) for state in states]
        f_interp = interp1d(time_points, fidelities, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_fidelities = f_interp(smooth_time_points)
        axs[4].plot(smooth_time_points, smooth_fidelities, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[4].scatter(time_points, fidelities, s=10)
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Fidelity')
    axs[4].set_title('Fidelity Evolution')
    axs[4].legend()

    plt.tight_layout()
    plt.show()



# 绘制保真度随参数变化的图表
def plot_fidelity_vs_params(fidelities, params):
    # 假设 params 是一个列表，其中每个元素是一个包含所有参数的列表
    A_vals, mu_vals, sigma_vals, f_vals, phi_vals = zip(*params)

    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    axs[0].plot(A_vals, fidelities, label='Fidelity vs Amplitude')
    axs[0].set_xlabel('Amplitude')
    axs[0].set_ylabel('Fidelity')
    axs[0].legend()

    axs[1].plot(mu_vals, fidelities, label='Fidelity vs Pulse Center')
    axs[1].set_xlabel('Pulse Center')
    axs[1].set_ylabel('Fidelity')
    axs[1].legend()

    axs[2].plot(sigma_vals, fidelities, label='Fidelity vs Pulse Width')
    axs[2].set_xlabel('Pulse Width')
    axs[2].set_ylabel('Fidelity')
    axs[2].legend()

    axs[3].plot(f_vals, fidelities, label='Fidelity vs Frequency')
    axs[3].set_xlabel('Frequency')
    axs[3].set_ylabel('Fidelity')
    axs[3].legend()

    axs[4].plot(phi_vals, fidelities, label='Fidelity vs Phase')
    axs[4].set_xlabel('Phase')
    axs[4].set_ylabel('Fidelity')
    axs[4].legend()

    plt.tight_layout()
    plt.show()

class QuantumCircuitEnv(gym.Env):
    def __init__(self, num_qubits, time_steps, dt, masses, couplings):
        super(QuantumCircuitEnv, self).__init__()
        self.num_qubits = num_qubits
        self.time_steps = time_steps
        self.dt = dt
        self.masses = masses
        self.couplings = couplings
        # 每个量子比特有5个动作参数
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_qubits, 5), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2**num_qubits,), dtype=np.float32)
        self.state = None
        self.hamiltonian = None
        self.target_state = Statevector.from_label('0' * num_qubits)

    def step(self, action):
        # 确保动作数组具有正确的结构
        if action.shape != (self.num_qubits, 5):
            raise ValueError(f"Expected action array of shape ({self.num_qubits}, 5), but got {action.shape}")

        # 解析动作并应用
        U = np.eye(2**self.num_qubits, dtype=complex)
        for qubit in range(self.num_qubits):
            A, mu, sigma, f, phi = action[qubit]
            pulse = gaussian_pulse(np.arange(self.time_steps), A, mu, sigma, f, phi)
            for t in range(self.time_steps):
                U = np.dot(scipy.linalg.expm(-1j * self.hamiltonian * self.dt * pulse[t]), U)

        evolution_operator = Operator(U)
        circuit = QuantumCircuit(self.num_qubits)
        circuit.append(evolution_operator, range(self.num_qubits))
        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(circuit, backend)
        job = execute(transpiled_circuit, backend)
        result = job.result()
        new_state = Statevector(result.get_statevector())

        # 计算保真度
        fidelity = state_fidelity(self.target_state, new_state)
        reward = fidelity
        self.state = new_state.data
        done = True
        return self.state, reward, done, {}

    def reset(self):
        # 重置环境状态
        m, g = np.random.choice(self.masses), np.random.choice(self.couplings)
        self.hamiltonian = build_hamiltonian(m, g, self.num_qubits)
        self.state = Statevector.from_label('0' * self.num_qubits).data
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")


# 主程序
def main():
    num_qubits = 4
    time_steps = 100
    dt = 0.1
    masses = [0.5, 1.0, 1.5, 2.0]
    couplings = [0.1, 0.5, 1.0, 1.5]
    env = QuantumCircuitEnv(num_qubits, time_steps, dt, masses, couplings)
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练代理
    model.learn(total_timesteps=10000)

    # 使用优化后的参数进行模拟
    states_list = []
    hamiltonians = []
    all_fidelities = []  # 存储所有模拟的保真度
    all_params = []  # 存储所有模拟的参数

    for m in masses:
        for g in couplings:
            hamiltonian = build_hamiltonian(m, g, num_qubits)
            hamiltonians.append(hamiltonian)
            initial_state = Statevector.from_label('0' * num_qubits).data
            initial_state_real = np.abs(initial_state).reshape(1, -1)

            # 生成整体动作数组
            action, _ = model.predict(initial_state_real)
            action = action.reshape(num_qubits, 5)

            states, fidelities = time_evolution(hamiltonian, '0' * num_qubits, time_steps, dt, num_qubits, action.flatten())
            states_list.append(states)
            all_fidelities.extend(fidelities)
            for a in action:
                all_params.append(tuple(a))

    plot_graphs(states_list, hamiltonians, masses, couplings, num_qubits,dt)
    plot_fidelity_vs_params(all_fidelities, all_params)

if __name__ == "__main__":
    main()

    
