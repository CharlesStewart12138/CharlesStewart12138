import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import X, Y, Z
from qiskit.visualization import plot_state_city
from qiskit.quantum_info import state_fidelity
import scipy.linalg
from scipy.interpolate import interp1d

# 设置字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 设置参数
num_qubits = 4
masses = [0.5, 1.0, 1.5, 2.0]  # 质量值
couplings = [0.1, 0.5, 1.0, 1.5]  # 耦合强度
time_steps = 100  # 时间步数
dt = 0.1  # 时间间隔
noise_strength = 0.02  # 噪声强度

# 定义哈密顿量的构建函数
def build_hamiltonian(m, g):
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

# 时间演化函数（含量子电路构建）
def time_evolution(hamiltonian, initial_state, steps, interval, noise_strength=0.02):
    states = []
    backend = Aer.get_backend('statevector_simulator')
    evolution_circuit = QuantumCircuit(num_qubits)

    for i, state in enumerate(initial_state):
        if state == '1':
            evolution_circuit.x(i)

    U = scipy.linalg.expm(-1j * hamiltonian * interval)
    evolution_operator = Operator(U)

    for _ in range(steps):
        temp_circuit = evolution_circuit.copy()
        temp_circuit.append(evolution_operator, range(num_qubits))

        # 添加噪声：对每个量子比特应用小的随机旋转
        for qubit in range(num_qubits):
            temp_circuit.rx(noise_strength * np.random.randn(), qubit)
            temp_circuit.ry(noise_strength * np.random.randn(), qubit)
            temp_circuit.rz(noise_strength * np.random.randn(), qubit)

        transpiled_circuit = transpile(temp_circuit, backend)
        job = execute(transpiled_circuit, backend)
        result = job.result()
        state = Statevector(result.get_statevector())
        states.append(state)
        evolution_circuit = temp_circuit

    return states

# 绘图函数
def plot_graphs(states_list, hamiltonians, masses, couplings):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))
    # 对每个子图应用相同的横纵比例

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

    #绘制保真度的演化图
    for states, m, g in zip(states_list, masses, couplings):
        time_points = np.arange(len(states))
        initial_state = states[0]
        fidelities = [state_fidelity(initial_state, state) for state in states]
        # 使用插值使曲线更光滑
        f_interp = interp1d(time_points, fidelities, kind='cubic')
        smooth_time_points = np.linspace(time_points.min(), time_points.max(), 500)
        smooth_fidelities = f_interp(smooth_time_points)
        axs[4].plot(smooth_time_points, smooth_fidelities, linestyle='-.', linewidth=1, label=f'm={m}, g={g}')
        axs[4].scatter(time_points, fidelities, s=10)  # 添加原始数据点
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Fidelity')
    axs[4].set_title('Fidelity Evolution')
    axs[4].legend()

    plt.tight_layout()
    plt.show()

# 主程序
def main():
    states_list = []
    hamiltonians = []
    for m in masses:
        for g in couplings:
            hamiltonian = build_hamiltonian(m, g)
            hamiltonians.append(hamiltonian)
            initial_state = '0' * num_qubits
            states = time_evolution(hamiltonian, initial_state, time_steps, dt, noise_strength)
            states_list.append(states)
    plot_graphs(states_list, hamiltonians, masses, couplings)

main()
