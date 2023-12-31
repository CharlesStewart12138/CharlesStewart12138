from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rz(pi / 3, qreg_q[0])
circuit.x(qreg_q[1])
circuit.cz(qreg_q[1], qreg_q[2])
circuit.x(qreg_q[0])
circuit.rx(pi, qreg_q[1])
circuit.ry(pi, qreg_q[2])
circuit.cz(qreg_q[1], qreg_q[2])
circuit.rz(pi / 3, qreg_q[2])
circuit.rz(2 * pi / 3, qreg_q[1])
circuit.x(qreg_q[2])
circuit.cz(qreg_q[1], qreg_q[0])
circuit.rx(pi, qreg_q[0])
circuit.ry(pi, qreg_q[1])
circuit.cz(qreg_q[0], qreg_q[1])
circuit.x(qreg_q[1])
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.x(qreg_q[0])
circuit.cry(pi, qreg_q[1], qreg_q[2])
circuit.rx(pi, qreg_q[1])
circuit.ry(pi, qreg_q[2])
circuit.cry(pi / 2, qreg_q[1], qreg_q[2])
circuit.x(qreg_q[2])
circuit.x(qreg_q[1])
circuit.cz(qreg_q[1], qreg_q[2])
circuit.rx(pi, qreg_q[1])
circuit.ry(pi, qreg_q[2])
circuit.cz(qreg_q[1], qreg_q[2])
circuit.x(qreg_q[2])
circuit.cry(pi, qreg_q[0], qreg_q[1])
circuit.ry(pi, qreg_q[1])
circuit.rx(pi, qreg_q[0])
circuit.x(qreg_q[2])
circuit.cry(pi / 2, qreg_q[0], qreg_q[1])
circuit.ry(pi, qreg_q[2])
circuit.x(qreg_q[0])
circuit.cz(qreg_q[0], qreg_q[1])
circuit.rx(pi, qreg_q[1])
circuit.cz(qreg_q[0], qreg_q[1])
circuit.x(qreg_q[1])
