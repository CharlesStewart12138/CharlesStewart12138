from __future__ import annotations
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import *
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
import numpy as np
import networkx as nx

import numpy as np

Q = np.array([
    [ 0.,  1., -2.,  0.,  1.5],
    [ 1.,  0.,  1., -1., -1.5],
    [-2.,  1.,  0.,  1.,  0.5],
    [ 0., -1.,  1.,  0., -2.5],
    [ 1.5, -1.5, 0.5, -2.5, 0.]
])

from qiskit.quantum_info import Pauli, SparsePauliOp

def objective_value_qubo(x: np.ndarray, Q: np.ndarray) -> float:
    """Compute the objective value for a QUBO problem.
    
    Args:
        x: Binary string as numpy array.
        Q: QUBO matrix.
        
    Returns:
        Objective value of the QUBO problem for the given x.
    """
    # 计算x^T Q x
    value = np.dot(x, np.dot(Q, x))
    return value

def get_qubo_operator(Q: np.ndarray) -> tuple[SparsePauliOp, float]:
    """Generate Hamiltonian for a QUBO model.
    
    Args:
        Q: The QUBO matrix defining the optimization problem.
        
    Returns:
        Operator for the Hamiltonian and a constant shift for the objective function.
    """
    num_variables = len(Q)
    pauli_list = []
    coeffs = []
    shift = 0.0

    for i in range(num_variables):
        for j in range(i, num_variables):
            if Q[i, j] != 0:
                x_p = np.zeros(num_variables, dtype=bool)  # No X Paulis for QUBO
                z_p = np.zeros(num_variables, dtype=bool)
                if i == j:
                    # Diagonal elements: contribute a Z term
                    z_p[i] = True
                    coeffs.append(Q[i, j])
                else:
                    # Off-diagonal elements: contribute a ZZ term
                    z_p[i] = True
                    z_p[j] = True
                    coeffs.append(Q[i, j])
                    shift += Q[i, j]  # Adjust shift for off-diagonal elements

                pauli_list.append(Pauli((z_p, x_p)))

    return SparsePauliOp(pauli_list, coeffs=coeffs), shift

qubit_op, offset = get_qubo_operator(Q)

def bitfield(n: int, L: int) -> list[int]:
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli, Statevector
from qiskit.result import QuasiDistribution
from qiskit_algorithms.utils import algorithm_globals

sampler = Sampler()

def sample_most_likely(state_vector: QuasiDistribution | Statevector) -> np.ndarray:
    """Compute the most likely binary string from state vector.
    Args:
        state_vector: State vector or quasi-distribution.

    Returns:
        Binary string as an array of ints.
    """
    if isinstance(state_vector, QuasiDistribution):
        values = list(state_vector.values())
    else:
        values = state_vector
    n = int(np.log2(len(values)))
    k = np.argmax(np.abs(values))
    x = bitfield(k, n)
    x.reverse()
    return np.asarray(x)

algorithm_globals.random_seed = 10598

optimizer = COBYLA()
qaoa = QAOA(sampler, optimizer, reps=2)

result = qaoa.compute_minimum_eigenvalue(qubit_op)

x = sample_most_likely(result.eigenstate)

print(x)
print(f'Objective value computed by QAOA is {objective_value_qubo(x, Q)}')
