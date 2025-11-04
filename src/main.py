import os

from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
    matrix_to_hamiltonian,
    pauli_operator_to_matrix,
)
import matplotlib.pyplot as plt
from classiq import (
    ClassiqBackendPreferences,
    IBMBackendPreferences,
    Pauli,
    SparsePauliOp,
)
import numpy as np
from classiq import ClassiqBackendPreferences, ClassiqNvidiaBackendNames
from pandas.io.formats.style import plt
from optimizer import VqlsOptimizer
from vqls import Vqls
from dotenv import load_dotenv
from utils import laplacian_2d, genrate_random_b

# %%

load_dotenv()
tk = os.environ["IBMQ_API_KEY"]


# be_name =get_ibm_backends(tk)[0].name
# print(be_name)
# backend_preferences = IBMBackendPreferences(
#     backend_name=be_name,
#     access_token=tk,
#     channel="ibm_quantum_platform",
#     instance_crn=crn,
# )


backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")



pauli_pressure = (
    0.5 * Pauli.I(0) * Pauli.I(1)  
    - 0.25 * Pauli.I(0) * Pauli.X(1)  
    - 0.25 * Pauli.X(0) * Pauli.I(1)  
)
# %%
pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)
normalization = sum([p.coefficient for p in pauli_terms_structs.terms])

num_system_qubits = pauli_terms_structs.num_qubits


# %%
p = pauli_terms_structs
A_num = pauli_operator_to_matrix(p) / normalization
A_inv = np.linalg.inv(A_num)
b = genrate_random_b(A_num)
x = np.dot(A_inv, b.T)
classical_probs = np.real((x / np.linalg.norm(x))) ** 2
# %%
ansatz_param_count = 9
vqls = Vqls(ansatz_param_count, p, b, input("chose test name"))
# %% 
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(204800, backend_preferences=backend_preferences)
print("optimizing")
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
# %%

df = vqls.results.dataframe
amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude
# Preprocessed quantum solution: we know the solution is real, and that the last point is positive
global_phase = np.angle(amplitudes[-1])
amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
if (
    amplitudes[-1] < 0
):  
    amplitudes *= -1
print(amplitudes)
probabilities = amplitudes**2
print(
    "overlap =",
    (b.dot(A_num.dot(amplitudes) / (np.linalg.norm(A_num.dot(amplitudes))))) ** 2,
)
