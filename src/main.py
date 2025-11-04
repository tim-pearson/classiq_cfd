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
from utils import create_poisson_matrix_pauli, generate_guaranteed_b, laplacian_2d, genrate_random_b, normalize

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


backend_preferences = ClassiqBackendPreferences(
    backend_name="simulator_statevector"
)


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


# %%
# p = pauli_pressure
# A_num = pauli_operator_to_matrix(p)
p, A_num = create_poisson_matrix_pauli(2)
print(A_num)
A_inv = np.linalg.inv(A_num)
b = generate_guaranteed_b(A_num)
x = np.dot(A_inv, b.T)
print("x = " , x )
classical_probs = np.real((x / np.linalg.norm(x))) ** 2
# %%
ansatz_param_count = 12
num_system_qubits = p.num_qubits
vqls = Vqls(ansatz_param_count, p, b, "blank")
# %%
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(204800, backend_preferences=backend_preferences)
print("optimizing")
# optimal_params = vqls.optimizer.optimize_with_better_settings()
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
# %%

df = vqls.results.dataframe
amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude
global_phase = np.angle(amplitudes[-1])
amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
if amplitudes[-1] < 0:
    amplitudes *= -1
print(amplitudes)
probabilities = amplitudes**2

print(
    "overlap =",
    (b.dot(A_num.dot(amplitudes) / (np.linalg.norm(A_num.dot(amplitudes)))))
    ** 2,
)
# %%

print("\n--- Comparison: VQLS vs Classical ---")

# Classical solution (exact)
x_classical = np.dot(A_inv, b.T)
x_classical = x_classical.real  # Take real part
x_classical /= np.linalg.norm(x_classical)  # Normalize
print(x_classical)
est = A_num.dot(amplitudes)
est = est / np.linalg.norm(est)
est
sol = A_num.dot(x_classical)
sol = sol / np.linalg.norm(sol)
np.real(sol)
print(np.real(b / np.linalg.norm(b)))

# VQLS solution (estimated)
x_vqls = amplitudes  # From your VQLS results

# Print both solutions
print("Classical x (exact):")
print(np.array2string(x_classical, precision=4, suppress_small=True))
print("\nVQLS x (estimated):")
print(np.array2string(x_vqls, precision=4, suppress_small=True))

# Calculate and print the error
error = np.linalg.norm(x_classical - x_vqls)
print(f"\nL2 Error: {error:.4f}")

# Calculate the overlap (fidelity)
overlap = np.abs(x_classical.dot(x_vqls)) ** 2
print(f"Overlap (fidelity): {overlap:.4f}")
# %%


