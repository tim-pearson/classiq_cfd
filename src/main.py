import os
import classiq
# classiq.authenticate(overwrite=True)

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

backend_preferences = ClassiqBackendPreferences(
    backend_name="simulator_statevector"
)

# %%
p, A_num = create_poisson_matrix_pauli(2)
print("Poisson matrix:")
print(A_num)

print(f"Paulis: {p}")

b = generate_guaranteed_b(A_num, seed=32)
print(f"b vector: {b}")

# Classical solution
A_inv = np.linalg.inv(A_num)
x_classical = np.dot(A_inv, b.T)
x_classical = x_classical.real
print(f"Classical solution: {x_classical}")

# Verify classical solution
Ax_classical = A_num.dot(x_classical)
print(f"A @ x_classical: {Ax_classical}")
print(f"Should equal b:  {b}")
print(f"Verification error: {np.linalg.norm(Ax_classical - b):.2e}")
# %%

# Normalized versions for comparison
x_classical_normalized = x_classical / np.linalg.norm(x_classical)

# VQLS setup
ansatz_param_count = 7
num_system_qubits = p.num_qubits

# %%
vqls = Vqls(ansatz_param_count, p, b, "A")

print("Creating quantum program...")
vqls.create_qrog()

print("Initializing optimizer...")
vqls.init_optimizer(204800, backend_preferences=backend_preferences)

print("Optimizing...")
optimal_params = vqls.optimizer.optimize()
# %%
print("Evaluating ansatz...")
vqls.evaluate_ansatz(optimal_params)

# Process results
df = vqls.results.dataframe
amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude

# Phase correction
global_phase = np.angle(amplitudes[-1])
amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
if amplitudes[-1] < 0:
    amplitudes *= -1

print(
    "CLASSIQS Overlap",
    (b.dot(A_num.dot(amplitudes) / (np.linalg.norm(A_num.dot(amplitudes))))) ** 2,
)

print(f"VQLS amplitudes: {amplitudes}")

# Normalize VQLS solution for comparison
x_vqls_normalized = amplitudes / np.linalg.norm(amplitudes)

print("\n" + "="*50)
print("COMPARISON: VQLS vs Classical")
print("="*50)

print("Classical x (normalized):")
print(np.array2string(x_classical_normalized, precision=4, suppress_small=True))

print("\nVQLS x (normalized):")
print(np.array2string(x_vqls_normalized, precision=4, suppress_small=True))

# Calculate metrics using NORMALIZED vectors
error = np.linalg.norm(x_classical_normalized - x_vqls_normalized)
overlap = np.abs(x_classical_normalized.dot(x_vqls_normalized)) ** 2

print(f"\nL2 Error: {error:.4f}")
print(f"Overlap (fidelity): {overlap:.4f}")

# Check what VQLS actually achieved
A_x_vqls = A_num.dot(amplitudes)
A_x_vqls_norm = A_x_vqls / np.linalg.norm(A_x_vqls)
actual_cost = np.linalg.norm(A_x_vqls_norm - b)**2

print(f"\nVQLS achieved cost: {actual_cost:.6f}")
print(f"b · (A|x_vqls⟩/||A|x_vqls⟩||): {np.abs(b.dot(A_x_vqls_norm)):.6f}")

print("\n" + "="*50)
print("VERIFICATION")
print("="*50)
print(f"A @ x_classical: {A_num.dot(x_classical)}")
print(f"b:               {b}")
print(f"Match: {np.allclose(A_num.dot(x_classical), b, atol=1e-10)}")

# %%

print("A @ x_vqls:", A_num.dot(x_vqls_normalized))
print("Normalized:", A_num.dot(x_vqls_normalized)/np.linalg.norm(A_num.dot(x_vqls_normalized)))
print("b:", b)
