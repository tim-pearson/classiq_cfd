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
from ansatz import ansatz_4_hardware
from optimizer import VqlsOptimizer
from vqls import Vqls
from dotenv import load_dotenv
from utils import (
    create_poisson_matrix_pauli,
    create_preconditioned_poisson_system,
    generate_guaranteed_b,
    laplacian_2d,
    genrate_random_b,
    normalize,
    test_ansatz_expressibility,
    verify_linear_system,
)

# %%

load_dotenv()
tk = os.environ["IBMQ_API_KEY"]

backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")

# %%
# Base Poisson System
N = 4
p, A_base = create_poisson_matrix_pauli(N)
b_base = generate_guaranteed_b(A_base, seed=2)

x_classical, x_classical_normalized, error = verify_linear_system(
    A_base, b_base, "Original Poisson System"
)
# %%
n_qubits = 4  # 16x16 system
preconditioner_type = "incomplete_cholesky"
max = -1
max_i = 0
(
    pauli_operator,
    A_precond,
    M_inv,
    A_original,
    b_precond,
    b_original,
    improvement,
    x_precond,
) = create_preconditioned_poisson_system(n_qubits, preconditioner_type)

# %%
target_solution_precond = x_precond / np.linalg.norm(x_precond)

fidelity, best_params = test_ansatz_expressibility(
    target_solution_precond,
    ansatz_4_hardware,  
    24, max_iterations=100

)
fidelity
# %%


# Normalized versions for comparison
x_classical_normalized = x_classical / np.linalg.norm(x_classical)


# VQLS setup
ansatz_param_count = 8
num_system_qubits = p.num_qubits

# %%
vqls = Vqls(ansatz_param_count, p, b_base, "A")

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
    (b_base.dot(A_base.dot(amplitudes) / (np.linalg.norm(A_base.dot(amplitudes)))))
    ** 2,
)

print(f"VQLS amplitudes: {amplitudes}")

# Normalize VQLS solution for comparison
x_vqls_normalized = amplitudes / np.linalg.norm(amplitudes)

print("\n" + "=" * 50)
print("COMPARISON: VQLS vs Classical")
print("=" * 50)

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
A_x_vqls = A_base.dot(amplitudes)
A_x_vqls_norm = A_x_vqls / np.linalg.norm(A_x_vqls)
actual_cost = np.linalg.norm(A_x_vqls_norm - b_base) ** 2

print(f"\nVQLS achieved cost: {actual_cost:.6f}")
print(f"b · (A|x_vqls⟩/||A|x_vqls⟩||): {np.abs(b_base.dot(A_x_vqls_norm)):.6f}")

print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)
print(f"A @ x_classical: {A_base.dot(x_classical)}")
print(f"b:               {b_base}")
print(f"Match: {np.allclose(A_base.dot(x_classical), b_base, atol=1e-10)}")
