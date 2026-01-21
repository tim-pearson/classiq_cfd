import os
from classiq import ClassiqBackendPreferences
from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
    matrix_to_pauli_operator,
)
import numpy as np
from ansatz import (
    ansatz_4_balanced,
    ansatz_4_compact,
    ansatz_4_hardware,
    test_ansatz_expressibility,
)
from poisson import create_poisson_and_guaranteed_b, incomplete_cholesky_pc
from vqls import Vqls
from dotenv import load_dotenv

# %%

N = 4
A, b = create_poisson_and_guaranteed_b(N)
A_pauli = matrix_to_pauli_operator(A, tol=1e-3)
print(f"Original A Pauli terms: {len(A_pauli.terms)}")
print(A_pauli.terms)
A_pre, b_pre, M_inv, _ = incomplete_cholesky_pc(A, b, 1e-3)

x_classical = np.linalg.solve(A, b)

target_solution = x_classical / np.linalg.norm(x_classical)

# fidelity, best_params = test_ansatz_expressibility(
#     target_solution, ansatz_4_compact, 8, max_iterations=100
# )
# %%

ansatz_param_count = 8
num_system_qubits = A_pauli.num_qubits

vqls = Vqls(ansatz_param_count, A_pauli, b, ansatz_4_compact)

print("Creating quantum program...")
vqls.create_qrog()

print("Initializing optimizer...")

backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
vqls.init_optimizer(204800, backend_preferences=backend_preferences)


print("Optimizing...")
optimal_params = vqls.optimizer.optimize(M_inv)
# %%

df = vqls.results.dataframe
amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude

quantum_state_preconditioned = amplitudes / np.linalg.norm(amplitudes)
quantum_state_original = M_inv @ quantum_state_preconditioned
quantum_state_original = quantum_state_original / np.linalg.norm(quantum_state_original)

x_vqls_scaled = quantum_state_original * np.linalg.norm(x_classical)

print("\n" + "=" * 50)
print("VQLS WITH PRECONDITIONED COST FUNCTION")
print("=" * 50)

print(f"Original A Pauli terms: {len(A_pauli.terms)}")
print(f"Condition number original A: {np.linalg.cond(A):.4f}")
print(f"Condition number preconditioned A: {np.linalg.cond(A_pre):.4f}")

print("\nClassical solution x:")
print(np.array2string(x_classical, precision=4, suppress_small=True))
print("\nVQLS solution x:")
print(np.array2string(x_vqls_scaled, precision=4, suppress_small=True))

relative_error = np.linalg.norm(x_vqls_scaled - x_classical) / np.linalg.norm(
    x_classical
)
print(f"\nRelative error: {relative_error:.6f}")

residual_classical = np.linalg.norm(A @ x_classical - b)
residual_vqls = np.linalg.norm(A @ x_vqls_scaled - b)

print(f"\nResidual ||A@x - b||:")
print(f"Classical: {residual_classical:.6e}")
print(f"VQLS:      {residual_vqls:.6e}")

cosine_similarity = np.abs(x_classical @ x_vqls_scaled) / (
    np.linalg.norm(x_classical) * np.linalg.norm(x_vqls_scaled)
)
print(f"Cosine similarity between solutions: {cosine_similarity:.6f}")

final_cost = vqls.optimizer.my_cost(optimal_params)
print(f"Final preconditioned cost: {final_cost:.6e}")
