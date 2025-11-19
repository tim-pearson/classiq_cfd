import os

from classiq import (
    ClassiqBackendPreferences,
)
from classiq.applications.hamiltonian.pauli_decomposition import hamiltonian_to_matrix, matrix_to_pauli_operator
import numpy as np
from ansatz import ansatz_4_balanced, ansatz_4_compact, ansatz_4_hardware
from vqls import Vqls
from dotenv import load_dotenv
from utils import (
    create_poisson_and_guaranteed_b,
    incomplete_cholesky_pc,
    test_ansatz_expressibility,
)

# %%
backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
# %%
N = 4
A, b = create_poisson_and_guaranteed_b(N)
A_pre, b_pre, M_inv, pauli = incomplete_cholesky_pc(A, b, 1e-3)
A_p = hamiltonian_to_matrix(pauli)
x_pre = np.linalg.solve(A_pre, b_pre)
print(len(pauli.terms))

# %%

target_solution_precond = x_pre / np.linalg.norm(x_pre)

fidelity, best_params = test_ansatz_expressibility(
    target_solution_precond,
    ansatz_4_compact,  
    8, max_iterations=100

)
print("for solution x = ", target_solution_precond)
# %%

# Normalized versions for comparison
x_classical_normalized = x_pre / np.linalg.norm(x_pre)


# VQLS setup
ansatz_param_count = 8
num_system_qubits = pauli.num_qubits
# %%
vqls = Vqls(ansatz_param_count, pauli, b_pre, ansatz_4_compact)

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

# %%
# Process results
df = vqls.results.dataframe
amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude
# %%
quantum_state = amplitudes / np.linalg.norm(amplitudes)
print(x_pre / np.linalg.norm(x_pre))
# %%


A_x_vqls = A_pre @ quantum_state

A_x_vqls_normalized = A_x_vqls / np.linalg.norm(A_x_vqls)
b_normalized = b_pre / np.linalg.norm(b_pre)

vqls_cost = np.linalg.norm(A_x_vqls_normalized - b_normalized)**2

overlap = np.abs(b_normalized @ A_x_vqls_normalized)**2

overlap
# %%
print("\n" + "=" * 50)
print("CORRECTED VQLS PERFORMANCE")
print("=" * 50)
print(f"VQLS cost function: {vqls_cost:.6f}")
print(f"Overlap |⟨b|A|x⟩|²: {overlap:.6f}")
# print(f"Expected cost from optimizer: {optimal_params.fun:.6f}")

# For comparison with classical solution, we need to find the scaling factor
# Since A|x_vqls⟩ ∝ b, find the best scaling factor α that minimizes ||α·A|x_vqls⟩ - b||
alpha = (b_pre @ A_x_vqls) / (A_x_vqls @ A_x_vqls)
x_vqls_scaled = alpha * quantum_state

print(f"\nBest scaling factor α: {alpha:.6f}")

# Now compare with classical solution
print("\n" + "=" * 50)
print("SOLUTION COMPARISON")
print("=" * 50)
print("Classical solution x:")
print(np.array2string(x_pre, precision=4, suppress_small=True))
print("\nVQLS solution (scaled) x:")
print(np.array2string(x_vqls_scaled, precision=4, suppress_small=True))

# Calculate relative error
relative_error = np.linalg.norm(x_vqls_scaled - x_pre) / np.linalg.norm(x_pre)
print(f"\nRelative error: {relative_error:.6f}")

# Verify both solutions satisfy the equation
residual_classical = np.linalg.norm(A_pre @ x_pre - b_pre)
residual_vqls = np.linalg.norm(A_pre @ x_vqls_scaled - b_pre)

print(f"\nResidual ||A@x - b||:")
print(f"Classical: {residual_classical:.6e}")
print(f"VQLS:      {residual_vqls:.6e}")

# Check if VQLS solution is proportional to classical solution
cosine_similarity = np.abs(x_pre @ x_vqls_scaled) / (np.linalg.norm(x_pre) * np.linalg.norm(x_vqls_scaled))
print(f"Cosine similarity between solutions: {cosine_similarity:.6f}")
# %%
# Instead of overlap between x vectors, measure what VQLS actually optimizes:

def verify_vqls_success(amplitudes, A_pre, b_pre):
    """Verify what VQLS actually achieved"""
    quantum_state = amplitudes / np.linalg.norm(amplitudes)
    A_x = A_pre @ quantum_state
    
    # What VQLS optimizes:
    A_x_norm = A_x / np.linalg.norm(A_x)
    b_norm = b_pre / np.linalg.norm(b_pre)
    vqls_overlap = np.abs(b_norm @ A_x_norm)**2
    vqls_cost = (1 - vqls_overlap) / 2
    
    print(f"VQLS metric - |⟨b|A|x⟩|²: {vqls_overlap:.4f}")
    print(f"VQLS metric - cost: {vqls_cost:.4f}")
    # print(f"Optimizer reported cost: {vqls.optimizer.intermediate.values(-1)}")
    
    # Now find the actual solution by scaling
    alpha = (b_pre @ A_x) / (A_x @ A_x)
    x_vqls_scaled = alpha * quantum_state
    
    # Compare with classical solution
    classical_overlap = np.abs(x_pre @ x_vqls_scaled)**2 / (
        np.linalg.norm(x_pre)**2 * np.linalg.norm(x_vqls_scaled)**2
    )
    print(f"Solution overlap |⟨x_classical|x_vqls⟩|²: {classical_overlap:.4f}")
    print(
        "overlap =",
        (b.dot(A_pre.dot(amplitudes) / (np.linalg.norm(A_pre.dot(amplitudes))))) ** 2,
    )
    
    return vqls_overlap, classical_overlap

# Run it:
vqls_overlap, classical_overlap = verify_vqls_success(amplitudes, A_pre, b_pre)
# %%
