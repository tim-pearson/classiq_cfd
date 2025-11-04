# %% Imports
from classiq import Pauli
from classiq.applications.hamiltonian.pauli_decomposition import hamiltonian_to_matrix
import numpy as np
from scipy.linalg import eigh, pinvh

# %% Define A matrix (2x2 pressure)

pauli_terms = (
    0.5 * Pauli.I(0) * Pauli.I(1)  
    - 0.25 * Pauli.I(0) * Pauli.X(1)  
    - 0.25 * Pauli.X(0) * Pauli.I(1)  
)

A = hamiltonian_to_matrix(pauli_terms)
print("A matrix:\n", A)

# %% Check A's properties
eigenvalues, eigenvectors = eigh(A)
print("\nEigenvalues:", eigenvalues)
print("Condition number:", np.linalg.cond(A))

# %% Define a list of b vectors to test
b_vectors = [
    np.ones(4) / np.sqrt(4),
    np.array([0.25, 0.001, 0.001, 0.45]),
    np.array([1, 0, 0, 1]),
    np.array([0, 1, 1, 0]),
    np.random.rand(4),
]
b_vectors = [b / np.linalg.norm(b) for b in b_vectors]


# %% Test each b vector (clean output)
for i, b in enumerate(b_vectors):
    print(f"\n--- Testing b vector {i+1} ---")
    print(f"b = {np.array2string(b, precision=4, suppress_small=True)}")

    try:
        x = np.linalg.solve(A, b)
        print(f"Solution exists:")
        print(f"x = {np.array2string(x.real, precision=4, suppress_small=True)}")
    except np.linalg.LinAlgError:
        print("No exact solution exists.")
        # Try pseudo-inverse for least-squares solution
        x_pinv = pinvh(A) @ b
        print(f"Least-squares solution:")
        print(f"x = {np.array2string(x_pinv.real, precision=4, suppress_small=True)}")
# %% Check if b is in the range of A
for i, b in enumerate(b_vectors):
    # Project b onto the range of A
    b_proj = A @ pinvh(A) @ b
    print(f"\n--- b vector {i+1} ---")
    print(f"Original b: {np.array2string(b, precision=4, suppress_small=True)}")
    print(f"Projected b: {np.array2string(b_proj.real, precision=4, suppress_small=True)}")
    print(f"Is b in range of A? {np.allclose(b, b_proj, atol=1e-6)}")

# %% Generate a b that is guaranteed to be solvable
# Use a random linear combination of A's eigenvectors
coeffs = np.random.rand(len(eigenvalues))
b_solvable = eigenvectors @ (coeffs * eigenvalues)
b_solvable /= np.linalg.norm(b_solvable)
print("\nGuaranteed solvable b:", b_solvable)
x_solvable = np.linalg.solve(A, b_solvable)
print("Solution x:", x_solvable)
# %%
def check_ansatz_expressibility(target_state, num_params=6):
    """Check if ansatz can approximate the target state"""
    print("\n=== Ansatz Expressibility Check ===")
    
    # For 2-qubit RY-only ansatz, the amplitudes have constraints:
    # The pattern is determined by tensor products of real rotations
    
    # Check if target has the right pattern for real-only ansatz
    target_norm = target_state / np.linalg.norm(target_state)
    
    print("Target state pattern:")
    for i, amp in enumerate(target_norm):
        print(f"|{i:02b}⟩: {amp:7.4f} {'✓' if abs(amp.imag) < 1e-10 else '✗ (complex)'}")
    
    # For RY-only ansatz, all amplitudes should be real
    if np.max(np.abs(target_norm.imag)) > 1e-10:
        print("WARNING: Target has complex amplitudes but RY ansatz produces only real!")
        print("Consider using RZ gates for phase control.")

# Add this before optimization:x
x=[-0.29128408,-0.8323651,0.13335101,0.45226036]
check_ansatz_expressibility(x)
