from classiq.applications.hamiltonian.pauli_decomposition import matrix_to_pauli_operator
import numpy as np
import scipy as sp
from scipy.sparse import diags, eye, kron

def create_poisson_and_guaranteed_b(N):
    """Create 2D Poisson matrix and guaranteed valid right-hand side vector b"""

    # 1D Poisson matrix in x direction (size N)
    main_diag_x = 2.0 * np.ones(N)
    off_diag_x = -1.0 * np.ones(N - 1)
    Ax = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(N, N))

    # 1D Poisson matrix in y direction (size N)
    main_diag_y = 2.0 * np.ones(N)
    off_diag_y = -1.0 * np.ones(N - 1)
    Ay = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(N, N))

    # 2D Poisson matrix: A = Iy ⊗ Ax + Ay ⊗ Ix
    Ix = eye(N)
    Iy = eye(N)
    A = kron(Iy, Ax) + kron(Ay, Ix)
    A = A.toarray().astype(np.float64)  # Convert to dense float array

    # Generate guaranteed valid b
    # Method 1: Try random vectors
    for _ in range(10):
        x_random = np.random.randn(A.shape[1])
        b = A @ x_random
        if np.linalg.norm(b) > 1e-12:
            b = b / np.linalg.norm(b)  # Normalize
            return A, b

    # Method 2: If random fails, use SVD to find column space
    try:
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        rank = np.sum(s > 1e-12)
        if rank > 0:
            coeffs = np.random.randn(rank)
            b = U[:, :rank] @ coeffs
            b = b / np.linalg.norm(b)
            return A, b
    except:
        pass

    # Method 3: Use first non-zero column
    for j in range(A.shape[1]):
        b = A[:, j].copy()
        if np.linalg.norm(b) > 1e-12:
            b = b / np.linalg.norm(b)
            return A, b

    # Final fallback: use ones vector (should work for Poisson)
    b = np.ones(A.shape[0])
    b = b / np.linalg.norm(b)
    return A, b


def incomplete_cholesky_pc(A, b, ptol):
    # Simple IC that maintains symmetry
    L = np.zeros_like(A)
    n = A.shape[0]

    for i in range(n):
        # Diagonal
        L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i] ** 2))
        # Off-diagonal (only for original non-zero pattern)
        for j in range(i + 1, n):
            if abs(A[j, i]) > 1e-10:  # Original sparsity
                L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i])) / L[i, i]

    # Now L @ L.T ≈ A exactly for the sparsity pattern
    M_inv = np.linalg.inv(L)
    A_preconditioned = M_inv @ A @ M_inv.T  # This stays perfectly Hermitian
    b_preconditioned = M_inv @ b

    pauli_operator = matrix_to_pauli_operator(A_preconditioned)
    return A_preconditioned, b_preconditioned, M_inv, pauli_operator



def verify_linear_system(A, b, description="System", debug=False, error=1e-14):
    """
    Verify linear system solution and print diagnostics
"""
    if debug:
        print(f"\n{'='*50}")
        print(f"VERIFICATION: {description}")
        print(f"{'='*50}")

        print(f"A shape: {A.shape}")
        print(f"A condition number: {np.linalg.cond(A):.4f}")
        print(f"A Hermitian: {np.allclose(A, A.T)}")

        print(f"\nb vector: {b}")

    # Classical solution
    A_inv = np.linalg.inv(A)
    x_classical = A_inv @ b
    x_classical = x_classical.real

    # Verification
    Ax_classical = A @ x_classical
    verification_error = np.linalg.norm(Ax_classical - b)
    x_classical_normalized = x_classical / np.linalg.norm(x_classical)
    if debug:
        print(f"\nClassical solution: {x_classical}")
        print(f"A @ x_classical: {Ax_classical}")
        print(f"Should equal b:   {b}")
        print(f"Verification error: {verification_error:.2e}")
        print(f"Normalized solution: {x_classical_normalized}")

    assert verification_error <= error

    return x_classical, x_classical_normalized, verification_error
