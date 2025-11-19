# %%
from random import random
import numpy as np
from scipy.sparse import diags
from scipy.linalg import inv
import numpy as np
from scipy.sparse import diags, kron, eye


def create_poisson_1d():
    """Create ORIGINAL 4x4 1D Poisson matrix for 2 qubits"""
    N = 4
    main_diag = 2.0 * np.ones(N)
    off_diag = -1.0 * np.ones(N - 1)
    A = diags(
        [off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)
    ).toarray()
    return A


def create_poisson_2d():
    N = 4

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

    return A.toarray()


# For the 2×2 case (4 points total):


def is_hermitian(matrix, tol=1e-10):
    """Check if matrix is Hermitian (symmetric for real matrices)"""
    return np.allclose(matrix, matrix.T.conj(), atol=tol)


# %%
# Create original matrix
# A_original = create_poisson_2D()
A_original = create_poisson_2d()

print("ORIGINAL POISSON MATRIX (4x4):")
print(A_original)
cond_original = np.linalg.cond(A_original)
print(f"Original condition number: {cond_original:.6f}")
print(f"Hermitian: {is_hermitian(A_original)}")

# %%
# Apply Incomplete Cholesky preconditioning
print("\n" + "=" * 50)
print("INCOMPLETE CHOLESKY PRECONDITIONING")
print("=" * 50)

L = np.tril(A_original)  # Lower triangular part
M_inv = inv(L)  # Preconditioner inverse

# Symmetric preconditioning
A_preconditioned = M_inv @ A_original @ M_inv.T
cond_preconditioned = np.linalg.cond(A_preconditioned)

print("Preconditioned matrix:")
print(A_preconditioned)
print(f"\nCondition number: {cond_preconditioned:.6f}")
print(f"Improvement: {cond_original/cond_preconditioned:.2f}x")
print(f"Hermitian: {is_hermitian(A_preconditioned)}")
# %%

# Verify solution preservation
print("\n" + "=" * 50)
print("SOLUTION PRESERVATION VERIFICATION")
print("=" * 50)

b_test = [float(np.random.randint(-314, 314)) / 1000 for _ in range(16)]


# Solve original system
x_original = np.linalg.solve(A_original, b_test)

# Solve preconditioned system
b_precond = M_inv @ b_test
x_new = np.linalg.solve(A_preconditioned, b_precond)

# Recover original solution
x_recovered = M_inv.T @ x_new

print(f"Original solution:    {x_original}")
print(f"Recovered solution:   {x_recovered}")
print(f"Solutions match: {np.allclose(x_original, x_recovered)}")
print(f"Difference norm: {np.linalg.norm(x_original - x_recovered):.2e}")

# Verify both satisfy original equation
residual_original = np.linalg.norm(A_original @ x_original - b_test)
residual_recovered = np.linalg.norm(A_original @ x_recovered - b_test)

print(f"\nResidual (original):    {residual_original:.2e}")
print(f"Residual (recovered):   {residual_recovered:.2e}")
print(
    f"Both satisfy A x = b: {residual_original < 1e-10 and residual_recovered < 1e-10}"
)

# %%
# Final assessment
print("\n" + "=" * 50)
print("FINAL ASSESSMENT")
print("=" * 50)

hermitian_ok = is_hermitian(A_preconditioned)
solution_ok = np.allclose(x_original, x_recovered)
improvement_ok = cond_original / cond_preconditioned > 1.1

print(f"✅ Hermitian: {hermitian_ok}")
print(f"✅ Solution preserved: {solution_ok}")
print(
    f"✅ Good improvement ({cond_original/cond_preconditioned:.2f}x): {improvement_ok}"
)

if hermitian_ok and solution_ok and improvement_ok:
    print("\n🎯 INCOMPLETE CHOLESKY IS SUITABLE FOR VQLS!")
else:
    print("\n❌ INCOMPLETE CHOLESKY HAS ISSUES FOR VQLS")


# %%
import numpy as np
from scipy.sparse import diags, kron, eye

def create_poisson_and_guaranteed_b():
    """Create 2D Poisson matrix and guaranteed valid right-hand side vector b"""
    N = 4

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
    for attempt in range(10):
        x_random = np.random.randn(A.shape[1])
        b = A @ x_random
        if np.linalg.norm(b) > 1e-12:
            b = b / np.linalg.norm(b)  # Normalize
            return A, b
    
    # Method 2: If random fails, use SVD to find column space
    try:
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
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

# Usage:
A, b = create_poisson_and_guaranteed_b()
print("Matrix A shape:", A.shape)
print("Vector b shape:", b.shape)
print("Norm of b:", np.linalg.norm(b))

