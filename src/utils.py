import json
import random
import os
from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
    matrix_to_hamiltonian,
    matrix_to_pauli_operator,
)
from classiq import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh,  pinvh
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron

from ansatz import ansatz_4_hardware


def save_stats_to_json(stats, filename="vqls_stats", folder="results"):
    """
    Save a dictionary of statistics to a JSON file.
"""
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
    filepath = os.path.join(folder, filename + ".json")
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {filepath}")


def plot_classical_vs_quantum(
    classical_probs,
    quantum_probs,
    name="Classical vs Quantum probabilities",
):
    """
    Plots classical and quantum probabilities side by side for comparison.
"""
    os.makedirs("data", exist_ok=True)

    N = len(classical_probs)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    x_indices = np.arange(N)

    # Classical probabilities (blue)
    ax.bar(
        x_indices - bar_width / 2,
        classical_probs,
        width=bar_width,
        color="#1f77b4",
        alpha=0.8,
        label="Classical",
    )

    # Quantum probabilities (orange)
    ax.bar(
        x_indices + bar_width / 2,
        quantum_probs,
        width=bar_width,
        color="#ff7f0e",
        alpha=0.8,
        label="Quantum",
    )

    # Add labels and title BEFORE saving
    ax.set_xlabel("Vector space basis")
    ax.set_ylabel("Probability")
    ax.set_title(name)
    ax.set_xticks(x_indices)
    ax.legend()
    plt.tight_layout()

    # Save AFTER setting everything up
    fig.savefig(f"data/{name}.png")
    print(f"✅ Saved figure as data/{name}.png")

    plt.show()


def show_save_results(folder="data"):
    """
    Reads all JSON result files from a folder and plots each result.
"""
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found.")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not json_files:
        print(f"⚠️ No JSON files found in '{folder}'.")
        return

    print(f"Found {len(json_files)} result file(s) in '{folder}':\n")

    for filename in json_files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"📊 {filename}")
        print(f"  Iterations: {data.get('iterations', 'N/A')}")
        print(f"  Overlap: {data.get('overlap', 'N/A'):.6f}")
        print(f"  MSE: {data.get('mse', 'N/A'):.6e}")
        print(f"  Cosine similarity: {data.get('cosine_similarity', 'N/A'):.6f}")
        print()

        # Extract probability vectors
        classical_probs = np.array(data.get("classical_probs", []))
        quantum_probs = np.array(data.get("quantum_probs", []))

        # Plot and save
        name = os.path.splitext(filename)[0]
        plot_classical_vs_quantum(classical_probs, quantum_probs, name=name)




def generate_guaranteed_b(A, seed=42):
    """Absolutely guaranteed to work for any non-zero matrix
"""
    np.random.seed(seed)

    # Method 1: Direct column space construction (most robust)
    x_random = np.random.randn(A.shape[1])
    b = A @ x_random

    # Check if we got zero vector (if A is zero matrix or x in null space)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        # Try a few more random vectors
        for attempt in range(10):
            x_random = np.random.randn(A.shape[1])
            b = A @ x_random
            b_norm = np.linalg.norm(b)
            if b_norm > 1e-12:
                break

        if b_norm < 1e-12:
            raise ValueError("Matrix A appears to be zero matrix!")

    b /= b_norm

    # Verify b is valid
    try:
        # Use least-squares to handle rank-deficient cases
        x_solution, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if residual.size > 0 and residual[0] > 1e-8:
            print(
                f"Warning: Large residual {residual[0]:.2e} - matrix may be ill-conditioned"
            )
    except:
        pass

    return b


def make_real_if_close(vec, tol=1e-8):
    """If imaginary parts are small compared to tol, return real part; otherwise
    return original.
"""
    if np.max(np.abs(np.imag(vec))) < tol:
        return np.real(vec)
    return vec


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def fidelity(u, v):
    """Fidelity between two normalized states (complex).
"""
    u = normalize(u)
    v = normalize(v)
    return np.abs(np.vdot(u, v)) ** 2


# %% Corrected get_solution (replace your existing method)
def get_solution_from_results(results, num_system_qubits):
    """
        results: object with .dataframe having columns 'io' (int index) and
    'amplitude' (complex)
        num_system_qubits: int
        returns: amplitude vector (complex) normalized
"""
    N = 2**num_system_qubits
    df = results.dataframe

    amplitudes = np.zeros(N, dtype=complex)
    # ensure ordering: df.io must map 0..N-1 (computational basis)
    amplitudes[df.io.values.astype(int)] = df.amplitude.values

    # remove a uniform global phase (align last nonzero element)
    # find an index with largest magnitude to avoid dividing by tiny value
    idx = np.argmax(np.abs(amplitudes))
    if np.abs(amplitudes[idx]) > 1e-12:
        global_phase = np.angle(amplitudes[idx])
        amplitudes = amplitudes * np.exp(-1j * global_phase)

    # If target is expected real-valued, allow small imag noise removal:
    amplitudes = make_real_if_close(amplitudes, tol=1e-7)

    # normalize and preserve sign/phase (no squaring!)
    amplitudes = normalize(amplitudes)

    # If you want to enforce a real convention (optional):
    # if np.max(np.abs(np.imag(amplitudes))) < 1e-7:
    #     amplitudes = np.real(amplitudes)

    return amplitudes


def create_poisson_matrix_pauli(n_qubits):
    """
    Create 1D Poisson matrix and convert to Pauli decomposition
    This is the most reliable method
"""
    import numpy as np
    from scipy.sparse import diags

    # Matrix size
    N = 2**n_qubits

    # Create 1D Poisson matrix
    main_diag = 2.0 * np.ones(N)
    off_diag = -1.0 * np.ones(N - 1)
    A_poisson = diags(
        [off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)
    ).toarray()

    print(f"Poisson matrix ({N}x{N}):")
    print(A_poisson)

    # Convert to Pauli operator
    pauli_operator = matrix_to_pauli_operator(A_poisson)

    return pauli_operator, A_poisson

def create_poisson_2d(nx, ny):
    """Create 2D Poisson matrix for nx * ny grid using Kronecker products
"""
    # 1D Poisson matrix in x direction (size nx)
    main_diag_x = 2.0 * np.ones(nx)
    off_diag_x = -1.0 * np.ones(nx-1)
    Ax = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(nx, nx))
    
    # 1D Poisson matrix in y direction (size ny)
    main_diag_y = 2.0 * np.ones(ny)
    off_diag_y = -1.0 * np.ones(ny-1)
    Ay = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(ny, ny))
    
    # 2D Poisson matrix: A = Iy ⊗ Ax + Ay ⊗ Ix
    Ix = eye(nx)
    Iy = eye(ny)
    
    # Use sparse Kronecker product and convert to dense array
    A = np.kron(Iy, Ax) + np.kron(Ay, Ix)
    
    return A

def create_poisson_and_guaranteed_b():
    """Create 2D Poisson matrix and guaranteed valid right-hand side vector b
"""
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

def create_preconditioned_poisson_system(
    n_qubits, preconditioner_type="incomplete_cholesky"
):
    N = 2**n_qubits

    # Create original system
    A_original = create_poisson_2d(N) 
    b_original = generate_guaranteed_b(A_original, seed=2)

    print("ORIGINAL SYSTEM:")
    x_classical, x_classical_norm, error = verify_linear_system(
        A_original, b_original, "Original Poisson"
    )

    if preconditioner_type == "incomplete_cholesky":
        L = np.tril(A_original)
        M_inv = np.linalg.inv(L)
        A_preconditioned = M_inv @ A_original @ M_inv.T  # Symmetric preconditioning
        b_preconditioned = M_inv @ b_original

    elif preconditioner_type == "jacobi":
        # Jacobi preconditioning
        D_inv = np.diag(1 / np.diag(A_original))
        A_preconditioned = D_inv @ A_original @ D_inv  # Symmetric
        M_inv = D_inv
        b_preconditioned = M_inv @ b_original

    elif preconditioner_type == "symmetric_jacobi":
        # Symmetric Jacobi preconditioning
        D_sqrt_inv = np.diag(1 / np.sqrt(np.diag(A_original)))
        A_preconditioned = D_sqrt_inv @ A_original @ D_sqrt_inv
        M_inv = D_sqrt_inv
        b_preconditioned = M_inv @ b_original

    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner_type}")

    print(f"\n{'='*60}")
    print(f"PRECONDITIONED SYSTEM ({preconditioner_type.upper()})")
    print(f"{'='*60}")

    # Verify preconditioned system
    x_precond, x_precond_norm, precond_error = verify_linear_system(
        A_preconditioned, b_preconditioned, "Preconditioned"
    )

    # Verify solution recovery
    x_recovered = M_inv.T @ x_precond
    recovery_error = np.linalg.norm(A_original @ x_recovered - b_original)
    print(f"\nSolution recovery error: {recovery_error:.2e}")

    # Condition number improvement
    cond_original = np.linalg.cond(A_original)
    cond_precond = np.linalg.cond(A_preconditioned)
    improvement = cond_original / cond_precond

    print(f"\nCondition number improvement: {improvement:.2f}x")
    print(f"Original: {cond_original:.4f} -> Preconditioned: {cond_precond:.4f}")
    print(f"Hermitian: {np.allclose(A_preconditioned, A_preconditioned.T)}")

    # Convert to Pauli operator
    pauli_operator = matrix_to_pauli_operator(A_preconditioned)

    return (
        pauli_operator,
        A_preconditioned,
        M_inv,
        A_original,
        b_preconditioned,
        b_original,
        improvement, x_precond
    )


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

        # Normalized version for quantum comparison
        print(f"Normalized solution: {x_classical_normalized}")

    assert verification_error <= error

    return x_classical, x_classical_normalized, verification_error

def test_ansatz_expressibility(
    target_solution, 
    ansatz_func,
    param_count, 
    max_iterations=100
):
    """
    Standalone function to test ansatz expressibility
    Uses the same setup as your working VQLS optimizer
"""
    # Normalize target solution
    target_solution = target_solution / np.linalg.norm(target_solution)
    
    # Use the same execution preferences as your working optimizer
    backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
    execution_preferences = ExecutionPreferences(num_shots=20480,backend_preferences=backend_preferences)
    
    intermediate_costs = []
    def cost_function(params):
        """Cost function: 1 - fidelity between ansatz output and target
"""
        
        @qfunc
        def main(io: Output[QNum[4]]):
            allocate(io)
            ansatz_func(list(params), io)
        
        # Synthesize and run circuit
        qprog = synthesize(main)
        with ExecutionSession(
            qprog, 
            execution_preferences=execution_preferences
        ) as es:
            results = es.sample()
        
        # Reconstruct output statevector
        df = results.dataframe
        output_state = np.zeros(2**4).astype(complex)
        output_state[df.io] = df.amplitude
        
        # Normalize output state
        output_state = output_state / np.linalg.norm(output_state)
        
        # Calculate infidelity
        fidelity = np.abs(np.vdot(target_solution, output_state))**2
        intermediate_costs.append(1 - fidelity)
        return 1 - fidelity
    
    # Run optimization with same setup as your working optimizer
    random.seed(1000)
    initial_params = [
        float(np.random.randint(-314, 314)) / 1000
        for _ in range(param_count)
    ]

    print(f"Testing {ansatz_func.__name__} with {param_count} parameters on 4 qubits...")
    print(f"Initial parameters: {initial_params}")

    result = minimize(
        cost_function, 
        x0=initial_params, 
        method="COBYLA",
        options={"maxiter": max_iterations}
    )
    
    print(result)
    
    best_fidelity = 1 - result.fun
    best_params = result.x
    
    print(f"Maximum achievable fidelity: {best_fidelity:.4f}")
    print(f"Optimization success: {result.success}")
    
    # Interpretation
    if best_fidelity > 0.9:
        print("✅ Ansatz is EXCELLENT for this solution")
    elif best_fidelity > 0.7:
        print("✅ Ansatz is GOOD for this solution") 
    elif best_fidelity > 0.5:
        print("⚠️  Ansatz is MARGINAL for this solution")
    else:
        print("❌ Ansatz is POOR for this solution")

    plt.plot(
        [l for l in range(len(intermediate_costs))],
        intermediate_costs,
    )
    plt.title("VQLS Incomplete Choleski Precondition 4-Q Ansatz expressibility")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    
    return best_fidelity, best_params
