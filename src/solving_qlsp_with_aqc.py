# %%
import numpy as np
print("test")

# Define matrix A and vector b
A = np.array([[4, 1, 2, 0], [1, 3, 0, 1], [2, 0, 3, 1], [0, 1, 1, 2]])
b = np.array([12, 10, 17, 26])
# %%
import numpy as np


def compute_condition_number(A):
    """
    Computes the condition number (κ) of a matrix A.

    Parameters:
    A (numpy.ndarray): Input square matrix.

    Returns:
    float: The condition number of A.
    """
    try:
        # Compute the norm of A (2-norm, largest singular value)
        norm_A = np.linalg.norm(A, 2)

        # Compute the inverse of A
        A_inv = np.linalg.inv(A)

        # Compute the norm of A^-1 (2-norm)
        norm_A_inv = np.linalg.norm(A_inv, 2)

        # Compute the condition number κ
        condition_number = norm_A * norm_A_inv

        return condition_number
    except np.linalg.LinAlgError:
        return float("inf")  # Return infinity if the matrix is singular


condition_number = compute_condition_number(A)
print("Condition number:", condition_number)
# %%
from classiq import *
from classiq.execution import *


def setup_QLSP(A, b):
    # Normalize A
    norm_A = np.linalg.norm(A, "fro")
    A_normalized = A / norm_A

    # Normalize vector b
    b_normalized = b / np.linalg.norm(b)

    # Create the outer product of b
    outer_product_b = np.outer(b_normalized, b_normalized)

    # Define the identity matrix I with the same size as b
    identity_matrix = np.eye(len(b))

    # Compute Qb = I - outer_product_b
    Qb = identity_matrix - outer_product_b

    # Define the Pauli-X (σx) and Pauli-Y (σy) matrices
    pauli_x = np.array([[0, 1], [1, 0]])

    pauli_y = np.array([[0, -1j], [1j, 0]])

    # Define Pauli plus and minus operators
    pauli_plus = 0.5 * (pauli_x + 1j * pauli_y)
    pauli_minus = 0.5 * (pauli_x - 1j * pauli_y)

    # Compute the tensor product of Pauli-X and Qb
    H0 = np.kron(pauli_x, Qb)

    # Compute A*Qb and Qb*A
    A_Qb = np.dot(A, Qb)
    Qb_A = np.dot(Qb, A)

    # Compute the tensor products
    tensor_plus = np.kron(pauli_plus, A_Qb)
    tensor_minus = np.kron(pauli_minus, Qb_A)

    # Define H1 as the sum of the two tensor products
    H1 = tensor_plus + tensor_minus

    HO_HAMILTONIAN = matrix_to_hamiltonian(H0)
    H1_HAMILTONIAN = matrix_to_hamiltonian(H1)

    return H0, H1, HO_HAMILTONIAN, H1_HAMILTONIAN, A_normalized, b_normalized


# Setup

H0, H1, HO_HAMILTONIAN, H1_HAMILTONIAN, A_normalized, b_normalized = setup_QLSP(A, b)
# %%
# Define the time-dependent interpolated Hamiltonian, where T is the total evolution time


def hamiltonian_t(H0, H1, t, T):
    s = t / T
    return ((1 - s) * H0) + (s * H1)
# %%
import matplotlib.pyplot as plt


def plot_eigenvalues_evolution(ylim=None):
    time_steps = np.linspace(0, 1, 100)  # Discrete time steps

    # Store eigenvalues at each time step
    eigenvalues = []

    # Calculate eigenvalues across all time steps
    for t in time_steps:
        H_t = hamiltonian_t(H0, H1, t, 1)
        eigvals = np.linalg.eigvalsh(H_t)  # Sorted real eigenvalues
        eigenvalues.append(eigvals)

    # Convert eigenvalues list to a NumPy array for easier manipulation
    eigenvalues = np.array(eigenvalues)

    # Add small offsets to separate close eigenvalues visually
    offsets = np.linspace(
        -0.05, 0.05, eigenvalues.shape[1]
    )  # Small offsets for each eigenvalue line

    # Plot the eigenvalues across time steps
    plt.figure(figsize=(10, 6))
    for i in range(eigenvalues.shape[1]):
        plt.plot(time_steps, eigenvalues[:, i] + offsets[i], label=f"Eigenvalue {i+1}")

    # Highlight degenerate eigenvalues (if any)
    for step_idx, t in enumerate(time_steps):
        unique_vals, counts = np.unique(eigenvalues[step_idx], return_counts=True)

    # Apply y-axis limits if provided
    if ylim:
        plt.ylim(ylim)

    # Customize the plot
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Eigenvalues", fontsize=12)
    plt.title("Eigenvalues Evolution Across $s$", fontsize=14)
    plt.grid()

    # Move the legend to the side
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    # Show the plot
    plt.show()
# %%
plot_eigenvalues_evolution()
# %%
plot_eigenvalues_evolution(ylim=(-1.5, 1.5))
# %%
TOTAL_EVOLUTION_TIME = 7
# %%
NUM_STEPS = 50
# %%
@qfunc
def adiabatic_evolution_qfunc(
    H0: CArray[PauliTerm],
    H1: CArray[PauliTerm],
    evolution_time: int,
    num_steps: int,
    qba: QArray,
):
    # Time step for each increment
    delta_t = evolution_time / num_steps
    for step in range(num_steps):
        t = step * delta_t
        suzuki_trotter(
            H0,
            evolution_coefficient=delta_t * (1 - t / evolution_time),
            order=1,
            repetitions=1,
            qbv=qba,
        )
        suzuki_trotter(
            H1,
            evolution_coefficient=delta_t * (t / evolution_time),
            order=1,
            repetitions=10,
            qbv=qba,
        )
# %%
def get_model(H0, H1, b, evolution_time, num_steps):

    @qfunc
    def main(qba: Output[QArray]):
        prepare_state(
            probabilities=(np.abs(np.kron(np.array([1, 0]), b)) ** 2).tolist(),
            bound=0,
            out=qba,
        )
        adiabatic_evolution_qfunc(H0, H1, evolution_time, num_steps, qba)

    execution_preferences = ExecutionPreferences(
        num_shots=1,
        backend_preferences=ClassiqBackendPreferences(
            backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR
        ),
    )
    return create_model(main, execution_preferences=execution_preferences)


qmod_1 = get_model(
    HO_HAMILTONIAN, H1_HAMILTONIAN, b_normalized, TOTAL_EVOLUTION_TIME, NUM_STEPS
)
# %%
qprog_1 = synthesize(qmod_1)
write_qmod(qmod_1, "solving_qlsp_with_aqc", decimal_precision=5, symbolic_only=False)
show(qprog_1)

result_1_state_vector = execute(qprog_1).result_value().state_vector
# %%
def plot_state_probabilities(title, x, color="b"):
    # Ensure x is a numpy array and normalized
    x = np.array(x)

    # Calculate probabilities
    probabilities = np.abs(x) ** 2

    # Create labels for the states
    labels = [f"|{i}>" for i in range(len(x))]

    # Plot the probabilities
    plt.bar(labels, probabilities, color=color, alpha=0.7)
    plt.xlabel("States")
    plt.ylabel("Probabilities")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()


def compare_states(
    state1, state1_label, state2, state1_labe2, color1="gold", color2="b"
):
    # Plot a histogram of each state probabilities
    plot_state_probabilities(state1_label, state1, color1)
    plot_state_probabilities(state1_labe2, state2, color2)

    # Check the overlap between states
    overlap = np.abs(np.vdot(state1, state2)) ** 2
    print(f"Similarity of results: {overlap:.4f}")
# %%
# Print the solution vector x
x = np.linalg.solve(A_normalized, b_normalized)
print("Solution vector x:")
normalized_x = x / np.linalg.norm(x)
print(normalized_x)

# Convert dictionary values to complex numbers
print("State vector:")
state_vector = np.array([complex(value) for value in result_1_state_vector.values()])
print(state_vector)

compare_states(
    state_vector,
    "Quantum simulator state_vector",
    np.kron(np.array([1, 0]), normalized_x),
    "Classical solution to normalized_x",
)
# %%
print("Program depth:", qprog_1.transpiled_circuit.depth)
# %%
# Define matrix A and vector b
A = np.array(
    [
        [3.9270525, 1.06841123, 2.09661281, -0.10400811],
        [1.06841123, 2.93584295, -0.0906049, 1.09754032],
        [2.09661281, -0.0906049, 2.87204449, 1.13774997],
        [-0.10400811, 1.09754032, 1.13774997, 1.85170585],
    ]
)
b = np.array([12, 10, 17, 26])

condition_number = compute_condition_number(A)
print("Condition number:", condition_number)

H0, H1, HO_HAMILTONIAN, H1_HAMILTONIAN, A_normalized, b_normalized = setup_QLSP(A, b)
# %%
plot_eigenvalues_evolution(ylim=(-1.5, 1.5))
# %%
qmod_2 = get_model(
    HO_HAMILTONIAN, H1_HAMILTONIAN, b_normalized, TOTAL_EVOLUTION_TIME, NUM_STEPS
)
qprog_2 = synthesize(qmod_2)
show(qprog_2)
result_2_state_vector = execute(qprog_2).result_value().state_vector
# %%
# Print the solution vector x
x = np.linalg.solve(A_normalized, b_normalized)
print("Solution vector x:")
normalized_x = x / np.linalg.norm(x)
print(normalized_x)

# Convert dictionary values to complex numbers
print("State vector:")
state_vector = np.array([complex(value) for value in result_2_state_vector.values()])
print(state_vector)

compare_states(
    state_vector,
    "Quantum simulator state_vector",
    np.kron(np.array([1, 0]), normalized_x),
    "Classical solution to normalized_x",
)
# %%
print("Program depth:", qprog_2.transpiled_circuit.depth)
# %%
TOTAL_EVOLUTION_TIME = 10
NUM_STEPS = 100
# %%
qmod_3 = get_model(
    HO_HAMILTONIAN, H1_HAMILTONIAN, b_normalized, TOTAL_EVOLUTION_TIME, NUM_STEPS
)
qprog_3 = synthesize(qmod_3)
show(qprog_3)
result_3_state_vector = execute(qprog_3).result_value().state_vector
# %%
# Print the solution vector x
x = np.linalg.solve(A_normalized, b_normalized)
print("Solution vector x:")
normalized_x = x / np.linalg.norm(x)
print(normalized_x)

# Convert dictionary values to complex numbers
print("State vector:")
state_vector = np.array([complex(value) for value in result_3_state_vector.values()])
print(state_vector)

compare_states(
    state_vector,
    "Quantum simulator state_vector",
    np.kron(np.array([1, 0]), normalized_x),
    "Classical solution to normalized_x",
)
# %%
print("Program depth:", qprog_3.transpiled_circuit.depth)
