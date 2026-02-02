
from classiq import *
import numpy as np
# %%
import numpy as np

# For N=4 grid points, we need n=2 qubits (2^2=4 states)
N = 4  # Grid points
n = 2  # Qubits needed
dim = N  # Should be 4, NOT 16

# Classical Poisson matrix (4x4)
A_classical = np.array([
    [2, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 2]
])

# Now define L1, L2, L3 as 4x4 matrices matching the paper

# L1 = I ⊗ X (for n=2 qubits)
# This flips the LSB (qubit 0)
L1 = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

# L2 from paper pattern for n=2
# Based on equation (11): [1,0,1,1,0] pattern
L2 = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

# L3 = diagonal with alternating signs based on parity
L3 = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

Id = np.eye(4)
alpha = [2.5, -1, -1, -0.5]
A_hed = alpha[0]*Id + alpha[1]*L1 + alpha[2]*L2 + alpha[3]*L3

print("A_classical (4x4):")
print(A_classical)
print("\nA_hed (4x4):")
print(A_hed)
print(f"\nClose? {np.allclose(A_classical, A_hed)}")
print(f"\nMax difference: {np.max(np.abs(A_classical - A_hed))}")
# %%

@qfunc
def L1_circuit(system_qubits: QArray[QBit]):
    X(system_qubits[0])

@qfunc
def C1_circuit(system_qubits: QArray[QBit]):
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[1]))
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[1]))
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[1]))

@qfunc
def C2_circuit(system_qubits: QArray[QBit]):
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[2]))
    control(ctrl=system_qubits[1], stmt_block=lambda: X(system_qubits[2]))
    control(ctrl=system_qubits[0:2], stmt_block=lambda: X(system_qubits[2]))
    control(ctrl=system_qubits[1], stmt_block=lambda: X(system_qubits[2]))
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[2]))

@qfunc
def C3_circuit(system_qubits: QArray[QBit]):
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[1], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[2], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[0:3], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[2], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[1], stmt_block=lambda: X(system_qubits[3]))
    control(ctrl=system_qubits[0], stmt_block=lambda: X(system_qubits[3]))

@qfunc
def L2_circuit(system_qubits: QArray[QBit]):
    C1_circuit(system_qubits)
    C2_circuit(system_qubits)
    C3_circuit(system_qubits)

@qfunc
def L3_circuit(system_qubits: QArray[QBit]):
    target = system_qubits[0]
    H(target)
    if system_qubits.len > 1:
        control(ctrl=system_qubits[1:], stmt_block=lambda: X(target))
    H(target)
    X(target)
    H(target)
    if system_qubits.len > 1:
        control(ctrl=system_qubits[1:], stmt_block=lambda: X(target))
    H(target)
    X(target)

@qfunc
def prepare_hed_ancilla(ancilla_qubits: QArray[QBit]):
    alpha = [2.5, -1.0, -1.0, -0.5]
    alpha_mag = [abs(a) for a in alpha]
    total = sum(alpha_mag)
    probabilities = [a/total for a in alpha_mag]
    
    theta0 = 2 * np.arccos(np.sqrt(probabilities[0] + probabilities[1]))
    theta1 = 2 * np.arccos(np.sqrt(probabilities[0] / (probabilities[0] + probabilities[1]))) if probabilities[0] + probabilities[1] > 0 else 0
    theta2 = 2 * np.arccos(np.sqrt(probabilities[2] / (probabilities[2] + probabilities[3]))) if probabilities[2] + probabilities[3] > 0 else 0
    
    RY(theta0, ancilla_qubits[0])
    
    control(ctrl=ancilla_qubits[0], stmt_block=lambda: RY(theta1, ancilla_qubits[1]))
    X(ancilla_qubits[0])
    control(ctrl=ancilla_qubits[0], stmt_block=lambda: RY(theta2, ancilla_qubits[1]))
    X(ancilla_qubits[0])

@qfunc
def hed_linear_combination(system_qubits: QArray[QBit], ancilla_qubits: QArray[QBit]):
    prepare_hed_ancilla(ancilla_qubits)
    
    control(ctrl=[ancilla_qubits[0], ancilla_qubits[1]], stmt_block=lambda: L1_circuit(system_qubits))
    
    X(ancilla_qubits[1])
    control(ctrl=[ancilla_qubits[0], ancilla_qubits[1]], stmt_block=lambda: L2_circuit(system_qubits))
    X(ancilla_qubits[1])
    
    X(ancilla_qubits[0])
    control(ctrl=[ancilla_qubits[0], ancilla_qubits[1]], stmt_block=lambda: L3_circuit(system_qubits))
    X(ancilla_qubits[0])


@qfunc
def prepare_test_state(system_qubits: QArray[QBit]):
    for i in range(system_qubits.len):
        H(system_qubits[i])

@qfunc
def main(system_out: Output[QArray[QBit]], ancilla_out: Output[QArray[QBit]]):
    allocate(2, ancilla_out)
    allocate(n, system_out)
    prepare_test_state(system_out)
    hed_linear_combination(system_out, ancilla_out)

qprog = synthesize(main)


# %%
results =None
backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
execution_preferences = ExecutionPreferences(
            num_shots=1000, backend_preferences=backend_preferences
        )

with ExecutionSession(qprog, execution_preferences) as es:
            results = es.sample()
print(results)
# %%

# Verification functions
def extract_postselected_system_state(results, ancilla_bits=2, system_bits=n):
    """Extract system state when ancilla=|00⟩"""
    parsed = results.parsed_state_vector_states
    postselected_state = np.zeros(2**system_bits, dtype=complex)
    
    for full_state, amp in results.state_vector.items():
        ancilla_state = full_state[:ancilla_bits]
        system_state_str = full_state[ancilla_bits:ancilla_bits+system_bits]
        
        # Post-select on ancilla=|00⟩
        if ancilla_state == '00':
            idx = int(system_state_str, 2)
            postselected_state[idx] = amp
    
    # Normalize
    norm = np.linalg.norm(postselected_state)
    if norm > 0:
        postselected_state /= norm
    return postselected_state

# %%
# Create classical reference
N = 16
A = poisson_1d_matrix(N)
psi = np.array([0.25]*N)  # Uniform state matching quantum circuit
psi = psi / np.linalg.norm(psi)
classical_state = A @ psi
classical_state = classical_state / np.linalg.norm(classical_state)

# Get quantum result
system_state_post = extract_postselected_system_state(results)

# Compare
rel_error = np.linalg.norm(classical_state - system_state_post)
cos_sim = np.abs(np.vdot(classical_state, system_state_post))

print("Post-selected quantum state (first 8 amplitudes):")
for i in range(min(8, len(system_state_post))):
    print(f"  |{format(i, f'0{n}b')}⟩: {system_state_post[i]:.6f}")

print("\nClassical state (first 8 amplitudes):")
for i in range(min(8, len(classical_state))):
    print(f"  |{format(i, f'0{n}b')}⟩: {classical_state[i]:.6f}")

print(f"\nComparison Results:")
print(f"Relative error: {rel_error:.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")
print(f"Quantum state norm: {np.linalg.norm(system_state_post):.6f}")
print(f"Classical state norm: {np.linalg.norm(classical_state):.6f}")
