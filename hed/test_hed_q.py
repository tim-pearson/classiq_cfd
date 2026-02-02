# %%
from typing import Tuple
from classiq import RY, X, CArray, CInt, CReal, Output, QArray, QBit, QNum, allocate, control, qfunc, repeat
from classiq.synthesis import synthesize
import numpy as np


# %%
# 1. Create 1D Poisson matrix
def poisson_1d_matrix(N):
    A = 2*np.eye(N)
    for i in range(N-1):
        A[i, i+1] = -1
        A[i+1, i] = -1
    return A

N = 4
A = poisson_1d_matrix(N)
print("Original A:\n", A)

# %%
# 2. HED decomposition
def hed_decomposition(A):
    N = A.shape[0]
    L0 = np.eye(N)
    L1 = np.zeros((N, N))
    for i in range(N-1):
        L1[i,i+1] = -1
        L1[i+1,i] = -1
    L_matrices = [L0, L1]

    # Solve least squares for alpha
    A_vec = A.flatten()
    L_vecs = np.stack([L.flatten() for L in L_matrices], axis=1)
    alpha, _, _, _ = np.linalg.lstsq(L_vecs, A_vec, rcond=None)
    return alpha, L_matrices

def build_A_from_hed(alpha, L_matrices):
    result = np.zeros_like(A)
    for a, L in zip(alpha, L_matrices):
        result += a * L
    return result

alpha, L_matrices = hed_decomposition(A)
print("HED coefficients:", alpha)
print(f"Verify A = A_hed : ",np.allclose(A, build_A_from_hed(alpha, L_matrices)))

# %%
# 3. Prepare test state |psi>
psi = np.random.rand(N)
psi = psi / np.linalg.norm(psi)
print("Test state |psi>:", psi)

# Classical application for verification
def apply_hed_classical(alpha, L_matrices, psi):
    result = np.zeros_like(psi)
    for a, L in zip(alpha, L_matrices):
        result += a * (L @ psi)
    return result

result_classical = apply_hed_classical(alpha, L_matrices, psi)
print("Classical HED A|psi>:", result_classical)

# %%
# 4. Convert |psi> to angles for single-qubit rotations (toy for demo)
def state_to_angles(psi):
    # For N=2^n, we approximate with single-qubit RY rotations
    return [2*np.arcsin(np.sqrt(p)) for p in psi**2]

angles = state_to_angles(psi)
print("Rotation angles for |psi>:", angles)

# %%
import numpy as np
from classiq import *

# Define N
N = 4

# 5. Define shallow L_i circuits as qfuncs

@qfunc
def prepare_psi_state(system_qubits: QArray[QBit]):
    # Prepare |psi> - example probabilities
    psi = [0.25, 0.25, 0.25, 0.25]
    rotation_angles = [2*np.arcsin(np.sqrt(p)) for p in psi]
    
    # Apply RY to each qubit
    for i in range(N):
        RY(rotation_angles[i], system_qubits[i])
# Main function


ANCILLA_SIZE = 2  # Fixed number of ancilla qubits
@qfunc
def prepare_psi_state(system_qubits: QArray[QBit]):
    psi = [0.25]*system_qubits.len
    rotation_angles = [2*np.arcsin(np.sqrt(p)) for p in psi]
    for i in range(system_qubits.len):
        RY(rotation_angles[i], system_qubits[i])



# %%

@qfunc
def L1_circuit(system_qubits: QArray[QBit]):
    X(system_qubits[3])

@qfunc
def L2_circuit(system_qubits: QArray[QBit]):
    n = system_qubits.len
    for i in range(1, n):
        controls = system_qubits[:i]
        target = system_qubits[i]
        control(ctrl=controls, stmt_block=lambda t=target: X(t))

@qfunc
def L3_circuit(system_qubits: QArray[QBit]):
    target = system_qubits[3]
    control(ctrl=[system_qubits[0], system_qubits[1], system_qubits[2]], stmt_block=lambda: Z(target))
    control(ctrl=[system_qubits[0], system_qubits[1], system_qubits[2]], stmt_block=lambda: X(target))

@qfunc
def main(system_out: Output[QArray[QBit]], ancilla_out: Output[QArray[QBit]]):
    allocate(3, ancilla_out)
    allocate(4, system_out)
    prepare_psi_state(system_out)
    control(ctrl=ancilla_out[0], stmt_block=lambda: L1_circuit(system_out))
    control(ctrl=ancilla_out[1], stmt_block=lambda: L2_circuit(system_out))
    control(ctrl=ancilla_out[2], stmt_block=lambda: L3_circuit(system_out))

qprog = synthesize(main, auto_show=False)
print("Quantum HED circuit synthesized!")


# %%

results =None
sv =None
backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
execution_preferences = ExecutionPreferences(
            num_shots=1000, backend_preferences=backend_preferences
        )

with ExecutionSession(qprog, execution_preferences) as es:
            results = es.sample()
print(results)
# %%

# %%
def extract_postselected_system_state(results, ancilla_bits=3, system_bits=4):
    parsed = results.parsed_state_vector_states
    postselected_state = np.zeros(2**system_bits, dtype=complex)
    for full_state, amp in results.state_vector.items():
        ancilla_state = full_state[:ancilla_bits]
        system_state = full_state[ancilla_bits:]

        system_bits_str = full_state[-system_bits:]  # last system_bits bits
        if set(ancilla_state) == {'0'}:
            idx = int(system_bits_str, 2)
            postselected_state[idx] = amp
    # Normalize
    norm = np.linalg.norm(postselected_state)
    if norm > 0:
        postselected_state /= norm
    return postselected_state

# %%
def classical_HED_apply(alpha, L_matrices, psi):
    result = np.zeros_like(psi)
    for a, L in zip(alpha, L_matrices):
        result += a * (L @ psi)
    return result / np.linalg.norm(result)

# %%
def postselect_and_compare(results, alpha, L_matrices, psi):
    system_state_post = extract_postselected_system_state(results)
    classical_state = classical_HED_apply(alpha, L_matrices, psi)
    
    # Pad classical state if necessary
    if len(system_state_post) != len(classical_state):
        classical_state = np.pad(classical_state, (0, len(system_state_post) - len(classical_state)))
    
    rel_error = np.linalg.norm(classical_state - system_state_post)
    cos_sim = np.abs(np.vdot(classical_state, system_state_post))
    
    print("Post-selected system state amplitudes:\n", system_state_post)
    print("-"*65)
    print(f"Relative error: {rel_error:.6f}, Cosine similarity: {cos_sim:.6f}")
    return system_state_post, classical_state, rel_error, cos_sim


system_state_post, classical_state, rel_error, cos_sim = postselect_and_compare(
    results, alpha, L_matrices, psi
)
