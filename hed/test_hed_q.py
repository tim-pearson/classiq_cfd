# %%
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
def L0_circuit(system_qubits: QArray[QBit]):
    # Identity (does nothing)
    pass

@qfunc
def L1_circuit(system_qubits: QArray[QBit]):
    # Apply simple X gates between neighbors to mimic off-diagonal
    for i in range(N-1):
        X(system_qubits[i])
        X(system_qubits[i+1])

# Function to prepare |psi> state
# Alternative: if you know the size at compile time
@qfunc
def apply_ry_on_all_fixed(params: CArray[CReal], io: QArray[QBit]):
    # Manually apply to each qubit (if you know N=4)
    RY(params[0], io[0])
    RY(params[1], io[1])
    RY(params[2], io[2])
    RY(params[3], io[3])

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

# OR with Output type annotations:
@qfunc
def main(system_out: Output[QArray[QBit]], ancilla_out: Output[QArray[QBit]]):
    allocate(ANCILLA_SIZE, ancilla_out)
    allocate(N, system_out)
    
    prepare_psi_state(system_out)
    control(ancilla_out[0], lambda: L1_circuit(system_out))
# Create the quantum model
# Synthesize (simulate) the quantum circuit
qprog = synthesize(main, auto_show=False)
print("Quantum HED circuit synthesized!")

# %%
# 7. Verification (classical simulation of output)
# In practice, here we would measure and compute expectation values <psi|L_i|psi>
# For demonstration, we just check the classical reconstruction matches A@psi
print("\nComparison classical vs original A@psi:")
print("Original A@psi:", A @ psi)
print("HED reconstruction:", result_classical)
print("Frobenius error:", np.linalg.norm(A @ psi - result_classical))
