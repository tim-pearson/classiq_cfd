import matplotlib.pyplot as plt
from classiq import (
    H,
    CArray,
    CReal,
    ClassiqBackendPreferences,
    ExecutionPreferences,
    ExecutionSession,
    Output,
    Pauli,
    QNum,
    allocate,
    apply_to_all,
    lcu_pauli,
    qfunc,
    write_qmod,
)
from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
)
from classiq.synthesis import synthesize
import numpy as np

from ansatz import apply_fixed_3_qubit_system_ansatz
from block_encoding import block_encoding_vqls
from optimizer import VqlsOptimizer

pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)
normalization = sum([p.coefficient for p in pauli_terms_structs.terms])


ansatz_param_count = 9
num_system_qubits = pauli_terms_structs.num_qubits
num_ancila_qubits = (len(pauli_terms_structs.terms) - 1).bit_length()


@qfunc
def main(
    params: CArray[CReal, ansatz_param_count],
    ancillary_qubits: Output[QNum[num_ancila_qubits]],
    system_qubits: Output[QNum[num_system_qubits]],
):

    allocate(ancillary_qubits)
    allocate(system_qubits)

    block_encoding_vqls(
        ansatz=lambda: apply_fixed_3_qubit_system_ansatz(
            params, system_qubits
        ),
        block_encoding=lambda: lcu_pauli(
            operator=pauli_terms_structs,
            data=system_qubits,
            block=ancillary_qubits,
        ),
        prepare_b_state=lambda: apply_to_all(H, system_qubits),
    )


print("Starting synthesis...")
qprog_2 = synthesize(main, auto_show=False)
print("Synthesis done")

write_qmod(
    main, name="vqls_with_lcu", decimal_precision=15, symbolic_only=False
)

print("Setting up Backend and Execuations Preferences..")
backend_preferences = ClassiqBackendPreferences(
    backend_name="simulator_statevector"
)
execution_preferences = ExecutionPreferences(
    num_shots=204800, backend_preferences=backend_preferences
)
print("Backend and Execuations Preferences Set up")

print("Creating optimizer...")

optimizer = VqlsOptimizer(
    qprog_2,
    ansatz_param_count,
    "system_qubits",
    "ancillary_qubits",
    execution_preferences,
)

print("Optimizer created")

print("starting Optimization...")
optimal_params = optimizer.optimize()
print("Optimization finished, Optimal Params:")
print(optimal_params)


@qfunc
def main(io: Output[QNum[num_system_qubits]]):
    allocate(io)
    apply_fixed_3_qubit_system_ansatz(list(optimal_params.values()), io)


qprog_3 = synthesize(main)

with ExecutionSession(qprog_3, execution_preferences) as es:
    results = es.sample()

df = results.dataframe
print(df)


amplitudes = np.zeros(2**num_system_qubits).astype(complex)
amplitudes[df.io] = df.amplitude
# Preprocessed quantum solution: we know the solution is real, and that the last point is positive
global_phase = np.angle(amplitudes[-1])
amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
if (
    amplitudes[-1] < 0
):  # we can extract the solution up to a sign, align with the expected
    amplitudes *= -1
print(amplitudes)


probabilities = amplitudes**2


A_num = hamiltonian_to_matrix(pauli_terms_structs) / normalization
b = np.ones(8) / np.sqrt(8)


A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)
classical_probs = np.real((x / np.linalg.norm(x))) ** 2

print(
    "overlap =",
    (b.dot(A_num.dot(amplitudes) / (np.linalg.norm(A_num.dot(amplitudes)))))
    ** 2,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

ax1.bar(np.arange(0, 2**num_system_qubits), classical_probs, color="blue")
ax1.set_xlim(-0.5, 2**num_system_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(0, 2**num_system_qubits), probabilities, color="gold")
ax2.set_xlim(-0.5, 2**num_system_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

plt.show()
