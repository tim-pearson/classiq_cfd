from classiq import allocate, qfunc
from classiq.applications.chemistry import PauliOperator
from classiq import IBMBackendPreferences

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
    matrix_to_pauli_operator,
)
from classiq.synthesis import synthesize
import numpy as np

from ansatz import apply_fixed_3_qubit_system_ansatz
from block_encoding import block_encoding_vqls
from optimizer import VqlsOptimizer


class Vqls:
    def __init__(self, ansatz_param_count, pauli_terms_structs):
        self.num_system_qubits = pauli_terms_structs.num_qubits
        self.num_ancila_qubits = (
            len(pauli_terms_structs.terms) - 1
        ).bit_length()
        self.ansatz_param_count = ansatz_param_count
        self.pauli_terms_structs = pauli_terms_structs

    def create_qrog(self, qmod_file=False):
        @qfunc
        def main(
            params: CArray[CReal, self.ansatz_param_count],
            ancillary_qubits: Output[QNum[self.num_ancila_qubits]],
            system_qubits: Output[QNum[self.num_system_qubits]],
        ):

            allocate(ancillary_qubits)
            allocate(system_qubits)

            block_encoding_vqls(
                ansatz=lambda: apply_fixed_3_qubit_system_ansatz(
                    params, system_qubits
                ),
                block_encoding=lambda: lcu_pauli(
                    operator=self.pauli_terms_structs,
                    data=system_qubits,
                    block=ancillary_qubits,
                ),
                prepare_b_state=lambda: apply_to_all(H, system_qubits),
            )

        self.qprog_2 = synthesize(main, auto_show=False)
        if qmod_file:
            write_qmod(
                main,
                name="vqls_with_lcu",
                decimal_precision=15,
                symbolic_only=False,
            )

    def init_optimizer(self, num_shots=204800):
        backend_preferences = ClassiqBackendPreferences(
            backend_name="simulator_statevector"
        )
        self.execution_preferences = ExecutionPreferences(
            num_shots=num_shots, backend_preferences=backend_preferences
        )

        self.optimizer = VqlsOptimizer(
            self.qprog_2,
            self.ansatz_param_count,
            "system_qubits",
            "ancillary_qubits",
            self.execution_preferences,
        )

    def evaluate_ansatz(self, optimal_params):
        @qfunc
        def main(io: Output[QNum[self.num_system_qubits]]):
            allocate(io)
            apply_fixed_3_qubit_system_ansatz(
                list(optimal_params.values()), io
            )

        qprog_3 = synthesize(main)

        with ExecutionSession(qprog_3, self.execution_preferences) as es:
            self.results = es.sample()

    def compare_results(self):
        df = self.results.dataframe

        amplitudes = np.zeros(2**self.num_system_qubits).astype(complex)
        amplitudes[df.io] = df.amplitude
        global_phase = np.angle(amplitudes[-1])
        amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
        if amplitudes[-1] < 0:
            amplitudes *= -1

        self.quantum_probs = amplitudes**2

        normalization = sum(
            [p.coefficient for p in self.pauli_terms_structs.terms]
        )
        A_num = hamiltonian_to_matrix(self.pauli_terms_structs) / normalization
        b = np.ones(8) / np.sqrt(8)

        A_inv = np.linalg.inv(A_num)
        x = np.dot(A_inv, b)
        self.classical_probs = np.real((x / np.linalg.norm(x))) ** 2

        print(
            "overlap =",
            (
                b.dot(
                    A_num.dot(amplitudes)
                    / (np.linalg.norm(A_num.dot(amplitudes)))
                )
            )
            ** 2,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

        ax1.bar(
            np.arange(0, 2**self.num_system_qubits),
            self.classical_probs,
            color="blue",
        )
        ax1.set_xlim(-0.5, 2**self.num_system_qubits - 0.5)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")

        ax2.bar(
            np.arange(0, 2**self.num_system_qubits),
            self.quantum_probs,
            color="gold",
        )
        ax2.set_xlim(-0.5, 2**self.num_system_qubits - 0.5)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")

        plt.show()
