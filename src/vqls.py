import os
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
    inplace_prepare_state,
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
    def __init__(self, ansatz_param_count, pauli_terms_structs, b, A):
        self.A =A
        self.num_system_qubits = pauli_terms_structs.num_qubits
        self.num_ancila_qubits = (
            len(pauli_terms_structs.terms) - 1
        ).bit_length()
        self.ansatz_param_count = ansatz_param_count
        self.pauli_terms_structs = pauli_terms_structs
        b /= np.linalg.norm(b)
        self.b = b
        self.probs = (b**2) / np.sum(b**2)
        # self.probs =np.linalg.norm(b)
        self.backend = None

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
                prepare_b_state=lambda: inplace_prepare_state(
                    probabilities=self.probs,
                    bound=0.01,
                    target=system_qubits,
                ),
                # prepare_b_state=lambda: apply_to_all(H, system_qubits),
            )

        self.qprog_2 = synthesize(main, auto_show=False)
        if qmod_file:
            write_qmod(
                main,
                name="vqls_with_lcu",
                decimal_precision=15,
                symbolic_only=False,
            )

    def init_optimizer(self, num_shots=204800, backend_preferences=None):
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
        N = 2**self.num_system_qubits

        # --- 1) Get quantum amplitudes and probabilities ---
        df = self.results.dataframe
        amplitudes = np.zeros(N, dtype=complex)
        amplitudes[df.io.values.astype(int)] = df.amplitude.values

        # Remove global phase
        global_phase = np.angle(amplitudes[-1])
        amplitudes = np.real(amplitudes / np.exp(1j * global_phase))
        if amplitudes[-1] < 0:
            amplitudes *= -1

        probabilities = amplitudes**2
        self.quantum_probs = probabilities

        # --- 2) Classical solution ---
        normalization = sum(p.coefficient for p in self.pauli_terms_structs.terms)
        A_num = self.A
        b = np.ones(N) / np.sqrt(N)  # uniform RHS
        A_inv = np.linalg.inv(A_num)
        x = A_inv @ b
        classical_probs = np.real((x / np.linalg.norm(x)))**2
        self.classical_probs = classical_probs

        # --- 3) Compute overlap ---
        overlap = (b.dot(A_num.dot(amplitudes) / np.linalg.norm(A_num.dot(amplitudes))))**2
        print("overlap =", overlap)

        # --- 4) Plot classical vs quantum probabilities ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

        ax1.bar(np.arange(N), classical_probs, color="blue")
        ax1.set_xlim(-0.5, N-0.5)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")

        ax2.bar(np.arange(N), probabilities, color="gold")
        ax2.set_xlim(-0.5, N-0.5)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")

        plt.show()

