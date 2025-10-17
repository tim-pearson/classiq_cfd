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
    def __init__(self, ansatz_param_count, pauli_terms_structs, b):
        self.num_system_qubits = pauli_terms_structs.num_qubits
        self.num_ancila_qubits = (
            len(pauli_terms_structs.terms) - 1
        ).bit_length()
        self.ansatz_param_count = ansatz_param_count
        self.pauli_terms_structs = pauli_terms_structs
        b /= np.linalg.norm(b)
        self.b = b
        self.probs = (b**2) / np.sum(b**2)
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
                # prepare_b_state=lambda: inplace_prepare_state(
                #     probabilities=self.probs,
                #     bound=0.01,
                #     target=system_qubits,
                # ),
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
        res = self.results

        # --- Try to get amplitudes if available ---
        if hasattr(res, "dataframe") and "amplitude" in getattr(res, "dataframe", {}).columns:
            df = res.dataframe
            amplitudes = np.zeros(2**self.num_system_qubits, dtype=complex)
            amplitudes[df.io] = df.amplitude
            print("✅ Using amplitudes from statevector simulation.")
        else:
            # --- Fall back to probabilities from Qiskit sampler ---
            print("⚠️  No amplitudes found — reconstructing from measurement probabilities.")
            probs_dict = res.probabilities
            sorted_probs = [probs_dict.get(format(i, f"0{self.num_system_qubits}b"), 0.0)
                            for i in range(2**self.num_system_qubits)]
            amplitudes = np.sqrt(sorted_probs)  # assume real nonnegative amplitudes
            # note: this loses phase information, but it's all we can get from measurements

        # Normalize amplitudes
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Optional phase adjustment
        if np.iscomplexobj(amplitudes):
            global_phase = np.angle(amplitudes[-1])
            amplitudes = amplitudes / np.exp(1j * global_phase)
            amplitudes = np.real(amplitudes)
        if amplitudes[-1] < 0:
            amplitudes *= -1

        # Store quantum probabilities
        self.quantum_probs = np.real(amplitudes)**2

        # Classical normalization
        normalization = sum(p.coefficient for p in self.pauli_terms_structs.terms)
        A_num = hamiltonian_to_matrix(self.pauli_terms_structs) / normalization
        b = self.b


        # Classical solution
        A_inv = np.linalg.inv(A_num)
        x_classical = np.dot(A_inv, b)
        self.classical_probs = (np.real(x_classical / np.linalg.norm(x_classical)))**2

        # Summary
        # Estimated x from quantum ansatz (already computed)
        x_estimated = amplitudes / np.linalg.norm(amplitudes)

        # Classical solution (already computed)
        x_classical = np.real(x_classical / np.linalg.norm(x_classical))

        # 🧾 Display both vectors
        print("\n=== Comparison of Vectors ===")
        print(f"b vector:        {np.round(b, 4)}")
        print(f"Cost count: {self.optimizer.count}")
        print(f"x_classical:     {np.round(x_classical, 4)}")
        print(f"x_estimated(q):  {np.round(x_estimated, 4)}")

        # 🧮 Cosine similarity or overlap
        cosine_sim = np.dot(x_estimated, x_classical) / (
            np.linalg.norm(x_estimated) * np.linalg.norm(x_classical)
        )
        print(f"\nCosine similarity between classical and quantum x: {cosine_sim:.6f}")

        # Overlap
        overlap = (b.dot(A_num.dot(amplitudes)) / np.linalg.norm(A_num.dot(amplitudes)))**2
        print(f"Overlap: {overlap:.6f}\n")


        # Plot
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        ax1.bar(np.arange(0, 2**self.num_system_qubits), self.classical_probs, color="blue")
        ax1.set_xlim(-0.5, 2**self.num_system_qubits - 0.5)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")

        ax2.bar(np.arange(0, 2**self.num_system_qubits), self.quantum_probs, color="gold")
        ax2.set_xlim(-0.5, 2**self.num_system_qubits - 0.5)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")

        plt.show()
