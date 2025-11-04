import os
from classiq import H, ExecutionPreferences, allocate, apply_to_all, qfunc
from classiq.applications.chemistry import PauliOperator
from classiq import IBMBackendPreferences

import matplotlib.pyplot as plt
from classiq import (
    CArray,
    CReal,
    ExecutionSession,
    Output,
    QNum,
    allocate,
    lcu_pauli,
    qfunc,
    write_qmod,
    inplace_prepare_state,
)
from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
)
from classiq.synthesis import synthesize
import numpy as np

from ansatz import apply_fixed_2_qubit_system_ansatz, apply_fixed_2_qubit_system_ansatz_updated, apply_fixed_3_qubit_system_ansatz
from block_encoding import block_encoding_vqls
from optimizer import VqlsOptimizer
from utils import fidelity, normalize, plot_classical_vs_quantum, save_stats_to_json

DATA_DIR = "data/"


class Vqls:
    def __init__(self, ansatz_param_count, pauli_terms_structs, b, name):
        self.name = name
        self.num_system_qubits = pauli_terms_structs.num_qubits
        print(self.num_system_qubits)
        self.num_ancila_qubits = (
            len(pauli_terms_structs.terms) - 1
        ).bit_length()

        print(self.num_ancila_qubits)
        self.ansatz_param_count = ansatz_param_count
        self.pauli_terms_structs = pauli_terms_structs
        b /= np.linalg.norm(b)
        self.b = b
        self.probs = (b**2) / np.sum(b**2)

        if np.imag(self.probs).sum() > 0.01:
            raise Exception("probabilities and not real")
        else:
            self.probs = np.real(self.probs)

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

            print("num_system_qubits:", self.num_system_qubits)
            print("length of self.b:", len(self.b))
            print("sum of probabilities:", np.sum(self.probs))
            block_encoding_vqls(
                # ansatz=lambda: apply_fixed_2_qubit_system_ansatz(
                #     params, system_qubits
                # ),
                # ansatz=lambda: apply_fixed_2_qubit_system_ansatz_updated(
                #     params, system_qubits
                # ),

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
        self.num_shots = num_shots

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
            # apply_fixed_2_qubit_system_ansatz(
            #     list(optimal_params.values()), io
            # )

        qprog_3 = synthesize(main)

        with ExecutionSession(qprog_3, self.execution_preferences) as es:
            self.results = es.sample()


    def compare_results(self, x, A):
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
        A_num =A
        # b = np.ones(N) / np.sqrt(N)  # uniform RHS
        print(" classical solution x  = ", x)
        classical_probs = np.real((x / np.linalg.norm(x))) ** 2
        self.classical_probs = classical_probs

        b = self.b
        # --- 3) Compute statistics ---
        overlap = (
            b.dot(
                A_num.dot(amplitudes) / np.linalg.norm(A_num.dot(amplitudes))
            )
        ) ** 2
        mse = np.mean((amplitudes - x / np.linalg.norm(x)) ** 2)
        cosine_similarity = np.dot(probabilities, classical_probs) / (
            np.linalg.norm(probabilities) * np.linalg.norm(classical_probs)
        )

        # Store stats in a dictionary
        stats = {
            "iterations": self.optimizer.count,
            "Shots per iteration": self.num_shots,
            "overlap": float(np.real(overlap)),
            "mse": float(np.real(mse)),
            "cosine_similarity": float(np.real(cosine_similarity)),
            "classical_probs": classical_probs.tolist(),
            "quantum_probs": probabilities.tolist(),
        }

        # Print metrics
        print(f"Iterations = {stats['iterations']}")
        print(f"Overlap = {stats['overlap']:.6f}")
        print(f"MSE = {stats['mse']:.6e}")
        print(f"Cosine similarity = {stats['cosine_similarity']:.6f}")
        print(f"Classical probs = {classical_probs}")
        print(f"Estimated probs = {probabilities.tolist()}")
        print(f"Shots per iteration: {self.num_shots}")

        # --- 4) Save stats to JSON ---
        save_stats_to_json(stats, self.name, folder="data")

        # --- 5) Plot ---
        plot_classical_vs_quantum(classical_probs, probabilities, self.name)

        return stats

    def compare_quantum_classical(A, b, quantum_amplitudes, verbose=True):
        """
        A: full matrix (NxN)
        b: target vector (length N) — ideally normalized to unit norm
        quantum_amplitudes: |x> returned from get_solution_from_results
"""
        x_q = quantum_amplitudes
        x_q = normalize(x_q)
        b_norm = normalize(b)

        # In VQLS we expect A @ x_q proportional to b. Compare normalized versions:
        b_est = A @ x_q.T
        b_est_norm = normalize(b_est)

        fid = fidelity(b_norm, b_est_norm)       # fidelity between target and A|x>
        residual = np.linalg.norm(b_norm - b_est_norm)

        if verbose:
            print("quantum |x> (amplitudes):", x_q)
            print("A @ |x> (unnorm):", b_est)
            print("A @ |x> (norm):", b_est_norm)
            print("target b (norm):", b_norm)
            print(f"fidelity(target, A|x>) = {fid:.6f}")
            print(f"residual norm(target - A|x>) = {residual:.6e}")

        return {"fidelity": fid, "residual": residual, "b_est": b_est, "b_est_norm": b_est_norm}
    def get_solution(self):
        N = 2**self.num_system_qubits
        df = self.results.dataframe
        amplitudes = np.zeros(N, dtype=complex)
        amplitudes[df.io.values.astype(int)] = df.amplitude.values

        # Remove global phase
        global_phase = np.angle(amplitudes[-1])
        amplitudes /= np.exp(1j * global_phase)

        # Ensure last amplitude is positive real
        if np.real(amplitudes[-1]) < 0:
            amplitudes *= -1

        # Return normalized real part (VQLS expects real x)
        return np.real(amplitudes / np.linalg.norm(amplitudes))


