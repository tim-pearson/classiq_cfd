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

            block_encoding_vqls(
                ansatz=lambda: apply_fixed_2_qubit_system_ansatz(
                    params, system_qubits
                ),
                # ansatz=lambda: apply_fixed_2_qubit_system_ansatz_updated(
                #     params, system_qubits
                # ),

                # ansatz=lambda: apply_fixed_3_qubit_system_ansatz(
                #     params, system_qubits
                # ),
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
            # apply_fixed_3_qubit_system_ansatz(
            #     list(optimal_params.values()), io
            # )
            apply_fixed_2_qubit_system_ansatz(
                list(optimal_params.values()), io
            )

        qprog_3 = synthesize(main)

        with ExecutionSession(qprog_3, self.execution_preferences) as es:
            self.results = es.sample()



