from classiq import  ExecutionPreferences, QBit, allocate, qfunc

from classiq import (
    CArray,
    CReal,
    ExecutionSession,
    Output,
    QNum,
    allocate,
    lcu_pauli,
    qfunc,
    inplace_prepare_state,
)

from classiq.qmod.qmod_variable import QArray
from classiq.synthesis import synthesize
import numpy as np

from ansatz import ansatz_4_hardware
from block_encoding import block_encoding_vqls
from optimizer import VqlsOptimizer




class Vqls:
    def __init__(self, ansatz_param_count, pauli_terms_structs, b, ansatz):
        self.num_system_qubits = pauli_terms_structs.num_qubits
        self.num_ancila_qubits = (len(pauli_terms_structs.terms) - 1).bit_length()
        self.ansatz_param_count = ansatz_param_count
        self.ansatz = ansatz
        self.pauli_terms_structs = pauli_terms_structs
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
                ansatz=lambda: self.ansatz(params, system_qubits),
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
            )

        self.qprog_2 = synthesize(main, auto_show=False)

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
            self.ansatz(list(optimal_params.values()), io)

        qprog_3 = synthesize(main)

        with ExecutionSession(qprog_3, self.execution_preferences) as es:
            self.results = es.sample()
