
# %%

from typing import List

import numpy as np

from classiq import *

# %%

@qfunc
def block_encoding_vqls(
    ansatz: QCallable,
    block_encoding: QCallable,
    prepare_b_state: QCallable,
) -> None:
    ansatz()
    block_encoding()
    invert(lambda: prepare_b_state())

# %%

import random

import matplotlib.pyplot as plt
from scipy.optimize import minimize


class VqlsOptimizer:
    def __init__(
        self, qprog, ansatz_param_count, ansatz_var_name, aux_var_name, exe_prefs=None
    ):
        self.qprog = qprog
        self.ansatz_param_count = ansatz_param_count
        self.ansatz_var_name = ansatz_var_name
        self.aux_var_name = aux_var_name
        if exe_prefs is None:
            self.es = ExecutionSession(qprog)
        else:
            self.es = ExecutionSession(qprog, exe_prefs)
        self.intermediate = {}

    def get_cond_prop(self, res):
        aux_prob_0 = 0
        all_prob_0 = 0
        for s in res:
            if s.state[self.aux_var_name] == 0:
                aux_prob_0 += s.shots
                if s.state[self.ansatz_var_name] == 0:
                    all_prob_0 = s.shots
        return all_prob_0 / aux_prob_0

    def my_cost(self, params):
        results = self.es.sample(params)
        return 1 - self.get_cond_prop(
            results.parsed_counts_of_outputs([self.ansatz_var_name, self.aux_var_name])
        )

    def f(self, x):
        cost = self.my_cost(
            {"params_" + str(k): x[k] for k in range(self.ansatz_param_count)}
        )
        self.intermediate[tuple(x)] = cost
        return cost

    def optimize(self):
        random.seed(1000)
        self._out = out = minimize(
            self.f,
            x0=[
                float(random.randint(0, 3000)) / 1000
                for i in range(0, self.ansatz_param_count)
            ],
            method="COBYLA",
            options={"maxiter": 2000},
        )
        print(out)
        self._out_f = out_f = [out["x"][0 : self.ansatz_param_count]]
        print(out_f)
        plt.plot(
            [l for l in range(len(self.intermediate))], list(self.intermediate.values())
        )

        return {
            "params_" + str(k): list(self.intermediate.keys())[-1][k]
            for k in range(self.ansatz_param_count)
        }

# %%

pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)
normalization = sum([p.coefficient for p in pauli_terms_structs.terms])

num_system_qubits = pauli_terms_structs.num_qubits
num_ancila_qubits = (len(pauli_terms_structs.terms) - 1).bit_length()
ansatz_param_count = 9

# %%

@qfunc
def apply_ry_on_all(params: CArray[CReal], io: QArray[QBit]):
    repeat(count=io.len, iteration=lambda index: RY(params[index], io[index]))


@qfunc
def apply_fixed_3_qubit_system_ansatz(
    angles: CArray[CReal], system_qubits: QArray[QBit]
):
    apply_ry_on_all([angles[0], angles[1], angles[2]], system_qubits)
    repeat(
        count=(system_qubits.len - 1),
        iteration=lambda index: CZ(system_qubits[0], system_qubits[index + 1]),
    )
    CZ(system_qubits[1], system_qubits[2])
    apply_ry_on_all([angles[3], angles[4], angles[5]], system_qubits)
    repeat(
        count=(system_qubits.len - 1),
        iteration=lambda index: CZ(
            system_qubits[system_qubits.len - 1], system_qubits[index]
        ),
    )
    CZ(system_qubits[1], system_qubits[0])
    apply_ry_on_all([angles[6], angles[7], angles[8]], system_qubits)

# %%

@qfunc
def main(
    params: CArray[CReal, ansatz_param_count], system_qubits: Output[QArray[QBit]]
):
    allocate(3, system_qubits)
    apply_fixed_3_qubit_system_ansatz(params, system_qubits)


qprog_1 = synthesize(main)
show(qprog_1)

# %%

@qfunc
def main(
    params: CArray[CReal, ansatz_param_count],
    ancillary_qubits: Output[QNum[num_ancila_qubits]],
    system_qubits: Output[QNum[num_system_qubits]],
):

    allocate(ancillary_qubits)
    allocate(system_qubits)

    block_encoding_vqls(
        ansatz=lambda: apply_fixed_3_qubit_system_ansatz(params, system_qubits),
        block_encoding=lambda: lcu_pauli(
            operator=pauli_terms_structs, data=system_qubits, block=ancillary_qubits
        ),
        prepare_b_state=lambda: apply_to_all(H, system_qubits),
    )

# %%

qprog_2 = synthesize(main)
show(qprog_2)

# %%

write_qmod(main, name="vqls_with_lcu", decimal_precision=15, symbolic_only=False)

# %%

backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
execution_preferences = ExecutionPreferences(
    num_shots=204800, backend_preferences=backend_preferences
)

optimizer = VqlsOptimizer(
    qprog_2,
    ansatz_param_count,
    "system_qubits",
    "ancillary_qubits",
    execution_preferences,
)
optimal_params = optimizer.optimize()

# %%

@qfunc
def main(io: Output[QNum[num_system_qubits]]):
    allocate(io)
    apply_fixed_3_qubit_system_ansatz(list(optimal_params.values()), io)


qprog_3 = synthesize(main)

# %%

with ExecutionSession(qprog_3, execution_preferences) as es:
    results = es.sample()

df = results.dataframe

# %%

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

# %%

probabilities = amplitudes**2

# %%

A_num = hamiltonian_to_matrix(pauli_terms_structs) / normalization
b = np.ones(8) / np.sqrt(8)

# %%

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)
classical_probs = np.real((x / np.linalg.norm(x))) ** 2
classical_probs

# %%

print(
    "overlap =",
    (b.dot(A_num.dot(amplitudes) / (np.linalg.norm(A_num.dot(amplitudes))))) ** 2,
)

# %%

import matplotlib.pyplot as plt

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
