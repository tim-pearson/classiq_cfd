import random

from classiq import ExecutionSession
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


class VqlsOptimizer:
    def __init__(
        self,
        qprog,
        ansatz_param_count,
        ansatz_var_name,
        aux_var_name,
        exe_prefs=None,
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
        self.count = 0

    def get_hadamard_expectation(self, res):
        """
        Compute expectation value from Hadamard test results
        Returns ⟨Z_ancilla⟩ = P(ancilla=0) - P(ancilla=1)
        This gives Re(⟨ψ|U|ψ⟩) or Im(⟨ψ|U|ψ⟩) depending on the circuit
"""
        ancilla_0_count = 0
        ancilla_1_count = 0
        total_shots = 0

        for sample in res:
            ancilla_state = sample.state[self.aux_var_name]
            shots = sample.shots

            if ancilla_state == 0:
                ancilla_0_count += shots
            elif ancilla_state == 1:
                ancilla_1_count += shots
            total_shots += shots

        if total_shots == 0:
            return 0.0

        # ⟨Z⟩ = P(0) - P(1)
        p0 = ancilla_0_count / total_shots
        p1 = ancilla_1_count / total_shots
        expectation = p0 - p1

        return expectation


    def get_vqls_cost(self, res):
        expectation = self.get_hadamard_expectation(res)  # in [-1,1]
        cost = (1.0 - expectation) / 2.0
        return float(np.clip(cost, 0.0, 1.0))

    def my_cost(self, params):
        results = self.es.sample(params)
        parsed = results.parsed_counts_of_outputs(
            [self.ansatz_var_name, self.aux_var_name]
        )
        return self.get_vqls_cost(parsed)

    def f(self, x):
        cost = self.my_cost(
            {"params_" + str(k): x[k] for k in range(self.ansatz_param_count)}
        )
        self.intermediate[tuple(x)] = cost
        return cost

    def optimize(self):
        random.seed(1000)

        initial_params = [
            float(random.randint(-157, 157))
            / 1000  # Range: -0.314 to 0.314 radians
            for _ in range(self.ansatz_param_count)
        ]

        print(f"Initial parameters: {initial_params}")

        self._out = out = minimize(
            self.f,
            x0=initial_params,  # Use the better initialization
            method="COBYLA",
            options={"maxiter": 2000},
        )
        print(out)
        self._out_f = out_f = [out["x"][0 : self.ansatz_param_count]]
        print(out_f)

        # Plot convergence
        plt.plot(
            [l for l in range(len(self.intermediate))],
            list(self.intermediate.values()),
        )
        plt.title("VQLS Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

        return {
            "params_" + str(k): list(self.intermediate.keys())[-1][k]
            for k in range(self.ansatz_param_count)
        }
