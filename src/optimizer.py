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

    def get_vqls_cost_residual(self, x_statevector):
        """
        Residual cost: ||A x(theta) - b||^2
        """
        Ax = self.A @ x_statevector
        residual = Ax - self.b
        return float(np.vdot(residual, residual).real)

    def my_cost(self, params):
        # Run the quantum circuit and get the STATEVECTOR of x(theta)
        results = self.es.sample(params)

        # You MUST extract the ansatz output statevector
        x_statevector = results.get_statevector(self.ansatz_var_name)

        # Compute residual cost
        return self.get_vqls_cost_residual(x_statevector)

    def f(self, x):
        cost = self.my_cost(
            {"params_" + str(k): x[k] for k in range(self.ansatz_param_count)}
        )
        self.intermediate[tuple(x)] = cost
        return cost

    def optimize(self):
        random.seed(1000)

        initial_params = [
            float(random.randint(-157, 157)) / 1000  # Range: -0.314 to 0.314 radians
            for _ in range(self.ansatz_param_count)
        ]

        print(f"Initial parameters: {initial_params}")

        self._out = out = minimize(
            self.f,
            x0=initial_params,
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
