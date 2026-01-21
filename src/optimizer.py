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

    def get_quantum_residual_vector(self, results):
        counts = results.parsed_counts_of_outputs(
            [self.ansatz_var_name, self.aux_var_name]
        )

        # Success branch = ancilla = 0
        total = 0
        for s in counts:
            if s.state[self.aux_var_name] == 0:
                total += s.shots

        vec = []
        for basis in sorted(counts.states(self.ansatz_var_name)):
            shots = 0
            for s in counts:
                if (
                    s.state[self.aux_var_name] == 0
                    and s.state[self.ansatz_var_name] == basis
                ):
                    shots += s.shots
            vec.append(shots / total)

        return np.array(vec)

    def preconditioned_cost(self, params):
        # 1. Run quantum sampling
        results = self.es.sample(params)

        # 2. Extract approximated Ax(θ)
        Ax_vec = self.get_quantum_residual_vector(results)

        # 3. Compute classical residual r = A x - b
        r = Ax_vec - self.b_vec      # b_vec must be given as np.array earlier

        # 4. Apply classical preconditioner
        pre_r = self.M_inv @ r

        # 5. Norm squared
        cost = float(pre_r.T @ pre_r)

        return cost

    def f(self, x):
        # Convert optimizer parameters into dict for quantum execution
        param_dict = {"params_" + str(k): x[k] for k in range(self.ansatz_param_count)}

        cost = self.preconditioned_cost(param_dict)

        self.intermediate[tuple(x)] = cost
        return cost

    def optimize(self, M_inv):
        self.M_inv = M_inv
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
