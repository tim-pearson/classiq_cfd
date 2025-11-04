import random

from classiq import ExecutionSession
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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

    def get_cond_prop(self, res):
        aux_prob_0 = 0
        all_prob_0 = 0
        for s in res:
            if s.state[self.aux_var_name] == 0:
                aux_prob_0 += s.shots
                if s.state[self.ansatz_var_name] == 0:
                    all_prob_0 += s.shots
        return all_prob_0 / aux_prob_0

    def my_cost(self, params):

        self.count += 1
        results = self.es.sample(params)

        return 1 - self.get_cond_prop(
            results.parsed_counts_of_outputs(
                [self.ansatz_var_name, self.aux_var_name]
            )
        )

    def f(self, x):
        cost = self.my_cost(
            {"params_" + str(k): x[k] for k in range(self.ansatz_param_count)}
        )
        self.intermediate[tuple(x)] = cost
        return cost

    def optimize(self):
        random.seed(1000)

        initial_params = [
            float(random.randint(-314, 314))
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

    def optimize_with_better_settings(self):
        random.seed(1000)

        # Better initialization - try multiple small perturbations
        base_params = [0.125, -0.213, 0.089, 0.047, -0.25, 0.165, 0.01, -0.01, 0.2]
        initial_params = [
            p + random.uniform(-0.05, 0.05)
            for p in base_params[: self.ansatz_param_count]
        ]

        print(f"Initial parameters: {[f'{x:.3f}' for x in initial_params]}")

        # Use different optimization method or settings
        self._out = out = minimize(
            self.f,
            x0=initial_params,
            method="Nelder-Mead",  # Better for noisy landscapes
            options={
                "maxiter": 5000,
                "xatol": 1e-6,
                "fatol": 1e-6,
                "adaptive": True,
            },
        )

        print(out)
        self._out_f = [out["x"][: self.ansatz_param_count]]
        print(f"Final parameters: {self._out_f}")

        # Plot convergence
        if hasattr(self, "intermediate") and self.intermediate:
            plt.figure(figsize=(10, 6))
            plt.plot(list(self.intermediate.values()))
            plt.title("VQLS Convergence")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.yscale("log")  # Log scale to see improvements better
            plt.grid(True)
            plt.show()

        return {
            "params_" + str(k): list(self.intermediate.keys())[-1][k]
            for k in range(self.ansatz_param_count)
        }
