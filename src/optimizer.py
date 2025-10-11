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
        self._out = out = minimize(
            self.f,
            x0=[
                float(random.randint(0, 3000)) / 1000
                for _ in range(0, self.ansatz_param_count)
            ],
            method="COBYLA",
            options={"maxiter": 2000},
        )
        print(out)
        self._out_f = out_f = [out["x"][0 : self.ansatz_param_count]]
        print(out_f)
        plt.plot(
            [l for l in range(len(self.intermediate))],
            list(self.intermediate.values()),
        )

        return {
            "params_" + str(k): list(self.intermediate.keys())[-1][k]
            for k in range(self.ansatz_param_count)
        }
