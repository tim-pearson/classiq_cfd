from classiq import Pauli
import numpy as np

from vqls import Vqls


pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)

ansatz_param_count = 9


x= np.array([0.2, 0.1, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])
vqls = Vqls(ansatz_param_count, pauli_terms_structs, x)
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(204800)
print("optimizing")
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
print("comparing results")
vqls.compare_results()
