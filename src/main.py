from classiq import Pauli

from vqls import Vqls


pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)

ansatz_param_count = 9
vqls = Vqls(ansatz_param_count, pauli_terms_structs)
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(2048)
print("optimizing")
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
print("comparing results")
vqls.compare_results()
