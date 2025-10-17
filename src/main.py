import os

import matplotlib.pyplot as plt
from classiq import ClassiqBackendPreferences, IBMBackendPreferences, Pauli
import numpy as np
from classiq import ClassiqBackendPreferences, ClassiqNvidiaBackendNames
from pandas.io.formats.style import plt
from optimizer import VqlsOptimizer
from vqls import Vqls
from dotenv import load_dotenv

# %%

load_dotenv()
tk = os.environ["IBMQ_API_KEY"]


# be_name =get_ibm_backends(tk)[0].name
# print(be_name)
# backend_preferences = IBMBackendPreferences(
#     backend_name=be_name,
#     access_token=tk,
#     channel="ibm_quantum_platform",
#     instance_crn=crn,
# )

backend_preferences = ClassiqBackendPreferences(
    backend_name="simulator_statevector"
)


pauli_terms_structs_1 = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)

pauli_terms_structs_2 = (
    0.6 * Pauli.I(0)
    + 0.2 * Pauli.Z(0) * Pauli.I(1) * Pauli.I(2)
    + 0.2 * Pauli.I(0) * Pauli.Z(1) * Pauli.Z(2)
)

pauli_terms_structs_3 = (
    0.6 * Pauli.I(0)
    + 0.2 * Pauli.Z(0) * Pauli.I(1) * Pauli.I(2)
    + 0.2 * Pauli.I(0) * Pauli.Y(1) * Pauli.Z(2)
)


# %%
ansatz_param_count = 9
# b = np.array([0.2, 0.1, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])  
# b /= np.linalg.norm(b)
b = np.ones(8) / np.sqrt(8)
# %%
vqls = Vqls(ansatz_param_count, pauli_terms_structs_3, b, "ps3_test")
# vqls = Vqls(ansatz_param_count, pauli_terms_structs_2, b, "ps2_test")
# vqls = Vqls(ansatz_param_count, pauli_terms_structs_1, b, "ps1_test")
# %%
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(204800, backend_preferences=backend_preferences)
print("optimizing")
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
print("comparing results")
vqls.compare_results()
# %%

from utils import visualize_vqls_results
visualize_vqls_results("data")
