import os

from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
    matrix_to_hamiltonian,
)
import matplotlib.pyplot as plt
from classiq import (
    ClassiqBackendPreferences,
    IBMBackendPreferences,
    Pauli,
    SparsePauliOp,
)
import numpy as np
from classiq import ClassiqBackendPreferences, ClassiqNvidiaBackendNames
from pandas.io.formats.style import plt
from optimizer import VqlsOptimizer
from vqls import Vqls
from dotenv import load_dotenv
from utils import laplacian_2d, genrate_random_b

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


backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")

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


pauli_terms_structs_4 = (
    0.5 * Pauli.I(0) * Pauli.I(1) * Pauli.I(2)  # main diagonal
    + 0.25 * Pauli.X(0) * Pauli.I(1) * Pauli.I(2)  # off-diagonal +1/-1 (flips qubit 0)
    + 0.25 * Pauli.I(0) * Pauli.X(1) * Pauli.I(2)  # off-diagonal +1/-1 (flips qubit 1)
)
pauli_pressure = (
    0.5 * Pauli.I(0) * Pauli.I(1)  # main diagonal
    - 0.25 * Pauli.X(0) * Pauli.I(1)  # flips qubit 0 → connects |00>↔|10>, |01>↔|11>
    - 0.5 * Pauli.I(0) * Pauli.X(1)  # flips qubit 1 → connects |00>↔|01>, |10>↔|11>
)

pauli_pressure_2x2_norm = (
    0.5 * Pauli.I(0) * Pauli.I(1)  
    - 0.25 * Pauli.I(0) * Pauli.X(1)  
    - 0.25 * Pauli.X(0) * Pauli.I(1)  
)
# %%


# %%
ansatz_param_count = 6
b, x, A = genrate_random_b(pauli_pressure_2x2_norm)
x = np.real(x)
x = x /  np.linalg.norm(x)
print("b = " , np.real(b))
print("x = " , x )
vqls = Vqls(ansatz_param_count, pauli_pressure_2x2_norm, b, "2x2 pressure b even super pos 2x2")
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
vqls.compare_results(x)
# %%
