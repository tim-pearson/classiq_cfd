import os

from classiq import ClassiqBackendPreferences, IBMBackendPreferences, Pauli
import numpy as np
from classiq import ClassiqBackendPreferences, ClassiqNvidiaBackendNames

from backend_preferences import get_ibm_backends
from vqls import Vqls
from dotenv import load_dotenv

load_dotenv()
tk = os.environ["IBMQ_API_KEY"]

# backend_preferences = ClassiqBackendPreferences(
#     backend_name="simulator_statevector"
# )
crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/291223f70a5f4ae99c0776d6a216f4c5:6bb73fb8-c76f-400e-b9f6-04fa38999975::"
be_name =get_ibm_backends(tk)[0].name
print(be_name)
backend_preferences = IBMBackendPreferences(
    backend_name=be_name,
    access_token=tk,
    channel="ibm_quantum_platform",
    instance_crn=crn,
)


pauli_terms_structs = (
    0.55 * Pauli.I(0)
    + 0.225 * Pauli.I(0) * Pauli.Z(1) * Pauli.I(2)
    + 0.225 * Pauli.I(0) * Pauli.I(1) * Pauli.Z(2)
)

ansatz_param_count = 9


b = np.array([0.2, 0.1, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])
vqls = Vqls(ansatz_param_count, pauli_terms_structs, b)
print("creating qprog")
vqls.create_qrog()
print("init optimizer")
vqls.init_optimizer(1024, backend_preferences=backend_preferences)
print("optimizing")
optimal_params = vqls.optimizer.optimize()
print("evalutating ansatz")
vqls.evaluate_ansatz(optimal_params)
print("comparing results")
vqls.compare_results()
