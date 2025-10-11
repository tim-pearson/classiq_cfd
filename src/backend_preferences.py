# %
from qiskit_ibm_runtime import QiskitRuntimeService


def get_ibm_backends(api_key):

    QiskitRuntimeService.save_account(
        token=api_key,
        channel="ibm_quantum_platform",  # or "ibm_cloud" if using IBM Cloud
        overwrite=True
    )

    # Load the service
    service = QiskitRuntimeService(channel="ibm_quantum_platform")

    backends = service.backends()
    # for backend in backends:
    #     print(f"{backend.name:<25} |  Qubits: {backend.num_qubits} | Status: {backend.status().status_msg}")

    return backends
