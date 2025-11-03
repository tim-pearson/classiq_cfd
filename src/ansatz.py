from classiq import CZ, RY, CArray, CReal, QArray, QBit, qfunc, repeat


@qfunc()
def apply_ry_on_all(params: CArray[CReal], io: QArray[QBit]):
    repeat(count=io.len, iteration=lambda index: RY(params[index], io[index]))


@qfunc
def apply_fixed_3_qubit_system_ansatz(
    angles: CArray[CReal], system_qubits: QArray[QBit]
):
    apply_ry_on_all([angles[0], angles[1], angles[2]], system_qubits)
    repeat(
        count=(system_qubits.len - 1),
        iteration=lambda index: CZ(system_qubits[0], system_qubits[index + 1]),
    )
    CZ(system_qubits[1], system_qubits[2])
    apply_ry_on_all([angles[3], angles[4], angles[5]], system_qubits)
    repeat(
        count=(system_qubits.len - 1),
        iteration=lambda index: CZ(
            system_qubits[system_qubits.len - 1], system_qubits[index]
        ),
    )
    CZ(system_qubits[1], system_qubits[0])
    apply_ry_on_all([angles[6], angles[7], angles[8]], system_qubits)



@qfunc
def apply_fixed_2_qubit_system_ansatz(angles: CArray[CReal], system_qubits: QArray[QBit]):
    # angles should be length 6 (3 layers of RY for 2 qubits)
    
    # Layer 1: RY on both qubits
    RY(angles[0], system_qubits[0])
    RY(angles[1], system_qubits[1])
    
    # Entangling layer
    CZ(system_qubits[0], system_qubits[1])
    
    # Layer 2: RY on both qubits
    RY(angles[2], system_qubits[0])
    RY(angles[3], system_qubits[1])
    
    # Entangling layer (optional: flip CZ direction)
    CZ(system_qubits[1], system_qubits[0])
    
    # Layer 3: RY on both qubits
    RY(angles[4], system_qubits[0])
    RY(angles[5], system_qubits[1])