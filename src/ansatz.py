from classiq import CX, CZ, RY, RZ, U, CArray, CReal, QArray, QBit, qfunc, repeat


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
def ansatz_2_enhanced(
    angles: CArray[CReal], system_qubits: QArray[QBit]
):
    """
    Enhanced 2-qubit ansatz with better expressivity
    angles: length 12 for full expressivity
    """
    # Layer 1: Full single-qubit rotations
    U(angles[0], angles[1], angles[2], 0, system_qubits[0])  # U3 gate
    U(angles[3], angles[4], angles[5], 0, system_qubits[1])  # U3 gate
    
    # Entangling layer 1
    CX(system_qubits[0], system_qubits[1])
    
    # Layer 2: Single-qubit rotations
    RY(angles[6], system_qubits[0])
    RY(angles[7], system_qubits[1])
    
    # Entangling layer 2
    CZ(system_qubits[1], system_qubits[0])
    
    # Layer 3: Final rotations
    RZ(angles[8], system_qubits[0])
    RZ(angles[9], system_qubits[1])
    RY(angles[10], system_qubits[0])
    RY(angles[11], system_qubits[1])

@qfunc
def ansatz_2_efficient(angles: CArray[CReal], system_qubits: QArray[QBit]):
    """
    Efficient 2-qubit ansatz with 8 parameters
    Still maintains good expressibility with reduced parameter space
    """
    # Layer 1: Single-qubit rotations (4 params)
    RY(angles[0], system_qubits[0])
    RY(angles[1], system_qubits[1])
    RZ(angles[2], system_qubits[0])
    RZ(angles[3], system_qubits[1])
    
    # Entangling layer (1 entangling gate)
    CX(system_qubits[0], system_qubits[1])
    
    # Layer 2: Single-qubit rotations (4 params)
    RY(angles[4], system_qubits[0])
    RY(angles[5], system_qubits[1])
    RZ(angles[6], system_qubits[0])
    RZ(angles[7], system_qubits[1])

@qfunc
def ansatz_4_hardware(angles: CArray[CReal], qubits: QArray[QBit]):
    """
    4-qubit hardware-efficient ansatz with linear entanglement
    24 parameters
    """
    #Layer 1: Single-qubit rotations on all qubits
    for i in range(4):
        RY(angles[i], qubits[i])
    
    # Linear entanglement
    CX(qubits[0], qubits[1])
    CX(qubits[1], qubits[2]) 
    CX(qubits[2], qubits[3])
    
    # Layer 2: Single-qubit rotations
    for i in range(4):
        RY(angles[4 + i], qubits[i])
        RZ(angles[8 + i], qubits[i])
    
    # Reverse linear entanglement
    CX(qubits[2], qubits[3])
    CX(qubits[1], qubits[2])
    CX(qubits[0], qubits[1])
    
    # Final layer
    for i in range(4):
        RY(angles[12 + i], qubits[i])

@qfunc
def ansatz_4_compact(angles: CArray[CReal, 8], qubits: QArray[QBit]):
    """
    Very compact 4-qubit ansatz - 8 parameters
    Good for avoiding barren plateaus
    """
    # Layer 1: Single-qubit rotations
    for i in range(4):
        RY(angles[i], qubits[i])
    
    # Minimal entanglement
    CX(qubits[0], qubits[1])
    CX(qubits[2], qubits[3])
    
    # Layer 2: Final rotations
    for i in range(4):
        RY(angles[4 + i], qubits[i])

@qfunc
def ansatz_4_balanced(angles: CArray[CReal, 12], qubits: QArray[QBit]):
    """
    Balanced 4-qubit ansatz - 12 parameters
    Good mix of expressibility and trainability
    """
    # Layer 1: RY rotations
    for i in range(4):
        RY(angles[i], qubits[i])
    
    # Linear entanglement
    CX(qubits[0], qubits[1])
    CX(qubits[1], qubits[2])
    CX(qubits[2], qubits[3])
    
    # Layer 2: RY + RZ rotations
    for i in range(4):
        RY(angles[4 + i], qubits[i])
        RZ(angles[8 + i], qubits[i])
