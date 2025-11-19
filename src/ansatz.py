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


def test_ansatz_expressibility(
    target_solution, ansatz_func, param_count, max_iterations=100
):
    """
    Standalone function to test ansatz expressibility
    Uses the same setup as your working VQLS optimizer
    """
    # Normalize target solution
    target_solution = target_solution / np.linalg.norm(target_solution)

    # Use the same execution preferences as your working optimizer
    backend_preferences = ClassiqBackendPreferences(
        backend_name="simulator_statevector"
    )
    execution_preferences = ExecutionPreferences(
        num_shots=20480, backend_preferences=backend_preferences
    )

    intermediate_costs = []

    def cost_function(params):
        """Cost function: 1 - fidelity between ansatz output and target"""

        @qfunc
        def main(io: Output[QNum[4]]):
            allocate(io)
            ansatz_func(list(params), io)

        # Synthesize and run circuit
        qprog = synthesize(main)
        with ExecutionSession(qprog, execution_preferences=execution_preferences) as es:
            results = es.sample()

        # Reconstruct output statevector
        df = results.dataframe
        output_state = np.zeros(2**4).astype(complex)
        output_state[df.io] = df.amplitude

        # Normalize output state
        output_state = output_state / np.linalg.norm(output_state)

        # Calculate infidelity
        fidelity = np.abs(np.vdot(target_solution, output_state)) ** 2
        intermediate_costs.append(1 - fidelity)
        return 1 - fidelity

    # Run optimization with same setup as your working optimizer
    random.seed(1000)
    initial_params = [
        float(np.random.randint(-314, 314)) / 1000 for _ in range(param_count)
    ]

    print(
        f"Testing {ansatz_func.__name__} with {param_count} parameters on 4 qubits..."
    )
    print(f"Initial parameters: {initial_params}")

    result = minimize(
        cost_function,
        x0=initial_params,
        method="COBYLA",
        options={"maxiter": max_iterations},
    )

    print(result)

    best_fidelity = 1 - result.fun
    best_params = result.x

    print(f"Maximum achievable fidelity: {best_fidelity:.4f}")
    print(f"Optimization success: {result.success}")

    # Interpretation
    if best_fidelity > 0.9:
        print("✅ Ansatz is EXCELLENT for this solution")
    elif best_fidelity > 0.7:
        print("✅ Ansatz is GOOD for this solution")
    elif best_fidelity > 0.5:
        print("⚠️  Ansatz is MARGINAL for this solution")
    else:
        print("❌ Ansatz is POOR for this solution")

    plt.plot(
        [l for l in range(len(intermediate_costs))],
        intermediate_costs,
    )
    plt.title("VQLS Incomplete Choleski Precondition 4-Q Ansatz expressibility")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    return best_fidelity, best_params
