# Mathematical Foundations of VQLS with LCU Block Encoding

## Problem Statement

We aim to solve a system of linear equations in the form:
$$Ax=b$$

or in quantum notation:
$$A \ket{x} = \ket{b}$$

## Conditions on Matrix A

For quantum implementation, we require that matrix $A$ satisfies:

- **Unitary condition**: $AA^{\dagger} = I$
- **Hermitian condition**: $A^{\dagger} = A$
- **Real eigenvalues**: $A_{ij} = A_{ij}^*$ for $i \ne j$ (ensures stability
and unitary embeddings)

These conditions are necessary because quantum circuits only implement unitary
operations.

## Block Encoding of A

The block encoding of matrix $A$ is defined as:
$$\bra{0}_{anc} U \ket{0}_{anc} = A$$

We represent this as:
$$
U_A = \begin{bmatrix}
A & \cdot \\
\cdot & \cdot
\end{bmatrix}
$$

We "complete" $U$ by filling in the $\cdot$'s so that $U_A$ is unitary. The
values inside the $\cdot$'s don't affect the outcome due to post-selection and
projection.

## State Preparation

### Target State
$$\ket{\Psi} = \frac{A \ket{x}}{\sqrt{\bra{x} A^{\dagger} A \ket{x}}}$$

This is the solution state we are looking for (theoretically
$\ket{\Psi} = \ket{b}$).

### Variational Ansatz
We approximate $\ket{x}$ with a variational unitary circuit $V(w)$ where
$w = (w_0, w_1, \ldots)$:
$$\ket{x} = V(w)\ket{0}$$

### Preparation of $\ket{b}$
For any quantum state $\ket{b}$, there exists a unitary $U_b$ such that:
$$\ket{b} = U_b \ket{0}$$
where:
$$\ket{b} =\sum_{i=0}^{2^n -1} c_i \ket{i} \quad \text{with} \quad \sum_{i}
|c_i|^2 = 1$$

## Cost Function

The probability we want to maximize is:
$$|\braket{b|\Psi}|^2$$

This represents the fidelity/overlap between the two states. The cost function
we minimize is:
$$
C = 1 - |\braket{b|\Psi}|^2
$$

## Quantum Circuit Implementation

### Circuit Diagram
\[
\Qcircuit @C=1.0em @R=1.0em {
\lstick{\ket{0}_1} & \qw & \multigate{6}{\text{BEnc of A}} & \qw      & \meter
\\
\lstick{\ket{0}_j} & \qw & \ghost{\text{BEnc of A}}        & \qw      & \meter
\\
\lstick{\ket{0}_{m-1}} & \qw & \ghost{\text{Block Enc.}}        & \qw      &
\meter \\
    \vdots        &     &                            &          &    \vdots
\\
\lstick{\ket{0}_1} & \multigate{2}{V(\omega)} & \ghost{\text{Block Enc.}} &
\multigate{2}{U_b^\dagger} & \meter \\
\lstick{\ket{0}_j} & \ghost{V(\omega)}        & \ghost{\text{Block Enc.}} &
\ghost{U_b} & \meter \\
\lstick{\ket{0}_{n-1}}& \ghost{V(\omega)}        & \ghost{\text{Block Enc.}} &
\ghost{U_b} & \meter
}
\]

### Step-by-Step Execution

1. **Initial state preparation**:
   $$\ket{\psi_0} =  \ket{0}_{anc} \otimes \ket{0}_{sys}$$

2. **Apply variational ansatz**:
   $$\ket{\psi_1} =  \ket{0}_{anc} \otimes V(w)\ket{0}_{sys} = \ket{0}_{anc}
\otimes \ket{x}$$

3. **Apply block encoding of A**:
   For a 2D representation:
   $$
   U_A = \begin{bmatrix}
   A & B \\
   C & D
   \end{bmatrix}
   $$
   $$
   U\begin{bmatrix} \ket{x} \\ 0 \end{bmatrix} = \begin{bmatrix} A & B\\ C & D
   \end{bmatrix} \begin{bmatrix}
   \ket{x} \\ 0 \end{bmatrix} = \begin{bmatrix}
   A\ket{x} \\ C \ket{x} \end{bmatrix}
   $$
   $$
   \ket{\psi_2} = \ket{0}_{anc} \otimes A \ket{x} + \ket{1}_{anc} \otimes
\ket{\phi_{\perp}}
   $$

4. **Post-selection**:
   We project the ancillary qubits to the $\ket{0}$ state:
   $$P_0 = \ket{0^m}\bra{0^m}_{anc} \otimes I_{sys}$$
   $$P_0 \ket{\psi_2} = \ket{0^m}_{anc} \otimes A \ket{x}_{sys}$$

5. **Success probability**:
   $$p_{succ} = |\bra{\psi_2} P_0\ket{\psi_2}|^2 = \bra{x} A^\dagger A \ket{x}$$

6. **Normalized system state** (after successful post-selection):
   $$\ket{\Psi}_{sys} = \frac{A \ket{x}}{\sqrt{\bra{x} A^\dagger A \ket{x}}}$$

## Cost Function Evaluation

The cost function is evaluated as:
$$
C = 1 - |\braket{b|\Psi}|^2 = 1 - |\bra{0} U_b^\dagger \ket{\Psi}|^2
$$

This represents the probability of the system being in state $\ket{0}$ after
applying $U_b^\dagger$, given that the ancilla was post-selected to $\ket{0^m}$.

## Results Interpretation

After sampling the quantum circuit execution, we obtain results with:

- **io**: bitstrings representing basis outcomes (e.g., "010", "011")
- **amplitude**: complex statevector amplitudes
- **probabilities/counts**: derived from amplitudes

### Phase and Sign Ambiguity Resolution

Since quantum states are defined up to:
1. **Global phase**: Divide all amplitudes by $e^{i\theta}$ to align with
reference
2. **Sign ambiguity**: Multiply entire vector by $-1$ if reference amplitude is
negative

### Probability Calculation
After fixing phase conventions:
$$p_i = |\langle i | \psi \rangle|^2$$

These `quantum_probs` can be compared against classical solutions to evaluate
overlap:
$$\text{overlap} = |\langle b | \Psi \rangle|^2$$

## Circuit Execution Summary

1. **Prepare system and ancilla**:
   - System: $|x⟩ = V(w)|0⟩_{sys}$
   - Ancilla: $|0^m⟩$

2. **Apply block-encoding**:
   - Apply $U_A$ on $(|0^m⟩_{anc} \otimes |x⟩_{sys})$

3. **Measure ancilla**:
   - Keep only shots with outcome $0^m$
   - Success probability: $p_{succ} = ⟨x| A^\dagger A |x⟩$
   - Conditioned system state: $|Ψ⟩$

4. **Apply $U_b^\dagger$** to the system

5. **Measure system** in computational basis:
   - Frequency of $|0⟩$ estimates $|⟨b|Ψ⟩|^2$

6. **Compute cost function**:
   - $C = 1 - |⟨b|Ψ⟩|^2$
   - Update parameters $w$