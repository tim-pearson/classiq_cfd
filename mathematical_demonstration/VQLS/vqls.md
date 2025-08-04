# Mathematical Foundations of VQLS with LCU Block Encoding

## Problem Formulation
Given a matrix $A$ and vector $|b\rangle$, we seek $|x\rangle$ such that:
$$A|x\rangle \propto |b\rangle$$

The VQLS algorithm approximates the solution using a parameterized quantum circuit (ansatz):
$$|x(\theta)\rangle = V(\theta)|0\rangle$$

## Block Encoding via LCU
For matrix $A$ expressed as a linear combination of unitaries:
$$A = \sum_{l=0}^{L-1} c_l A_l$$
where $c_l \geq 0$ and $\sum_l c_l = 1$.

### Key Components:

1. **State Preparation**:
   The coefficients $c_l$ of the linear combination are embedded into a quantum state via a unitary $U_c$. This prepares a superposition where each basis state $|l\rangle$ has amplitude $\sqrt{c_l}$:

$$
|\sqrt{c}\rangle = U_c|0\rangle = \sum_{l=0}^{L-1} \sqrt{c_l}|l\rangle
$$

   Here:
   - $U_c$ is a unitary operator acting on $m = \lceil \log_2 L \rceil$ ancilla qubits.
   - The state $|\sqrt{c}\rangle$ encodes the coefficients $c_l$ in its amplitudes.
   - Normalization $\sum_l c_l = 1$ ensures valid quantum state preparation.

2. **Controlled Unitaries**:
   For each unitary $A_l$ in the LCU decomposition, we implement a controlled operation $CA_l$ that applies $A_l$ to the system register only when the ancilla register is in the corresponding state $|l\rangle$:

$$
CA_l |l'\rangle|\psi\rangle = 
\begin{cases} 
(A_l|\psi\rangle) \otimes |l\rangle & \text{if } l' = l \\
|\psi\rangle \otimes |l'\rangle & \text{otherwise}
\end{cases}
$$

   Properties:
   - Each $CA_l$ is a unitary operation on the combined system+ancilla space.
   - The ancilla register acts as a control for selecting which $A_l$ to apply.
   - When $l' \neq l$, the operation reduces to identity.

3. **Block Encoding Unitary**:
   The complete block encoding unitary $U$ embeds the matrix $A$ in its top-left block while leaving other blocks arbitrary:

$$
U = \begin{bmatrix} 
A & \cdot \\ 
\cdot & \cdot 
\end{bmatrix}
$$

   Key features:
   - $U$ acts on the tensor product space of system and ancilla qubits.
   - The block encoding satisfies $A = (\langle 0| \otimes I) U (|0\rangle \otimes I)$.
   - The remaining blocks (marked as ·) can be arbitrary as long as $U$ remains unitary.
   
## Cost Function

   The objective is to maximize overlap between $A|x(\theta)\rangle$ and $|b\rangle$:
$$
C(\theta) = 1 - |\langle b|\Psi(\theta)\rangle|^2$$
  where:
$$
|\Psi(\theta)\rangle = \frac{A|x(\theta)\rangle}{\|A|x(\theta)\rangle\|}$$

## Measurement Protocol
The cost function is estimated via:
1. Prepare the state $U_b^\dagger A V(\theta)|0\rangle$
2. Measure probability of all qubits in $|0\rangle$ state:
   $$P(0) = |\langle 0|U_b^\dagger A V(\theta)|0\rangle|^2 = |\langle b|\Psi(\theta)\rangle|^2$$

## Classical Optimization
Minimize $C(\theta)$ using classical optimizers (e.g., COBYLA):
$$\theta^* = \text{argmin}_\theta C(\theta)$$

## Solution Extraction
The optimal state is:
$$|x(\theta^*)\rangle = V(\theta^*)|0\rangle$$
with probabilities:
$$P(x_i) = |\langle x_i|x(\theta^*)\rangle|^2$$

## References
[1] Bravo-Prieto et al., Variational Quantum Linear Solver (2019)  
[2] Kothari, Efficient algorithms in quantum query complexity (2014)
