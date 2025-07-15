# Solving a Linear System with a Variational Quantum Algorithm

Given a matrix $A$ and a vector $|\mathbf{b}\rangle$, the goal is to solve the linear equation:

$$
A |\mathbf{x}\rangle = |\mathbf{b}\rangle
$$

where $|\mathbf{x}\rangle$ is the solution vector we're trying to find.

---

## Variational Ansatz

Instead of solving this directly, a quantum algorithm tries to find an approximate solution by preparing a **parametrized quantum state**:

$$
|x\rangle = V(w) |0\rangle
$$

Here:

- $V(w)$ is a unitary operation (a quantum circuit) that depends on parameters $w = (w_0, w_1, \dots)$.
- $|0\rangle$ is the initial quantum state (all qubits in zero).
- By tuning $w$, the algorithm tries to produce $|x\rangle$ that approximates the true solution.

---

## Block Encoding of Matrix $A$

We encode the matrix $A$ into a larger unitary matrix $U$ called a **block encoding**:

$$
U = \begin{bmatrix} A & \cdot \\ \cdot & \cdot \end{bmatrix}
$$

This means that $A$ appears as the top-left block of $U$. $U$ itself is unitary (quantum operations must be unitary), and $A$ might not be, so $U$ is constructed to embed $A$ inside a bigger unitary operator.

---

## Preparing the State Related to $A$

We then prepare the quantum state:

$$
|\Psi\rangle = \frac{A|x\rangle}{\sqrt{\langle x|A^\dagger A|x\rangle}}
$$

which is basically applying the matrix $A$ (embedded via $U$) to the variational state $|x\rangle$ and normalizing the result.

Our goal is for $|\Psi\rangle$ to approximate the vector $|b\rangle$:

$$
|\Psi\rangle \approx |b\rangle
$$

where $|b\rangle$ is the normalized state corresponding to the right-hand side vector $|\mathbf{b}\rangle$.

---

## Cost Function to Optimize

To measure how close $|\Psi\rangle$ is to $|b\rangle$, we use the cost function:

$$
C = 1 - |\langle b | \Psi \rangle|^2
$$

- If $|\Psi\rangle$ matches $|b\rangle$ perfectly, then the inner product $|\langle b | \Psi \rangle|^2 = 1$ and $C=0$ (minimum cost).
- The algorithm tries to **minimize $C$** by adjusting parameters $w$ in $V(w)$.

---

## Measuring Overlaps Using Probabilities

In the quantum circuit, the overlap $|\langle b|\Psi\rangle|^2$ can be related to measurable probabilities:

$$
|\langle b | \Psi \rangle|^2 = P(\mathrm{sys}=\mathrm{ground} \mid \mathrm{anc} = \mathrm{ground}) = \frac{P(\mathrm{all}=\mathrm{ground})}{P(\mathrm{anc}=\mathrm{ground})}
$$

- $P(\mathrm{sys}=\mathrm{ground} \mid \mathrm{anc} = \mathrm{ground})$ means the probability that the system qubits are in the ground state given the ancilla qubit is in the ground state.
- $P(\mathrm{all}=\mathrm{ground})$ is the probability that all qubits (system + ancilla) are in the ground state.
- This relation helps estimate the inner product by measuring outcomes on the quantum computer.

---

## Summary

- We want to solve $A|x\rangle = |b\rangle$.
- Instead of direct inversion, we use a **variational quantum approach** that prepares a guess $|x\rangle = V(w)|0\rangle$.
- We encode $A$ into a bigger unitary $U$.
- Apply $A$ to $|x\rangle$ and normalize to get $|\Psi\rangle$.
- We measure how close $|\Psi\rangle$ is to $|b\rangle$ using the cost function $C$.
- By adjusting $w$, we minimize $C$.
- The overlap is measured via certain quantum probabilities during the circuit run.

---

If you want, I can help you go even deeper into how each step is implemented on a quantum computer or how the optimization is done!

