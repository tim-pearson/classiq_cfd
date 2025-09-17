# VqlsOptimizer: Function Math Walkthrough


### Timothy Pearson

## __init__(...)
- Sets up the optimizer and execution session.
- Maps Python objects to your **quantum program**.
- Prepares storage for intermediate cost values.

Mathematically: initializes the parameter vector
$w = (w_0, w_1, \dots, w_{n-1})$ for the ansatz $V(w)$.

---

## get_cond_prop(res)
- Computes the **conditional probability**:
$$
  P_0 = P(\text{system} = |0\rangle_\text{sys} \mid \text{ancilla} =
|0^m\rangle_\text{anc})
$$
- Loops over all measurement outcomes in `res`.
- Accumulates counts:
$$
  P_0 = \frac{\#\text{shots with system=0 and ancilla=0}}{\#\text{shots with
ancilla=0}}
$$
- This corresponds to estimating $|\langle b | \Psi \rangle|^2$ in the VQLS
derivation.

---

## my_cost(params)
- Runs the quantum program with parameters `params`.
- Uses `get_cond_prop` to compute the **post-selected probability**:
$$
  C = 1 - |\langle b | \Psi \rangle|^2
$$
- Here,
$|\Psi\rangle = \frac{A|x\rangle}{\sqrt{\langle x | A^\dagger A | x \rangle}}$
is the system state **after post-selection** on ancilla $|0^m\rangle$.
- Minimizing $C$ is equivalent to **maximizing overlap** with the target state
$|b\rangle$.

---

## f(x) — Overview

The function `f(x)` converts a classical parameter vector 

$x = (x_0, x_1, \dots, x_{n-1})$ 

into a dictionary for the quantum circuit. This parametrizes the variational
ansatz:

$$
|x\rangle = V(x)|0\rangle
$$

It then evaluates the cost:

$$
C(x) = 1 - |\langle b | \Psi(x) \rangle|^2
$$

where 

$$
|\Psi(x)\rangle = \frac{A |x\rangle}{\sqrt{\langle x | A^\dagger A | x \rangle}}
$$

is the system state **after post-selecting the ancilla qubits**. 

The intermediate cost is stored for monitoring:

$$
\text{intermediate}[x] = C(x)
$$

Finally, `f(x)` returns $C(x)$ to the classical optimizer, which updates the
parameters to minimize the cost and maximize the overlap with the target state
$|b\rangle$:

$$
x_\text{opt} = \arg\min_x C(x)
$$

---

## optimize()

- Runs classical optimizer (COBYLA) to minimize:
$$
  \min_w C(w) = \min_w \left[ 1 - |\langle b | \Psi(w) \rangle|^2 \right]
$$
- Starts from a random initialization:
$$
  w_i \sim \text{Uniform}[0, 3]
$$
- Iteratively calls $f(w)$ to update parameters.
- Plots the **convergence of $C(w)$** over iterations.
- Returns the **optimal parameter set** $w_\text{opt}$ that prepares a state
$|x\rangle = V(w_\text{opt})|0\rangle$ such that $A|x\rangle \propto |b\rangle$.




---



Once the optimal variational weights $w_\text{opt}$ are found, the quantum
state 

$|x\rangle = V(w_\text{opt}) |0\rangle$ 

represents the solution prepared by the variational circuit. Here,
$V(w_\text{opt})$ is the unitary corresponding to the ansatz with the
parameters that minimize the cost function.

Measuring $|x\rangle$ in the computational basis means performing a standard
quantum measurement where the state collapses to one of the basis states
$|i\rangle$, with probability given by the squared amplitude of that basis
state in $|x\rangle$. Mathematically, the probability of observing basis state
$|i\rangle$ is

$P(i) = |\langle i | x \rangle|^2$.

By repeating the measurement many times, we can estimate the probabilities of
all basis states. These probabilities tell us how the quantum amplitude is
distributed across the computational basis, which in turn reflects the
structure of the solution found by the variational algorithm. This measurement
step is how we extract classical information from the quantum state after the
optimization has been performed.

---


Let's go through a **small example** to make this concrete.

Suppose we have a 2-qubit system. The computational basis states are:

$|00\rangle, |01\rangle, |10\rangle, |11\rangle$

After preparing the variational state $|x\rangle$, we measure it many times and
record the outcomes. Imagine we get the following **estimated probabilities**
from repeated measurements:

| Basis state  | Probability |
|--------------|-------------|
| $\ket{00}$ | 0.5         |
| $\ket{01}$ | 0.25        |
| $\ket{10}$ | 0.15        |
| $\ket{11}$ | 0.10        |

These probabilities correspond to the squared amplitudes:

$$
P(00) = |\langle 00 | x \rangle|^2 = 0.5, \quad
P(01) = |\langle 01 | x \rangle|^2 = 0.25, \dots
$$

From this, we can **reconstruct the amplitudes up to a phase**:

$$
|x\rangle \approx \sqrt{0.5} |00\rangle + \sqrt{0.25} |01\rangle + \sqrt{0.15}
|10\rangle + \sqrt{0.10} |11\rangle
$$

$$
|x\rangle \approx 0.707 |00\rangle + 0.5 |01\rangle + 0.387 |10\rangle + 0.316
|11\rangle
$$

- The **probabilities** come directly from measurements.  
- The **amplitudes** are the square roots of the probabilities.  
- Note: This gives the amplitudes **up to an unknown phase** (which we cannot
determine from standard measurements).  

So by measuring in the computational basis many times, we **estimate
$|x\rangle$’s amplitude distribution** over all basis states.