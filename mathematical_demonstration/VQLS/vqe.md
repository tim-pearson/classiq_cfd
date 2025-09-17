---
title: Variational Quantum Eigensolver (VQE)
author: Timothy Pearson
---

# Variational Quantum Eigensolver (VQE)

VQE is a hybrid quantum-classical algorithm used to find the **ground state
energy** of a Hamiltonian $H$. It is variational in nature, which means it
relies on the **variational principle**:

## Variational Principle

If $H$ is a Hermitian operator (the Hamiltonian), then for any normalized
quantum state $\lvert \psi(\vec{\theta}) \rangle$,
the expected energy is:

$$
E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle
$$

The variational principle guarantees:

$$
E(\vec{\theta}) \geq E_0
$$

where $E_0$ is the ground state energy (the smallest eigenvalue of $H$).

The goal is to find parameters $\vec{\theta}$ such that $E(\vec{\theta})$ is
minimized:

$$
\vec{\theta}^* = \arg\min_{\vec{\theta}} \langle \psi(\vec{\theta}) | H |
\psi(\vec{\theta}) \rangle
$$

## VQE Components

### 1. Ansatz State

A parametrized quantum circuit prepares a state:

$$
\lvert \psi(\vec{\theta}) \rangle = U(\vec{\theta}) \lvert 0 \rangle
$$

This is known as the **ansatz** — a flexible form for approximating the ground
state.

### 2. Hamiltonian Decomposition

Quantum computers measure in the **computational basis**, not directly in terms
of arbitrary Hermitian operators. So to evaluate the expectation value:

$$
E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle
$$

we must **decompose the Hamiltonian $H$** into **Pauli strings**, which are
tensor products of the identity and Pauli operators:

$$
H = \sum_j c_j P_j, \quad \text{where } c_j \in \mathbb{R}, \quad P_j \in \{I,
X, Y, Z\}^{\otimes n}
$$

Each $P_j$ is called a **Pauli string**, and the Hamiltonian becomes a **linear
combination** of them.

#### Why This Works

The Pauli matrices $\{I, X, Y, Z\}$ form a **basis** for the space of Hermitian
operators acting on qubits. Therefore, any Hermitian matrix (and hence any
quantum Hamiltonian) can be expanded in this basis.

For an $n$-qubit system, each $P_j$ is a tensor product of $n$ single-qubit
Pauli operators.

#### Example: 2-Qubit Hamiltonian

Let’s say we have a molecule like H₂, whose Hamiltonian has been mapped to
qubits using the Jordan-Wigner or Bravyi-Kitaev transformation. The Hamiltonian
might look like:

$$
H = c_0 I \otimes I + c_1 Z \otimes I + c_2 I \otimes Z + c_3 Z \otimes Z + c_4
Y \otimes Y + c_5 X \otimes X
$$

Each $P_j = P_j^{(1)} \otimes P_j^{(2)}$ acts on a subset or all of the qubits.

### Measuring the Energy

To compute:

$$
\langle \psi | H | \psi \rangle = \sum_j c_j \langle \psi | P_j | \psi \rangle
$$

we run the quantum circuit to prepare $\lvert \psi(\vec{\theta}) \rangle$ and
measure the expectation value of each $P_j$ **individually**, then combine them
classically with their weights $c_j$.

#### Important Notes (Detailed)

When estimating expectation values of Pauli strings $P_j$, there are several
practical considerations:

---

### 1. Basis Rotation for Measurement

Quantum hardware typically measures qubits in the **computational basis**
(i.e., the $Z$ basis). That means you can measure observables like
$Z \otimes Z \otimes I$ directly.

However, Pauli strings may involve $X$ or $Y$ operators. These are **not
diagonal in the computational basis**, so we must apply **basis-change gates**
to convert them into $Z$ measurements.

#### Rules for Basis Rotation:

- To measure $X$, rotate the basis using the **Hadamard** gate:
  $$
  H X H = Z
  $$
  So to measure $X$ on qubit $i$, apply $H$ to qubit $i$ before measurement.

- To measure $Y$, rotate using $S^\dagger$ (the inverse phase gate) followed by
$H$:
  $$
  H S^\dagger Y S H = Z
  $$
  So to measure $Y$ on qubit $i$, apply $S^\dagger$ then $H$ before measuring.

These transformations turn $X$ or $Y$ into $Z$ so that we can measure them
using the hardware's native $Z$-basis measurement.

#### Example:

Suppose $P_j = X \otimes Y \otimes I$. Then:

- Apply $H$ to qubit 0 (for $X$),
- Apply $S^\dagger$ then $H$ to qubit 1 (for $Y$),
- Do nothing on qubit 2 (since it's identity).

Then measure all qubits in the $Z$ basis.

---

### 2. Statistical Estimation via Shots

Quantum measurements are **probabilistic**. To estimate the expectation value
$\langle \psi | P_j | \psi \rangle$, we must:

- Prepare the quantum state $\lvert \psi(\vec{\theta}) \rangle$,
- Measure it **multiple times** (called **shots**, e.g., 1000),
- Compute the **average** result from these samples.

#### Example:

Say you're measuring $Z \otimes Z$ and you get the following outcomes over 8
shots:


### Summary of Hamiltonian Decomposition

- Decomposition of $H$ into Pauli strings is essential to compute
$E(\vec{\theta})$.
- Each Pauli term is easy to measure on quantum hardware.
- The total energy is reconstructed from the measured expectations of each term:

$$
E(\vec{\theta}) = \sum_j c_j \langle P_j \rangle
$$

This step allows the quantum part of VQE to be executed efficiently and links
VQE to classical optimization techniques.

### 3. Classical Optimization

The expected energy $E(\vec{\theta})$ is passed to a classical optimizer (e.g.,
gradient descent, Nelder-Mead, or COBYLA) to find the parameters $\vec{\theta}$
that minimize it.

## Summary of the VQE Loop

1. Choose an ansatz $U(\vec{\theta})$.
2. Prepare $\lvert \psi(\vec{\theta}) \rangle$ on the quantum computer.
3. Measure
$E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle$.
4. Update $\vec{\theta}$ using a classical optimizer.
5. Repeat until convergence.

---

# Why This Helps Understand VQLS

In **VQLS** (Variational Quantum Linear Solver), we similarly use a
parameterized quantum state to solve a **linear system of equations**:

$$
A \lvert x \rangle = \lvert b \rangle
$$

where $A$ is Hermitian and invertible. The structure of VQLS borrows directly
from VQE:

- Use a **variational ansatz** $\lvert x(\vec{\theta}) \rangle$.
- Define a **cost function** that captures how close
$A \lvert x(\vec{\theta}) \rangle$ is to $\lvert b \rangle$.
- Use **classical optimization** to minimize this cost.

The main difference lies in the cost function. In VQLS, it's typically defined
as:

$$
C(\vec{\theta}) = \| A \lvert x(\vec{\theta}) \rangle - \lvert b \rangle \|^2
$$

This is equivalent to:

$$
C(\vec{\theta}) = \langle x(\vec{\theta}) | A^\dagger A | x(\vec{\theta})
\rangle - 2 \text{Re}\left(\langle b | A | x(\vec{\theta}) \rangle\right) +
\langle b | b \rangle
$$

Since $A$ is Hermitian, $A^\dagger = A$.

VQLS inherits the same **ansatz-construction**, **expectation-value
estimation**, and **optimization loop** as VQE — it just changes the physical
problem: from ground-state energy estimation to solving linear systems.