# VQLS

## Variational Circuit (Quantum Part)

block encoding:

$$
U = 
\begin{bmatrix}
A & \cdot \\
\cdot & \cdot
\end{bmatrix}
$$

where we can prepare the state:

$$
\ket{\Psi} := A \ket{x} / \sqrt{\braket{x | A^ \dagger A|x}}
$$

we can approximate the solution $\ket{x}$ with a **variational qauntum circuit**

i.e. a **unitarty circuit** $V$, depending on a finite number of 
classical real parameters:
$\omega = (\omega_1 , \omega_2, \ldots)$:

$$
\ket{x} = V(\omega) \ket{0}
$$

Our objective is to address the task of **preparing** the quantum state $\ket{x}$
such that $A \ket{x}$ is **proportional** to $\ket{b}$


$$
\ket{\Psi} := \frac{A \ket{x} }{ \sqrt{\braket{x | A^ \dagger A|x}}} \approx \ket{b}
$$

The state $\ket{bh}$ arises form the unitary operatoin $U_b$ applied
to the ground state of $n$ qubits, i.e:

$$
\ket{b} = U_b \ket{0}
$$

To maximize the overlap between the quantum state $\ket{\Psi}$ and 
$\ket{b}$, we optimize the parameters, defining the cost function:

$$
C = 1 - |\braket{b|\Psi}|^2
$$

At a high level , the above could be impelemnted as follows:

we construct a quantum model with its this circuit:


\[
\Qcircuit @C=1.2em @R=1.5em {
\lstick{\ket{0}_1} & \qw & \multigate{6}{\text{Block Enc.}} & \qw      & \meter \\
\lstick{\ket{0}_j} & \qw & \ghost{\text{Block Enc.}}        & \qw      & \meter \\
\lstick{\ket{0}_{m-1}} & \qw & \ghost{\text{Block Enc.}}        & \qw      & \meter \\
    \vdots        &     &                            &          &    \vdots    \\
\lstick{\ket{0}_1} & \multigate{2}{V(\omega)} & \ghost{\text{Block Enc.}} & \multigate{2}{U_b} & \meter \\
\lstick{\ket{0}_j} & \ghost{V(\omega)}        & \ghost{\text{Block Enc.}} & \ghost{U_b} & \meter \\
\lstick{\ket{0}_{n-1}}& \ghost{V(\omega)}        & \ghost{\text{Block Enc.}} & \ghost{U_b} & \meter
}
\]

where:

- the qubits $1 \rightarrow m$ are **system qubits**
- the qubits $1 \rightarrow n$ are ancillary qubits**
- $n$: the number of **system qubits** used to represent the solution state $\ket{x}$.
- $m$: the number of **ancillary qubits** used in the block encoding of the matrix $A$.
- $V(\omega)$: a **parameterized quantum circuit** (ansatz) acting on the system qubits that prepares the variational state $\ket{x} = V(\omega) \ket{0}$.
- $U_b$: a **unitary operator** that prepares the state $\ket{b} = U_b \ket{0}$, representing the right-hand side of the equation $A \ket{x} \propto \ket{b}$.

### Block Encoding for VQLS – Overview

This function represents a core quantum routine in the Variational Quantum Linear Solver (VQLS) algorithm. It combines three key quantum operations to process and solve a linear system of equations $Ax = b$:

1. **Ansatz Application**  
   The ansatz is a parameterized quantum circuit that encodes a candidate solution vector $x$ as a quantum state. Applying the ansatz prepares this trial state, which will be refined through optimization.

2. **Block Encoding of Matrix $A$**  
   The block encoding circuit embeds the matrix $A$ into a larger unitary operation on the quantum computer. This allows the algorithm to perform operations involving $A$ efficiently in a quantum-compatible form.

3. **Inverse Preparation of the $b$ State**  
   The preparation circuit creates the quantum state corresponding to the right-hand side vector $b$. Taking its inverse (adjoint) effectively "unprepares" this state, which is essential for measuring overlaps and performing certain transformations required by the solver.

Together, these steps enable the quantum system to simulate the action of $A$ on $x$ and compare it to $b$, forming the basis for variational optimization to find the solution $x$. The function modularizes these components so they can be defined independently and combined flexibly depending on the problem setup.
From here, we only need to define `ansatz`, `block_encoding`, and 
`prepare_b_state` to fit the specific example above. Now we are 
ready to build our model, synthesize it, execute it, and analyze 
the results.

## Finding Optimal Parameters (Classical Part)

To estimate the overlap of the ground state with the post-selected
state, we could directly make use of the measurement samples. 
However, since we want to optimize the cost function, it is useful 
to express everything in terms of expectation values through Bayes'
theorem:
$$
\begin{align*}
|\braket{b|\Psi}|^2 &= P(sys= ground | anc = ground) \\
&= P(all=ground)/ P(anc=ground)
\end{align*}
$$

To evaluate the conditional probabilty from the above, we construct
a utility funtion


# Variational Quantum Linear Solver (VQLS) – Code-Specific Notes

We aim to **variationally solve a linear system** $A \ket{x} = \ket{b}$ by minimizing the following **cost function**:

$$
C(\omega) = 1 - \left| \braket{b | \Psi(\omega)} \right|^2
$$

where:

- $\ket{\Psi(\omega)} = \frac{A V(\omega) \ket{0}}{\| A V(\omega) \ket{0} \|}$
- $\ket{b}$ is the known target state,
- $V(\omega)$ is the **parameterized ansatz circuit** with classical parameters $\omega = (\omega_1, \omega_2, \dots, \omega_n)$.

---

## Overview of Class Components

### Class Initialization

- The class is initialized with:
  - A quantum program (`qprog`) that defines the circuit.
  - The number of ansatz parameters (`ansatz_param_count`).
  - The name of the qubit register encoding the solution (`ansatz_var_name`).
  - The name of the auxiliary register used for postselection (`aux_var_name`).
- An execution session is created from the quantum program for repeated sampling.

---

## Conditional Probability Estimation

- The circuit is executed and measurement results are parsed.
- From the counts, we estimate the conditional probability:

$$
P(\text{ansatz}=0 \mid \text{aux}=0) = \frac{N(\text{ansatz}=0, \text{aux}=0)}{N(\text{aux}=0)}
$$

- This gives an estimate of $|\braket{b|\Psi(\omega)}|^2$, the fidelity between the target and current ansatz state.
- The result depends on how often both the ansatz and auxiliary qubits are measured to be in state 0, out of the total
shots where the auxiliary qubit is 0.

---

## Cost Function

- The cost is computed as:

$$
C(\omega) = 1 - P(\text{ansatz}=0 \mid \text{aux}=0)
$$

- This cost function is minimized with respect to the parameters $\omega$.
- Internally, the cost is evaluated by:
  - Generating a dictionary of parameters from the input vector $\omega$.
  - Sampling the quantum circuit with those parameters.
  - Computing the cost from the conditional probability.
  - Storing the result in an internal dictionary to track optimization progress.

---

## Optimization

- A classical optimizer (COBYLA) is used to minimize the cost function.
- Initial values of parameters $\omega$ are randomly chosen in the range [0.0, 3.0].
- A maximum of 2000 iterations is allowed.
- During optimization:
  - The cost function is repeatedly evaluated.
  - Intermediate results are stored for plotting.
- After optimization:
  - The optimal parameter values $\omega^*$ are returned in dictionary format:

    - keys: `params_n`
    - values: $\omega_1^*$
  

- A plot of cost vs iteration is generated to visualize convergence.

---

## Summary

- **Quantum step**: Execute a parameterized circuit, post-select on the auxiliary qubit, and compute the fidelity with the target state.
- **Classical step**: Use a classical optimizer (COBYLA) to iteratively update parameters to minimize the cost function.
- The process returns parameters that prepare a quantum state $\ket{\Psi(\omega^*)} \approx \ket{x}$, the solution to the linear system.


