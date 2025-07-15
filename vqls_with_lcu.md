# Variational Quantum Linear Solver (VQLS) with Linear Combination of Unitaries (LCU) Block Encoding 


The Variational Quantum Linear Solver (VQLS) is a quantum algorithm that harnesses the power of Variational Quantum Eigensolvers (VQE) to solve systems of linear equations efficiently:

* **Input:** A matrix $\textbf{A}$ and a known vector $|\textbf{b}\rangle$.

* **Output:** An approximation of a normalized solution  $|x\rangle$ proportional to $|\textbf{x}\rangle$, satisfying the equation $\textbf{A} |\textbf{x}\rangle = |\textbf{b}\rangle$.

***


While the output of VQLS mirrors that of the HHL Quantum Linear-Solving Algorithm, VQLS distinguishes itself by its ability to operate on Noisy Intermediate-Scale Quantum (NISQ) computers. In contrast, HHL necessitates more robust quantum hardware and a larger qubit count, despite offering a faster computation speedup.


This tutorial covers an implementation example of a **Variational Quantum Linear Solver** [[1](#VQLS)] using block encoding. In particular, we use linear combinations of unitaries (LCUs) for the block encoding.

As with all variational algorithms, the VQLS is a hybrid algorithm in which we apply a classical optimization on the results of a parametrized (ansatz) quantum program.


## Building the Algorithm with Classiq


### Quantum Part: Variational Circuit


Given a block encoding of the matrix A:
$$
\begin{aligned}
U = \begin{bmatrix} A & \cdot \\ \cdot & \cdot \end{bmatrix}
\end{aligned}
$$

we can prepare the state 
$$|\Psi\rangle :=  A |x\rangle/\sqrt{\langle x |A^\dagger A |x\rangle}.$$


We can approximate the solution $|x\rangle$ with a variational quantum
circuit, i.e., a unitary circuit $V$, depending on a finite number of classical real parameters $w = (w_0, w_1, \dots)$:

$$|x \rangle = V(w) |0\rangle.$$


Our objective is to address the task of preparing a quantum state $|x\rangle$ such that $A |x\rangle$ is proportional to $|b\rangle$; or, equivalently, ensuring that

$$|\Psi\rangle :=  \frac{A |x\rangle}{\sqrt{\langle x |A^\dagger A |x\rangle}} \approx |b\rangle.$$

The state $|b\rangle$ arises from a unitary operation $U_b$ applied to the ground state of $n$ qubits; i.e.,

$$|b\rangle = U_b |0\rangle.$$

To maximize the overlap between the quantum states $|\Psi\rangle$ and $|b\rangle$, we optimize the parameters, defining a cost function:

$$C = 1- |\langle b | \Psi \rangle|^2.$$


At a high level, the above could be implemented as follows:

We construct a quantum model as depicted in the figure below. When measuring the circuit in the computational basis, the probability of
finding the system qubits in the ground state (given the ancillary qubits measured
in their ground state) is
    $|\langle 0 | U_b^\dagger |\Psi \rangle|^2 = |\langle b | \Psi \rangle|^2.$


![](./images/figure1.png)  



To block encode a Variational Quantum Linear Solver as explained above, we can define a high-level `block_encoding_vqls` function as follows:


From here, we only need to define `ansatz`, `block_encoding`, and `prepare_b_state` to fit the specific example above. Now we are ready to build our model, synthesize it, execute it, and analyze the results.


### Classical Part: Finding Optimal Parameters


To estimate the overlap of the ground state with the post-selected
state, we could directly make use of the measurement samples. However,
since we want to optimize the cost function, it is useful to express
everything in terms of expectation values through Bayes\' theorem:

$$|\langle b | \Psi \rangle|^2=
P( \mathrm{sys}=\mathrm{ground}\,|\, \mathrm{anc} = \mathrm{ground}) =
P( \mathrm{all}=\mathrm{ground})/P( \mathrm{anc}=\mathrm{ground})$$

To evaluate the conditional probability from the above equation, we construct the following utility function to operate on the measurement results:




To variationally solve our linear problem, we define the
cost function $C = 1- |\langle b | \Psi \rangle|^2$ that we are going to
minimize. As explained above, we express it in terms of expectation
values through Bayes\' theorem.

We define a classical function that gets the quantum program, minimizes the cost function using the COBYLA optimizer, and returns the optimal parameters.


***
Once the optimal variational weights `w` are found, we
can generate the quantum state $|x\rangle$. By measuring $|x\rangle$ in
the computational basis we can estimate the probability of each basis
state.
***


## Example Using LCU Block Encoding

We treat a specific example based on a system of three qubits:

$$
A  =  c_0 A_0 + c_1 A_1 + c_2 A_2 = \ 0.55 \mathbb{I} \ + \ 0.225 Z_2 \ + \ 0.225 Z_3
$$
$$
|b\rangle = U_b |0 \rangle = H_0  H_1  H_2 |0\rangle
$$

where $Z_j, X_j, H_j$ represent the Pauli $Z$, Pauli $X$, and Hadamard
gates applied to the qubit with index $j$.


### Block Encoding the Matrix A with LCU


Our first goal is to encode the Matrix A on a quantum circuit.  

Using *linear combinations of unitaries* (LCU) we can encode the operator A inside U as in Eq.([1](#mjx-eqn-1)). This block is defined by the subspace of all states where the auxiliary qubits are in state $|0\rangle$.


Given a $2^n \times 2^n$ matrix $A$, representable as a linear combination of $L$ unitary matrices $A_0, A_1, \dots, A_{L-1}$ we can implement a quantum circuit that applies the associated operator.

$$A = \sum_{l=0}^{L-1} c_l A_l,$$

where $c_l$ are arbitrary complex numbers. Crucially, we assume that each unitary $A_l$ can be efficiently implemented using a quantum circuit acting on $n$ qubits.







The concept of applying a linear combination of unitary operations has been explored in Ref[[2](#coherent)],  and we adopt a similar approach here.

We can assume, without loss of generality, that the coefficients $c=(c_1, c_2, \dots c_L)$  in the definition of $A$ form a positive and normalized probability distribution:

$$c_l \ge 0 \quad \forall l,  \qquad \sum_{l=0}^{L-1} c_l=1.$$


The complex phase of each coefficient $c_l$ can be absorbed into its associated unitary $A_l$, resulting in a vector comprised of positive values. Additionally, as the linear problem remains unchanged under constant scaling, we can normalize the coefficients to form a probability distribution.


To simplify matters, we assume that $L$ is a power of 2, specifically $L=2^m$ for a positive integer $m$. (We can always pad $c$ with additional zeros if needed.)

Let us consider a unitary circuit $U_c$, embedding the square root of
$c$ into the quantum state $|\sqrt{c}\rangle$ of $m$ ancillary qubits:

$$|\sqrt{c} \rangle =  U_c |0\rangle = \sum_{l=0}^{L-1} \sqrt{c_l} | l \rangle,$$

where $\{ |l\rangle \}$ is the computational basis of the ancillary
system.

Now, for each component $A_l$ of the problem matrix $A$, we can define
an associated controlled unitary operation $CA_l$, acting on the system
and on the ancillary basis states as follows:

$$\begin{aligned}
CA_l \, |j\rangle |l' \rangle  =
\Bigg\{
\begin{array}{c}
\left(A_l \otimes \mathbb{I}\right) \; |j\rangle |l \rangle \quad \; \mathrm{for}\; l'=l \\
\qquad \qquad |j\rangle |l' \rangle  \quad \mathrm{for}\; l'\neq l
\end{array},
\end{aligned}$$

i.e., the unitary $A_l$ is applied only when the ancillary is in
the corresponding basis state $|l\rangle$.

So our LCU quantum circuit looks as follows:
![Screenshot 2024-05-19 at 18.56.22.png](attachment:9b0897af-b689-4b83-b119-f7de76db95fd.png)






Let's apply the previous theory on our simple example based on a system of three qubits:


First we need to define unitary operations associated to the simple
example presented above.

The coefficients of the linear combination are three positive numbers
$(0.55, 0.225, 0.225)$. So we can embed them in the state of $m=2$ ancillary
qubits by adding a final zero element and normalizing their sum to $1$:





To block encode the A we only need these things:
1. Representation of it as a linear combination of $L$ unitary matrices (in our case we use Pauli matrices)
2. A unitary circuit $U_c$ that embeds the square root of $c$ (linear combination coefficients) into the quantum state
3. A circuit that encodes $CA_l$


#### Preparing $U_c$  
To construct $\vec{c}$ from the above example we want to apply our Pauli's strings decomposition. For this, we need additional functions:


Now we can construct $\vec{c}$:


To prepare $U_c$ we need to embed the square root of the probability distribution `c`
into the amplitudes of the ancillary state:


#### LCU Sequence of All Controlled Unitaries $CA_l$

Next, we are left to define the LCU sequence of all controlled unitaries $CA_l$,
acting as $A_l$ on the system whenever the ancillary state is
$|l\rangle$. 
The `prepare_ca` function iterates over a list of Pauli terms $A_l$ and applies $CA_l$ on the three-qubit state `phi` controlled by the ancillary state $\vec{c}$: 


### Fixed Hardware Ansatz

Let's consider our ansatz $V(w)$, such that 

$$|x\rangle = V(w) |0\rangle.$$


This allows us to "search" the state space by varying a set of parameters, $w$. 

The ansatz that we use for this three-qubit system implementation takes in nine parameters as defined in the `apply_fixed_3_qubit_system_ansatz` function:


To view our ansatz implementation we create a model and view the synthesis result: 


![Screenshot 2024-05-21 at 10.31.17.png](attachment:a8ec9628-7bf1-4b52-b5ca-3a23bfa29dd6.png)


This is called a **fixed hardware ansatz** in that the configuration of quantum gates remains the same for each run of the circuit, and all that changes are the parameters. Unlike the QAOA ansatz, it is not composed solely of Trotterized Hamiltonians. The applications of $Ry$ gates allow us to search the state space, while the $CZ$ gates create "interference" between the different qubit states.


### Running the VQLS

Now, we can define the main function: we call `block_encoding_vqls` with the arguments of our specific example.


Constructing the model, synthesizing, and executing on the Classiq simulator:


![Screenshot 2024-05-19 at 18.59.56.png](attachment:462bbae5-5d4b-4625-9b2b-3d90808d62fe.png)


We run the classical optimizer to get the optimal parameters:


### Measuring the Quantum Solution

Finally, we can apply the optimal parameters to measure the quantum results for $\vec{x}$:


### Comparing to the Classical Solution

Since the specific problem considered in this tutorial has a small size,
we can also solve it in a classical way and then compare the results
with our quantum solution.


We use the explicit matrix representation in terms of numerical NumPy arrays.


Classical calculation:


Calculating the classical $\vec{x}$ that solves the equation:


To compare the classical to the quantum results we compute the post-processing by applying $A$ to our optimal vector $|\psi\rangle_o$, normalizing it, then calculating the inner product squared of this vector and the solution vector, $|b\rangle$! We can put this all into code as follows:


The classical cost function basically agrees with the algorithm result.


## References

<a name='VQLS'>[1]</a>: [Bravo-Prieto et al.,Variational Quantum Linear Solver, 2020.](https://arxiv.org/pdf/1909.05820.pdf)


<a name='coherent'>[2]</a>: Robin Kothari. \"Efficient algorithms in quantum query complexity.\"
    PhD thesis, University of Waterloo, 2014.




