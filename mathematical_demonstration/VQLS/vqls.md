# Mathematical Foundations of VQLS with LCU Block Encoding

### Timothy Pearson

## Goal

We aim to solve a system of linear equations in the form:
$$Ax=b$$

or the quantum notation:

$$A \ket{x} = \ket{b}$$

## Conditions on $A$

However we need to make sure that $A$ is **Unitary** i.e:

$$AA^{\dagger} = I$$

Further more a need to satisfy the **Hermitian** condition:

$$A^{\dagger} = A$$

$A$ could contian complex numbers however must still must satisfy:
$$A_{ij} = A_{ij}^* \quad  i \ne j$$

This ensures the the eigen values are real which si important stability and
unitary embddings.

Quantum Circuits only implement unitaries


## Block Encoding of $A$

The definition of a Block encoding is as follows:
$$\bra{0}_{anc} U \ket{0}_{anc} = A$$

Let:

$$
U_A = \begin{bmatrix}
A & \cdot \\
\cdot & \cdot
\end{bmatrix}
$$

We "complete" $U$ by filling in the $\cdot$'s so that $U_A$ is unitary

We will see further on that the values inside the $\cdot$'s dont effect the
outcome due to **post-selectoin** and projection.



## Preparation of the state

$$\ket{\Psi} = A \ket{x} / \sqrt{\bra{x} A^{\dagger} A \ket{x}}
$$
This is the solution state we are looking for (theoretically
$\ket{\Psi} = \ket{b}$)

We can approximate $\ket{x}$ with a **Variantional Unitary Circuit**: $V$

$V(w)$ varies with $w= (w_0, w_1, \ldots)$ and:

$$\ket{x} = V(w)\ket{0}$$

## Preparing $\ket{b}$

We aim to prepare $\ket{x}$ such that:

$$\ket{\Psi} = \frac{A \ket{x}}{\sqrt{\bra{x}A^\dagger A \ket{x}}} \propto
\ket{b} $$
where we can create/preprare:
since, for any quantum state $\ket{b}, \quad  \exists \text{ unitary } U_b$ such
that: 

$$\ket{b} = U_b \ket{0} 
$$

$$\ket{b} =\sum_{i=0}^{2^n -1} c_i \ket{i}$$

with:

$$\sum_{i} |c_i|^2 = 1$$

## Cost Function

The probabilty we want to maximize is:
$$|\braket{b|\Psi}|^2$$
which is the prpbabilty of measuring $\ket{b}$ give we are in state $\ket{\Psi}$
essentially the fidelity/overlap between the two states 

So the cost function we want to minimize is 

$$
C = 1 - |\braket{b|\Psi}|^2
$$

## Quantum Circuit Implementation

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
![Diagram](/home/tim/.cache/nvim/circuits/diagram.png)


$$
\ket{\psi_0} =  \ket{0}_{anc} \otimes \ket{0}_{sys}
$$

After applying $V(w)$ to $\ket{0}_{sys}$ to obtain $\ket{x}$


$$
\begin{align*}
\ket{\psi_1} &=  \ket{0}_{anc} \otimes V(w)\ket{0}_{sys} \\
&=  \ket{0}_{anc} \otimes \ket{x} \\
&= \ket{0 \ldots 0}_{anc} \otimes \ket{x} \\
&= \ket{0 \ldots 0}_{anc} \otimes \ket{x} + \sum_{j \ne 0} \ket{j} \otimes 0 \\
\end{align*}
$$
this can also be represented as a block vector:

$$
\ket{\psi_1}= \begin{bmatrix} \ket{x} \\ \vdots \\ 0 \end{bmatrix}
$$


- The first row conresponds to ancilla $= \ket{0}$
- The second row conresponds to ancilla $= \ket{1}$


Apply block encoding of A $U_A$:

Now lets say (for 2D):

$$
U_A = \begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
$$

$$
U\begin{bmatrix} \ket{x} \\ 0 \end{bmatrix} = \begin{bmatrix} A & B\\ C & D
\end{bmatrix} = \begin{bmatrix} \ket{x} \\ 0 \end{bmatrix} \begin{bmatrix}
A\ket{x} \\ C \ket{x} \end{bmatrix}
$$

$$
\begin{align*}
\ket{\psi_2}&= \ket{0}_{anc} \otimes A V(w) \ket{0}_{sys} + \ket{1}_{anc}
\otimes
\ket{\phi_{\perp}} \\
&= \ket{0}_{anc} \otimes A \ket{x} + \ket{1}_{anc} \otimes \ket{\phi_{\perp}}
\end{align*}
$$

Now we perform **Post-Selection** and **project** the **ancillary** qubits to
the $\ket{0}$ state

we can define:

$$P_0 = \ket{0^m}\bra{0^m}_{anc} \otimes I_{sys}$$

$$
P_0 \ket{\psi_2} = \ket{0^m}_{anc} \otimes A \ket{x}_{sys}
$$

the **Sucess Probabilty** of the post selection is:
$$p_{succ} =|\bra{\psi_2} P_0\ket{\psi_2}|^2 = \bra{x} A^\dagger A \ket{x}
$$

(now we only are internested in the runs where the ancilla measure to
$\ket{0^m}$)

Normalize the system state after sucessful post-selection:

> we can drop out the explicit ancilla regeister since we its fixed at
$\ket{0^m}$

Conditioned on observing ancilla = $\ket{0^m}$

$$\ket{\Psi} _{sys} = \frac{A \ket{x}}{ p_{succ}} = \frac{A
\ket{x}}{\sqrt{\bra{x} A^\dagger A \ket{x}}}
$$

## Evaluating the Cost function

Now to evalued the cost function:

$$
\begin{align*}
C &= 1 - |\braket{b|\Psi}|^2 \\
&= 1 - |\bra{0} U_b^\dagger \ket{\Psi}|^2 \\
\end{align*}
$$

this is the **probabilty** of the $sys=\ket{0}$ ( after applying $U_b$)  given
$anc$ **post selected**
to $\ket{0^m}$

## Interpret the results

After sampling the `ExecutionSession` we can obtain a dataframe of results with
columns:

- **io**: bitstrings representing the basis outcomes (e.g. "010", "011")
- **amplitude**: complex statevector amplitudes associated with the baisi state
- **probabilties/counts**: derived from amplitudes if sampled.

> So `self.results` is essentially the measured ansatz state expressed in the
computational basis.

Since quantum states are only defined up to a global phase, we choose any of
the amplitudes as a reference.

Then we can "remove" the global phase by dividing each of the amplitudes by
$e^{i \theta}$ so the amplitudes are aligned with the classical solution’s
phase convention.

Now we want the last amplitude to be positive (a standard convention to fix the
sign ambiguity).

If the last amplitude is negative, we flip the sign of the entire vector.
This ensures a unique, consistent representation of the statevector.

Finally, we square the amplitudes to get the `quantum_probs` and we can
calculate the overlap


## Interpret the Results

After running the quantum circuit, the measurement (or statevector) outcomes
can be represented as a dataframe (or table) with entries such as:

- **io**: computational basis states (bitstrings like "010", "011")
- **amplitude**: complex amplitudes associated with each basis state
- **probabilities/counts**: estimated from amplitudes if we sample many shots

Thus, the quantum output corresponds to the prepared ansatz state expressed in
the computational basis.

Since quantum states are only defined up to a **global phase**, we can remove
this ambiguity by dividing all amplitudes by a reference phase factor. For a
chosen amplitude with phase $\theta$:

$$
\ket{\psi} \;\mapsto\; \frac{\ket{\psi}}{e^{i\theta}}
$$

This aligns the state with a classical reference convention.

Additionally, there is a **sign ambiguity**: if the entire statevector is
multiplied by $-1$, physical predictions remain unchanged. To fix this, one
typically requires a particular amplitude (often the last component) to be
positive. If it is negative, the whole vector is multiplied by $-1$:

$$
\ket{\psi} \;\mapsto\; -\ket{\psi} \quad \text{if reference amplitude } < 0
$$

With these conventions fixed, the amplitudes become uniquely defined. The
**probability distribution** over basis states is then obtained from the Born
rule:

$$
p_i = |\langle i | \psi \rangle|^2
$$

These probabilities, denoted as `quantum_probs`, can be compared against the
classical solution of the linear system to evaluate the overlap:

$$
\text{overlap} = |\langle b | \Psi \rangle|^2
$$

This quantity measures how close the variationally prepared quantum state is to
the normalized solution of the linear system.


### Summary fo Circiut

1. Prepare system and ancilla:  
   - System: $|x⟩ = V(w)|0⟩_{sys}$  
   - Ancilla: $|0^m⟩$  

2. Apply block-encoding:  
   - Apply $U_A$ on $(|0^m⟩_{anc} \otimes |x⟩_{sys})$  

3. Measure ancilla:  
   - Keep only shots with outcome $0^m$  
   - Success probability: $p_{succ} = ⟨x| A^\dagger A |x⟩$  
   - Conditioned system state: $|Ψ⟩$  

4. Apply $U_b^\dagger$ to the system.  

5. Measure the system in the computational basis:  
   - The frequency of $|0⟩$ among the kept shots estimates $|⟨b|Ψ⟩|^2$  

6. Compute cost function:  
   - $C = 1 - |⟨b|Ψ⟩|^2$  
   - Update parameters $w$.  
