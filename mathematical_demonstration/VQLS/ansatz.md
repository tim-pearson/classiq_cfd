## Definitions
For a single qubit rotation and controlled-Z:


$$
RY(\theta) = e^{-i \frac{\theta}{2} Y}= 
\begin{bmatrix}
\cos(\tfrac{\theta}{2}) & -\sin(\tfrac{\theta}{2}) \\
\sin(\tfrac{\theta}{2}) &  \cos(\tfrac{\theta}{2})
\end{bmatrix}
$$

$$
CZ_{i,j} = I \otimes \cdots \otimes 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1
\end{bmatrix}_{(i,j)}
\otimes \cdots \otimes I
$$

A layer of single-qubit rotations on $n$ qubits:

$$
L(\theta_0, \ldots, \theta_{n-1}) = \bigotimes_{j=0}^{n-1} RY(\theta_j)
$$

Total circuit unitary (rightmost gate acts first):

$$
U_{\text{total}} = L_3 \, E_2 \, L_2 \, E_1 \, L_1
$$

---

## 2-Qubit Ansatz

Angles: $\theta = [\theta_0, \ldots, \theta_5]$

$$
\begin{aligned}
L_1 &= RY(\theta_0) \otimes RY(\theta_1) \\
E_1 &= CZ_{0,1} \\
L_2 &= RY(\theta_2) \otimes RY(\theta_3) \\
E_2 &= CZ_{1,0} = CZ_{0,1} \\
L_3 &= RY(\theta_4) \otimes RY(\theta_5)
\end{aligned}
$$

**Total unitary:**

$$
\begin{align*}
U_2(\theta)
&= L_3 \, E_2 \, L_2 \, E_1 \, L_1 \\
&= \big( RY(\theta_4) \otimes RY(\theta_5) \big) \\
   & CZ_{0,1} \big( RY(\theta_2) \otimes RY(\theta_3) \big) \\
   & CZ_{0,1} \big( RY(\theta_0) \otimes RY(\theta_1) \big)
\end{align*}
$$

---

## 3-Qubit Ansatz

Angles: $\theta = [\theta_0, \ldots, \theta_8]$

$$
\begin{aligned}
L_1 &= RY(\theta_0) \otimes RY(\theta_1) \otimes RY(\theta_2) \\
E_1 &= CZ_{0,1} \, CZ_{0,2} \, CZ_{1,2} \\
L_2 &= RY(\theta_3) \otimes RY(\theta_4) \otimes RY(\theta_5) \\
E_2 &= CZ_{2,0} \, CZ_{2,1} \, CZ_{1,0} \\
L_3 &= RY(\theta_6) \otimes RY(\theta_7) \otimes RY(\theta_8)
\end{aligned}
$$

**Total unitary:**

$$
\begin{aligned}
U_3(\theta)
&= L_3 \, E_2 \, L_2 \, E_1 \, L_1 \\
&= 
\bigotimes_{j=0}^{2} RY(\theta_{6+j}) \; 
(CZ_{2,0} \, CZ_{2,1} \, CZ_{1,0}) \; \\
&\bigotimes_{j=0}^{2} RY(\theta_{3+j}) \; 
(CZ_{0,1} \, CZ_{0,2} \, CZ_{1,2}) \; \\
&\bigotimes_{j=0}^{2} RY(\theta_{0+j})
\end{aligned}
$$

---

## 4-Qubit Proposed Ansatz

Angles: $\theta = [\theta_0, \ldots, \theta_{11}]$

$$
\begin{aligned}
L_1 &= \bigotimes_{j=0}^{3} RY(\theta_j) \\
E_1 &= CZ_{0,1} \, CZ_{0,2} \, CZ_{0,3} \, CZ_{1,2} \, CZ_{1,3} \, CZ_{2,3} \\
L_2 &= \bigotimes_{j=0}^{3} RY(\theta_{4+j}) \\
E_2 &= CZ_{3,2} \, CZ_{3,1} \, CZ_{3,0} \, CZ_{2,1} \, CZ_{2,0} \, CZ_{1,0} \\
L_3 &= \bigotimes_{j=0}^{3} RY(\theta_{8+j})
\end{aligned}
$$

**Total unitary:**

$$
\begin{aligned}
U_4(\theta)
&= L_3 \, E_2 \, L_2 \, E_1 \, L_1 \\
&= 
\bigotimes_{j=0}^{3} RY(\theta_{8+j}) \;
\left( \prod_{i>j} CZ_{i,j}^{(\text{rev})} \right)
\bigotimes_{j=0}^{3} RY(\theta_{4+j}) \;
\left( \prod_{i<j} CZ_{i,j} \right)
\bigotimes_{j=0}^{3} RY(\theta_{0+j})
\end{aligned}
$$

