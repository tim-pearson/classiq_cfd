# VQLS CFD Application PFEE Classiq Status

**3-qubit Pauli structure results & Poisson pressure solve for incompressible
flows**  

Since the last meeting I have:
- implemented and tested different 2 qubit system fixed hardware ansatz
- constructed the 1d Possion pressure solve system
- created a small interactive python script that visually represents the 1d
pressure system and solve it classically
- research into Possion specifc decompositions


## Resuts
![MISSING IMAGE](./images/A_covergence.png)  
*Figure n: --*

Initial parameters: [0.125, -0.213, 0.089, 0.047, -0.25, 0.165, -0.145, 0.233, 0.133, -0.181, -0.087, -0.068]

CLASSIQS Overlap 0.8006995604146314
VQLS amplitudes: [-0.31158684  0.57683022 -0.63987099  0.39749872]

VQLS achieved cost: 0.210364
b · (A|x_vqls⟩/||A|x_vqls⟩||): 0.894818



## Ansatz (2)

```python
def ansatz_two_qubit(angles: CArray[CReal], system_qubits: QArray[QBit]):
    """Enhanced 2-qubit ansatz, 12 parameters"""
    # Layer 1: U3 rotations
    U(angles[0], angles[1], angles[2], 0, system_qubits[0])
    U(angles[3], angles[4], angles[5], 0, system_qubits[1])
    CX(system_qubits[0], system_qubits[1])
    # Layer 2: RY + CZ
    RY(angles[6], system_qubits[0])
    RY(angles[7], system_qubits[1])
    CZ(system_qubits[1], system_qubits[0])
    # Layer 3: Final rotations
    RZ(angles[8], system_qubits[0])
    RZ(angles[9], system_qubits[1])
    RY(angles[10], system_qubits[0])
    RY(angles[11], system_qubits[1])
```
\[
\Qcircuit @C=1em @R=1.2em {
  \lstick{\ket{0}} & \gate{U(\alpha_0,\alpha_1,\alpha_2)} & \ctrl{1} &
\gate{R_y(\alpha_6)} & \targ & \gate{R_z(\alpha_8)} & \gate{R_y(\alpha_{10})} &
\qw \\
  \lstick{\ket{0}} & \gate{U(\alpha_3,\alpha_4,\alpha_5)} & \targ &
\gate{R_y(\alpha_7)} & \ctrl{-1} & \gate{R_z(\alpha_9)} &
\gate{R_y(\alpha_{11})} & \qw
}
\]
![Diagram](/home/tim/.cache/nvim/circuits/ansatz.png)

---

## 1-D Poisson matrix
![MISSING IMAGE](./images/classic_correction.png)  
*1D Velocity-Divergence Plot*

Discrete Laplacian (interior points, $\Delta x=1$):
$$
-\Delta p_i \approx -(p_{i-1} - 2p_i + p_{i+1})
$$

- Comes from finite differences and tailor expansion $p$ 

Matrix form $A p = b$:
$$
A = \begin{bmatrix}
2 & -1 & 0 & \cdots \\
-1 & 2 & -1 & \cdots \\
0 & -1 & 2 & \cdots \\
\vdots & & \ddots & \ddots
\end{bmatrix}_{n\times n}
$$
RHS $b$ =  diverence values for incompressible fluids: $\nabla \cdot u = 0$.

---



## Next steps
- implement Possion specific block encoding for tridiagonal matricies
- create a 2 qubit ansatz with fewer parameters
- create a 4 qubit ansatz (allowing for 16 x 16 Poisson system)

