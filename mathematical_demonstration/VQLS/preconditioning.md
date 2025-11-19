# Incomplete Cholesky Preconditioning


**Incomplete Cholesky Preconditioning for VQLS**

Let $A \in \mathbb{C}^{n \times n}$ be Hermitian positive definite. The incomplete Cholesky decomposition computes approximate factor $L$ such that:
$$A \approx LL^T$$

For VQLS, we maintain symmetry via split preconditioning:
$$M^{-1}AM^{-T} = (L^{-1}A)(L^{-T})^T$$

The preconditioned system becomes:
$$A_{\text{pre}}x_{\text{pre}} = b_{\text{pre}}$$
where:
$$A_{\text{pre}} = L^{-1}AL^{-T},\quad b_{\text{pre}} = L^{-1}b,\quad x = L^{-T}x_{\text{pre}}$$

**Example**: 2D Poisson with $3\times3$ grid:
$$A = \begin{bmatrix}
4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
-1 & 4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 \\
0 & -1 & 4 & 0 & 0 & -1 & 0 & 0 & 0 \\
-1 & 0 & 0 & 4 & -1 & 0 & -1 & 0 & 0 \\
0 & -1 & 0 & -1 & 4 & -1 & 0 & -1 & 0 \\
0 & 0 & -1 & 0 & -1 & 4 & 0 & 0 & -1 \\
0 & 0 & 0 & -1 & 0 & 0 & 4 & -1 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & -1 & 4 & -1 \\
0 & 0 & 0 & 0 & 0 & -1 & 0 & -1 & 4
\end{bmatrix}$$

Incomplete Cholesky factor (dropping fill-in):
$$L = \text{tril}(A) = \begin{bmatrix}
4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
-1 & 4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & -1 & 4 & 0 & 0 & 0 & 0 & 0 & 0 \\
-1 & 0 & 0 & 4 & 0 & 0 & 0 & 0 & 0 \\
0 & -1 & 0 & -1 & 4 & 0 & 0 & 0 & 0 \\
0 & 0 & -1 & 0 & -1 & 4 & 0 & 0 & 0 \\
0 & 0 & 0 & -1 & 0 & 0 & 4 & 0 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & -1 & 4 & 0 \\
0 & 0 & 0 & 0 & 0 & -1 & 0 & -1 & 4
\end{bmatrix}$$

Then:
$$A_{\text{pre}} = L^{-1}AL^{-T},\quad b_{\text{pre}} = L^{-1}b$$

**Key properties**:
- $A_{\text{pre}}$ remains Hermitian: $(L^{-1}AL^{-T})^H = L^{-1}A^HL^{-T} = A_{\text{pre}}$
- Condition number $\kappa(A_{\text{pre}}) \ll \kappa(A)$
- Pauli decomposition complexity reduced via sparsity pattern preservation



**Mathematical Completion**

The preconditioned system satisfies:
$$A_{\text{pre}} = L^{-1}AL^{-T} = I + E$$
where $\|E\|_F \ll \|A\|_F$ due to the approximation $A \approx LL^T$.

**Pauli Decomposition Impact**:
Original $A$ requires many Pauli terms:
$$A = \sum_{j=1}^m \alpha_j P_j,\quad m = O(n^2)$$

After preconditioning:
$$A_{\text{pre}} = \sum_{j=1}^{m'} \beta_j P_j,\quad m' \ll m$$

The coefficients satisfy:
$$\sum |\beta_j|^2 = \|A_{\text{pre}}\|_F^2 \approx \|I\|_F^2 = n$$

With tolerance $\tau$, terms with $|\beta_j| < \tau$ are discarded, yielding sparse Pauli representation.

**Final Solution Recovery**:
VQLS minimizes $\|A_{\text{pre}}x_{\text{pre}} - b_{\text{pre}}\|$ to obtain $x_{\text{pre}}^*$, then:
$$x^* = L^{-T}x_{\text{pre}}^*$$

The condition number improvement:
$$\kappa(A_{\text{pre}}) = \frac{\lambda_{\max}(A_{\text{pre}})}{\lambda_{\min}(A_{\text{pre}})} \approx \sqrt{\kappa(A)}$$

accelerates VQLS convergence while maintaining exact Hermiticity for quantum processing.
