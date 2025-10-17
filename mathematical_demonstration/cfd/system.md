1. Given intermediate (predicted) face velocities
(u^*_{i+\tfrac12,j},;v^*_{i,j+\tfrac12}).

2. Continuous projection (take divergence):
$$
   \nabla^2 p = \frac{\rho}{\Delta t},\nabla!\cdot! \mathbf{u}^*.
$$

3. Discrete divergence (cell ((i,j)), centered)
$$
   (\nabla\cdot\mathbf{u}^*)_{i,j}
   =\frac{u^**{i+\tfrac12,j}-u^**{i-\tfrac12,j}}{\Delta x}
   +\frac{v^*_{i,j+\tfrac12}-v^*_{i,j-\tfrac12}}{\Delta y}.
$$

4. Discrete Poisson (5-point FD, multiply by (\Delta x^2\Delta y^2) as needed).
Let
   (\alpha_x=\frac{1}{\Delta x^2},;\alpha_y=\frac{1}{\Delta y^2}). Then for
interior ((i,j))
$$
   \alpha_x p_{i+1,j} + \alpha_x p_{i-1,j} + \alpha_y p_{i,j+1} + \alpha_y
p_{i,j-1}
   -2(\alpha_x+\alpha_y), p_{i,j} ;=; b_{i,j},
$$
   with
$$
   b_{i,j} ;=; \frac{\rho}{\Delta t},(\nabla\cdot\mathbf{u}^*)_{i,j}.
$$

5. Flattening index: (k = k(i,j)) (e.g. lexicographic (k=(j-1)N_x+(i-1))). Then
$$
   \sum_{m} A_{k,m}, p_m = b_k,
$$
   with nonzeros per row (interior):
$$
   A_{k,k} = -2(\alpha_x+\alpha_y),;
   A_{k,k_{\pm x}} = \alpha_x,;
   A_{k,k_{\pm y}} = \alpha_y.
$$

6. Boundary rows:

* Dirichlet pressure at boundary cell (r): overwrite row (r) with
(A_{r,r}=1,;b_r=p_{\text{bc}}).
* Neumann (natural) from velocity BCs: incorporate ghost-face values into
(b_{k}) (one-sided difference) or modify stencil coefficients accordingly.

7. Remove nullspace (Neumann total) — enforce reference:

* fix (p_{k_0}=0) (replace that row by identity), or
* impose mean-zero: add constraint (\sum_k p_k = 0) (augment system).

8. Final linear system to solve:
$$
   A p = b,\qquad A\in\mathbb{R}^{N\times N}\text{ (symmetric, sparse, SPD
after ref)},;b\in\mathbb{R}^N.
$$

9. After solution (p), velocity correction (face-centered):
$$
   u^{n+1}*{i+\tfrac12,j} = u^**{i+\tfrac12,j} - \frac{\Delta
t}{\rho},\frac{p_{i+1,j}-p_{i,j}}{\Delta x},
$$
$$
   v^{n+1}*{i,j+\tfrac12} = v^**{i,j+\tfrac12} - \frac{\Delta
t}{\rho},\frac{p_{i,j+1}-p_{i,j}}{\Delta y}.
$$

10. (Optional preconditioning statement as formula)
    If using preconditioner (M), solve
$$
   M^{-1}A p = M^{-1} b
$$
    or use Schur/patch decomposition: split domain into blocks
(A=\begin{bmatrix}A_{11}&A_{12}\A_{21}&A_{22}\end{bmatrix}), solve per-block
systems and assemble Schur complement.

---

That's the math from (\mathbf{u}^*) → assemble (A,b) → solve (Ap=b) → velocity
correction.
