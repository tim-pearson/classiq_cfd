Below is the **complete list of practically useful cost functions** used in *Variational Quantum Linear Solvers (VQLS)* — with explanations, pros/cons, and when to use each.
These come from the original **Bravo-Prieto et al. (2020)** VQLS paper, its later refinements, and modern quantum-ML practice.

---

# ✅ **Core Idea**

Given a linear system
$$
A x = b,
$$
VQLS prepares a quantum state $|x\rangle$ using a parameterized circuit.
The goal is to minimize a **cost function** $C(|x\rangle)$ such that the minimum corresponds to $|x\rangle \propto A^{-1} |b\rangle$.

---

# ⭐ **1. Global Hilbert–Schmidt Cost (HS Cost)**

*(Original VQLS cost)*

$$
C_{\text{HS}} = 1 - \frac{|\langle x | A | b \rangle|^2}{\langle x |A^\dagger A| x \rangle}
$$

Equivalent classical interpretation:

* We want (|x\rangle) to minimize
  $|A|x\rangle - |b\rangle|^2$

Properties:

* **Convex in an ideal space**, but generally **not** convex in circuit parameters
* **Robust** even for non-Hermitian (A)
* Requires estimation of overlaps like
  $\langle x | A^\dagger A | x \rangle$ → expensive

Use this when:

* You want the *canonical* VQLS cost
* Matrix $A$ is arbitrary (Hermitian or not)
* You can afford deeper circuits / more measurements

---

# ⭐ **2. Local Hilbert–Schmidt Cost (LHS Cost)**

(Used to reduce shot noise and circuit depth)

$$
C_{\text{local}} = \sum_{k=1}^{N} \left(
1 -
\frac{|\langle x|A_k|b\rangle|^{2}}
{\langle x|A_k^\dagger A_k|x\rangle}
\right)
$$

where
$$
A = \sum_{k=1}^N A_k,
$$
with each $A_k$ a Pauli term.

Advantages:

* **Lower variance**
* Faster convergence
* Works well for local Hamiltonians / sparse matrices

Use this when:

* $A$ is decomposed into Pauli operators
* You want a noise-friendly cost
* Experimental implementation is required

---

# ⭐ **3. The Fidelity / Overlap Cost**

Only valid when you *can prepare the exact state* $|x_{\text{ideal}}\rangle$ for benchmarking, so **not used in real problems**, only tests.

$$
C_\text{fid} = 1 - |\langle x | x_{\text{ideal}} \rangle|^2
$$

Use this **only**:

* For simulation validation
* When comparing ansatz quality

Not usable for real-world linear systems.

---

# ⭐ **4. The Residual Cost (preferred for classical preconditioners)**

This comes from rewriting
$$
A |x\rangle \approx |b\rangle
$$
and minimizing the residual norm:

$$
C_{\text{res}} = | A |x\rangle - |b\rangle |^2
$$

Expanded:

$$
C_{\text{res}}
= \langle x| A^\dagger A |x\rangle * 2 \Re\langle b|A|x\rangle - \langle b|b\rangle
$$

This is very similar to the HS cost but without normalization.

Advantages:

* Simple measurement structure
* Natural connection to classical preconditioning
* Effective when $A$ is Hermitian / SPD (like Poisson matrices!)
* Works great with Krylov-inspired ansatz

Use this when:

* Your system comes from PDEs (Poisson, diffusion, FEM, etc.)
* You apply **preconditioners**:
  solve $M^{-1} A x = M^{-1} b$
* You want the most “classical” cost analog

---

# ⭐ **5. “Projected Residual” Cost (best for PDE matrices like Poisson)**

Instead of full residual, you test residual on a set of basis test states (|i\rangle):

$$
C_{\text{proj}} = \sum_i | \langle i | (A|x\rangle - |b\rangle) |^2
$$

Idea:

* Project the residual on computational basis
* Measurement cost is **much lower**
* Works extremely well for **sparse, structured matrices** (Poisson, Helmholtz, CFD matrices)

Use this when:

* Solving **2D/3D Poisson equations**
* Matrix is **large but structured**
* Want minimal circuit depth & measurement overhead

---

# ⭐ **6. Quantum Natural Gradient / Geometry-Aware Costs**

These are not “cost functions” but improved optimization methods:

* Fisher Information Metric
* Quantum Natural Gradient (QNG)
* Geometric VQLS

They drastically stabilize training.

Use these if:

* You experience barren plateaus
* Ansatz is large
* Problem dimension ≥ 6 qubits

---

# ⭐ **7. Variance-Based Cost (Noise-Resilient VQLS)**

$$
C_{\text{var}} = \mathrm{Var}(A|x\rangle - |b\rangle)
$$

Helps avoid noise-induced bias.

Use in:

* NISQ (real hardware)
* Shallow circuits
* Problems dominated by readout noise

---

# 🧩 Summary: Which cost function should YOU use for the **2D pressure Poisson equation**?

The 2D Poisson operator is:

* Hermitian
* SPD
* Sparse
* Local
* Well structured

Thus the **best cost functions** for this specific problem are:

### ✅ BEST CHOICE

**Residual Cost**
$$
C_{\text{res}} = |A|x\rangle - |b\rangle|^2
$$

### 🚀 ALSO EXCELLENT

**Projected Residual Cost** (lowest measurement cost)

### ⚙️ IF MATRIX → Pauli decomposition

**Local Hilbert–Schmidt Cost**

### 🧪 FOR EXPERIMENT / VALIDATION

**Fidelity Cost**

