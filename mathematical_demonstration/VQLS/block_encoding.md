Creating an **LCU (Linear Combination of Unitaries) block encoding by hand**
boils down to one core mathematical idea:

# ⭐ **Represent a non-unitary operator as a weighted sum of unitaries, then use an ancilla to select and recombine them coherently.**

Below is the clean, structured mathematical explanation typically used in
quantum algorithms (HHL, Hamiltonian simulation, QSP/QSVT preprocessing, etc.).

---

# 🧠 **1. Start with the target operator**

Suppose you want to block-encode an operator (A), which is **not** unitary in
general.

You need to express it as a linear combination:

[
A = \sum_{k=0}^{m-1} \alpha_k U_k
]

where

* (U_k) are **unitaries**
* (\alpha_k \ge 0) are **positive coefficients**
* ( \lambda = \sum_k \alpha_k ) is the normalization (the block encoding scale)

This decomposition is the heart of the LCU method.

---

# 🧠 **2. Prepare a weighted superposition over the indices**

You introduce an **ancilla state** that encodes the coefficients:

[
|\psi\rangle
= \frac{1}{\sqrt{\lambda}}
\sum_{k=0}^{m-1} \sqrt{\alpha_k}, |k\rangle.
]

This "state preparation unitary" is usually called:

[
P: |0\rangle \mapsto |\psi\rangle.
]

---

# 🧠 **3. Controlled selection of the unitaries**

Define the **select unitary**:

[
\mathrm{SELECT}(U) = \sum_{k=0}^{m-1} |k\rangle\langle k| \otimes U_k.
]

This applies (U_k) to the **system register** depending on the **ancilla
index**.

---

# 🧠 **4. Put the pieces together**

The LCU block encoding circuit is:

[
W = (P^\dagger \otimes I),
\mathrm{SELECT}(U),
(P \otimes I).
]

---

# ⭐ **5. Why does this work? (The core block-encoding identity)**

Compute:

[
\langle 0|W|0\rangle
= \frac{1}{\lambda} A.
]

Here’s the derivation:

[
W(|0\rangle \otimes |\phi\rangle)
=================================

(P^\dagger \otimes I)
\left(
\sum_{k} \frac{\sqrt{\alpha_k}}{\sqrt{\lambda}} |k\rangle \otimes U_k
|\phi\rangle
\right)
]

Project onto (\langle 0|):

[
\langle 0|W|0\rangle
====================

\frac{1}{\lambda}
\sum_{k} \alpha_k U_k
=====================

\frac{A}{\lambda}.
]

Thus (W) is a **unitary block encoding** of (A / \lambda).

---

# 🎯 THE MATHEMATICAL IDEA IN ONE SENTENCE

> **Embed the operator as a weighted coherent superposition of unitaries, use
an ancilla to select and recombine them, and project onto the ancilla |0⟩ to
pull out the normalized operator.**

---

# 🧩 Optional: How to find the unitaries (U_k) “by hand”

To engineer this decomposition manually, you typically:

1. **Split the operator into Hermitian and anti-Hermitian parts**
   [
   A = B + iC,\quad B, C\ \text{Hermitian}
   ]
2. **Decompose each into Pauli terms**
   [
   B = \sum_j \beta_j P_j,\quad C = \sum_j \gamma_j P_j
   ]
   (each (P_j) is a tensor product of Pauli matrices and is already unitary)
3. Use
   [
   A = \sum_j \alpha_j U_j,
   \quad
   U_j \in {\pm P_j, \pm i P_j}.
   ]

This works for Hamiltonians, density matrices, matrix encodings, etc.

---

# If you want, I can also show:

* A step-by-step construction for a **specific matrix**
* A complete “by hand” example (e.g., a 2-qubit Hamiltonian)
* How this ties into **QSVT / quantum signal processing**
* A full **block-encoding diagram**

Just tell me!