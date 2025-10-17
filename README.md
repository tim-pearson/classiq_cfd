# **Hybrid Quantum-Classical Pressure Solvers in CFD with Classiq’s VQLS**

### Timothy Edward Pearson, Alexandre Gallardo

---

## Overview

This project explores the use of **Variational Quantum Linear Solvers (VQLS)**
for
simplified CFD pressure solves. The implementation is done within the **Classiq
framework**,
leveraging its tools for constructing quantum circuits, variational ansatz, and
block-encoded Hamiltonians. The focus is on replacing the classical
Poisson linear solver with a quantum-enhanced solver, benchmarking its
performance and accuracy on small, controlled systems.

We currently simulate small linear systems $Ax = b$, where **A** is represented
as a combination of Pauli operators and **b** is a simple superposition vector.

For more details on the implementation and parameter optimization workflow, see
[Classiq VQLS with LCU: Classical Part - Finding Optimal
Parameters](https://docs.classiq.io/latest/explore/algorithms/vqls/lcu_vqls/vqls
_with_lcu/#classical-part-finding-optimal-parameters).

---