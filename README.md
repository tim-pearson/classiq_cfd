# **Hybrid Quantum-Classical Pressure Solvers in CFD with Classiq’s VQLS**

### Timothy Edward Pearson, Alexandre Gallardo

---

## Overview

This project explores the use of **Variational Quantum Linear Solvers (VQLS)** for
simplified CFD pressure solves. The focus is on replacing the classical
Poisson linear solver with a quantum-enhanced solver, benchmarking its
performance and accuracy on small, controlled systems.

We currently simulate small linear systems $Ax = b$, where **A** is represented
as a combination of Pauli operators and **b**. 

---

## Motivation

Solving the Poisson equation in CFD is a computational bottleneck. Quantum
solvers like VQLS could offer new approaches, especially for small-scale
simulations or as testbeds for hybrid methods.

Tridiagonal matrices, common in finite-difference CFD discretizations, can
potentially be exploited with specialized **block-encoding strategies** to
reduce circuit depth, controlled operations, and measurement overhead.

