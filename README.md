---
title: Quantum-Enhanced CFD Using Classiq's VQLS and HHL
author: Timothy Edward Pearson, Alexandre Gallardo
---
## Project Title  

**Exploring Hybrid Quantum-Classical Solvers in CFD Using Classiq’s
VQLS and HHL**

## Overview

We aim to investigate how hybrid quantum-classical algorithms can enhance scientific simulations
— particularly in Computational Fluid Dynamics (CFD).

This project will leverage
Classiq’s Variational Quantum Linear Solver (VQLS) and its
implementation of the HHL (Harrow-Hassidim-Lloyd) algorithm. We aim to
benchmark these quantum solvers on simplified fluid simulations,
comparing them to classical numerical solvers.

## Motivation

Linear solvers are a major computational bottleneck in many numerical
simulations, including CFD. With quantum computing advancing rapidly,
we are interested in testing how quantum-enhanced solvers —
specifically VQLS and HHL — perform in small, controlled
fluid simulations.

We have developed a modular 2D Eulerian fluid simulator in Python that
includes:

- A semi-Lagrangian advection solver  
- A pressure projection step that solves a Poisson equation 
- Real-time visualization of velocity and density fields  

This simulator offers a clean and flexible platform for integrating
quantum solvers on reduced problem sizes (e.g., 4×4 or 8×8 grids).

## Objectives

The primary goal is to replace the classical linear solver used in the
pressure projection step with **quantum solvers** (VQLS and HHL) and
compare them to classical methods. Our key objectives are:

- Isolate the linear system $Ax = b$ within the simulation loop  
- Integrate quantum solvers  
- Benchmark the accuracy and performance of both quantum solvers
against each other as well as classical solvers  
- Generate visual comparisons using simulation output (e.g., velocity,
pressure, and density fields)  
- Evaluate the feasibility and usefulness of quantum solvers in
practical simulation workflows  

## Technical Approach

### 1. Quantum Solver Integration

- Identify and extract the linear system (`A`, `b`) from the Poisson
pressure projection step  
- Build a `quantum_solver.py` module that:  
  - Prepares inputs for quantum solvers  
  - Interfaces with Classiq’s VQLS and HHL pipelines via HLL  
  - Returns results in a format compatible with the simulator  
- Base implementations on Classiq’s documentation and HHL workshop  

### 2. Controlled Testing and Benchmarking

- Use small grid sizes (e.g., 4×4, 8×8) to ensure the quantum
problem is tractable on simulators or small quantum devices  
- Run simulations using:  
  - Classical solvers (baseline)  
  - VQLS  
  - HHL  
- Compare results based on:  
  - Numerical accuracy (e.g., L2 norm between solution vectors)  
  - Simulation behavior (e.g., divergence, velocity field artifacts)  
  - Visual similarity across solver outputs  
  - Performance metrics (where measurable)  

### 3. Visualization and Reporting

- Record video frames during simulation to visualize fluid behavior
under different solvers  
- Highlight and annotate regions of the domain where quantum solvers
were applied (e.g., small grid sections)  
- Include side-by-side video outputs for classical, VQLS, and HHL-enhanced simulations  

## Potential Extensions

If core integration is successful, we may also explore:

- Incorporating Quantum Singular Value Transformation (QSVT) methods  
- Evaluating other quantum solvers (e.g., GLLS)  
- Automating matrix preprocessing for quantum compatibility  
- Contributing general-purpose wrappers or utilities to Classiq’s
open ecosystem  
- Investigating error mitigation strategies or hybrid iterative
refinement techniques  

## Deliverables

- `quantum_solver.py` Python module with support for VQLS and HHL  
- Reproducible test cases and benchmark results  
- Side-by-side visual and video outputs comparing solver results  
- Final written report covering:  
  - Integration methodology  
  - Comparative evaluation of quantum vs. classical solvers  
  - Observed limitations and recommendations for future work  


## Conclusion

This project explores a novel application of hybrid quantum-classical
computing in the field of fluid simulation. By comparing VQLS and HHL
to classical solvers in a controlled setting, we hope to gain insights
into the current capabilities and limitations of quantum approaches in
engineering simulation. Visual comparisons will help contextualize
solver performance beyond raw metrics, highlighting potential future
applications of quantum computing in CFD and related domains.

