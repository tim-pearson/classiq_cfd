# VQLS Project Check-In: Status & Next Steps

## VQE and VQLS

- Detailed notes covering VQLS ,VQE and the Optimizer **(pdf)**
    - Variational Quantum Eigensolver (VQE) finds the ground state of a
    **Hamiltonian**
    - VQE cost: expectation of energy
    - VQLS cost: $C = 1 - |\braket{b|\psi}|^2$

## Code

- Studied the Classiq VQLS notebook demonstrating VQLS for a fixed 3x3 example
- Built a Python program to run VQLS on arbitrary $A$ and $b$ (3x3) 
  - Ready for writing tests and running on real quantum hardware
- C++ implementations for Eulerian-based fluid simulators *
  - Understanding the pressure solve to generate real pressure solve systems
  - including mass conservative methods for Semi-Laganian advection

## Figures and Files

- timothy_wind_tunnel : Windtunnel Prototype
- timothy_vqls_stats: Progam Running VQLS
- timothy_clear_divergence: Clear divergence Shortcut (512x512)
- timothy_airfoil : C++ Kokkos Pressure Solve Airfoil (512x512)
- timothy_cylinder : C++ Kokkos Pressure Solve Cyinder (1024x1024)
- pdf/ : Notes on VQLS and VQE