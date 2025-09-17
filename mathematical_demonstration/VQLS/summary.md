# VQLS Project Check-In: Status & Next Steps

## Vqe and VQLS

- Detailed notes covering VQLS and VQE **(notes/)**
    - Variational Quantum Eigensolver (VQE) finds the ground state of a
    **Hamiltonian**
    - VQE cost: expectation of energy
    - VQLS cost: $C = 1 - |\braket{b|\psi}|^2$

## Code

- Studied the Classiq VQLS notebook demonstrating VQLS for a fixed 3x3 example
- Built a Python program to run VQLS on arbitrary $A$ and $b$ (3x3) (**Figure
2**)
  - Ready for writing tests and running on real quantum hardware
- C++ implementations for Eulerian-based fluid simulators **Figures 3 and 4**
  - Understanding the pressure solve to generate real pressure solve systems
  - including mass conservative methods for Semi-Laganian advection

## Figures

- timothy_wind_tunnel : Windtunnel Prototype
- timothy_vqls_stats: Progam Running VQLS
- timothy_clear_divergence: Clear divergence Shortcut
- timothy_airfoil : C++ Kokkos Pressure Solve Airfoil (512x512)
- timothy_cylinder : C++ Kokkos Pressure Solve Cyinder (1024x1024)