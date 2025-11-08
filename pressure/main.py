import numpy as np
from interactive_mac import interactive_mac_grid
from plot_fixed import show_before_after

N = 8
u_init = np.ones(N+1)*0.5
u_init[3] = -0.2

# --- Interactive visualization ---
interactive_mac_grid(u_init)

# --- Show fixed plots before and after correction ---

np.random.seed(0)
x = np.linspace(0, 8, 9)
u = [
np.array([1.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -1.5]),
np.array([-0.5, 1.0, 0.0, -0.5, 0.5, 0.0, -1.0, 0.5, -1.5]),
0.5 * np.sin(np.pi * x / 8),
np.array([-1.0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0.0]),
np.random.uniform(-1, 1, 9)]

u[:][0] = -1  # fix boundaries
u[:][-1] = 1
# for v in u:
show_before_after(u[0])

