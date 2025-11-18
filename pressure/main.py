import numpy as np
from interactive_mac import interactive_mac_grid
from plot_fixed import show_before_after
from mac_utils import make_divergence_free_2d

# %%
N = 4
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
show_before_after(np.array([0.2, -0.3, 0.0, 0.5, 0.07]))

# %% [markdown]
r"""
# 2D mac grid
"""
# %%
Nx, Ny = 4, 4
dx = dy = 1.0
u = np.random.randn(Nx+1, Ny) * 0.1
v = np.random.randn(Nx, Ny+1) * 0.1

div_before = (u[1:, :] - u[:-1, :])/dx + (v[:,1:] - v[:,:-1])/dy
print("Initial max divergence:", np.abs(div_before).max())

u_corr, v_corr = make_divergence_free_2d(u, v, dx, dy, max_iter=1000,)

div_after = (u_corr[1:, :] - u_corr[:-1, :])/dx + (v_corr[:,1:] - v_corr[:,:-1])/dy
print("Final max divergence:", np.abs(div_after).max())

