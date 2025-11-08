import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def compute_divergence(u, dx=1.0):
    return np.array([ (u[i+1]-u[i])/dx for i in range(len(u)-1)])

import numpy as np
def make_divergence_free_1d(u, dx=1.0):
    """
    Make 1D face-centered velocity u divergence-free using Poisson solve.
    u: array of length N+1 (N cells)
    """
    N = len(u)-1
    div = u[1:] - u[:-1]   # divergence at cells

    if N <= 2:
        return u.copy()    # trivial

    # Build Poisson matrix for interior cells (p0=0, pN=0)
    A = np.zeros((N-2, N-2))
    rhs = div[1:-1] * dx

    for i in range(N-2):
        A[i,i] = -2
        if i>0: A[i,i-1] = 1
        if i<N-3: A[i,i+1] = 1

    # solve for interior pressures
    p = np.zeros(N)
    p[1:-1] = np.linalg.solve(A, rhs)

    # correct interior faces
    u_corr = u.copy()
    u_corr[1:-1] -= (p[1:] - p[:-1]) / dx

    return u_corr

def make_divergence_free_1d_iterative(u, dx=1.0, max_iter=20, tol=1e-6):
    """
    Iterative 1D MAC divergence-free projection using Jacobi iterations.
    
    u: face-centered velocity (length N+1)
    dx: cell size
    max_iter: max Jacobi iterations
    tol: stop if max divergence < tol
    """
    N = len(u) - 1  # number of cells
    div = u[1:] - u[:-1]  # initial divergence

    # pressure at cell centers
    p = np.zeros(N)
    p_new = np.zeros_like(p)

    for it in range(max_iter):
        # Jacobi iteration for interior pressures
        for i in range(1, N-1):
            p_new[i] = 0.5 * (p[i-1] + p[i+1] - dx * div[i])
        # boundaries remain zero
        p_new[0] = 0
        p_new[-1] = 0
        
        # check convergence
        if np.max(np.abs(p_new - p)) < tol:
            break
        
        p[:] = p_new

    # Correct face velocities
    u_corr = u.copy()
    u_corr[1:-1] -= (p[1:] - p[:-1]) / dx

    return u_corr

def plot_mac_fixed(u, dx=1.0, title="1D MAC Grid", show_arrows=True):
    """
    Plot a fixed MAC grid: velocity arrows + divergence bars.
    """
    N = len(u)-1
    x_faces = np.linspace(0, N, N+1)
    x_cells = (x_faces[:-1] + x_faces[1:])/2
    div = compute_divergence(u, dx)
    
    fig, ax = plt.subplots(2,1,figsize=(10,4), gridspec_kw={'height_ratios':[1,1]})
    
    # Top: cells + arrows
    ax_top = ax[0]
    for i, xc in enumerate(x_cells):
        color = plt.cm.RdBu((div[i]+2)/4)
        rect = Rectangle((x_faces[i], -0.2), 1, 0.4, color=color, alpha=0.8)
        ax_top.add_patch(rect)
        ax_top.text(xc, 0, f"{div[i]:+.2f}", ha="center", va="center", fontsize=8)
    if show_arrows:
        for xf, ui in zip(x_faces, u):
            ax_top.plot([xf, xf+0.3*np.sign(ui)], [0.25,0.25], color='r' if ui>=0 else 'b', lw=2)
            ax_top.text(xf,0.33,f"{ui:+.2f}", ha='center', va='bottom', fontsize=8)
    ax_top.set_xlim(-0.5,N+0.5)
    ax_top.set_ylim(-0.5,0.6)
    ax_top.set_yticks([])
    ax_top.set_title(title + " — Cells + Arrows")
    ax_top.grid(True, linestyle='--', alpha=0.3)
    
    # Bottom: line + bars
    ax_bottom = ax[1]
    ax_bottom.plot(x_faces, u, '-o', color='k', label='Velocity (u)')
    bars = ax_bottom.bar(x_cells, div, width=1, alpha=0.6, color=plt.cm.RdBu((div+2)/4))
    ax_bottom.axhline(0,color='k',lw=0.8)
    ax_bottom.set_ylim(-1.5,1.5)
    ax_bottom.set_xlim(-0.5,N+0.5)
    ax_bottom.set_xlabel("x (cell index →)")
    ax_bottom.set_title(title + " — Velocity Line + Divergence Bars")
    ax_bottom.grid(True, linestyle='--', alpha=0.3)
    ax_bottom.legend()
    
    plt.tight_layout()
    return fig, ax

