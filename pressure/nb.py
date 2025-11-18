import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

# --- Parameters ---
N = 8
dx = 1.0
x_faces = np.linspace(0, N, N + 1)
x_cells = (x_faces[:-1] + x_faces[1:]) / 2

# Initial velocity
# --- Compute divergence ---
def compute_divergence(u):
    return np.array([(u[i+1] - u[i])/dx for i in range(len(u)-1)])
u_init = np.ones(N + 1) * 0.5

u = u_init.copy()
u[3] = - 0.2


# %%

# --- Create figure and axes ---
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12,5), gridspec_kw={'height_ratios':[1,1]})
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.35, top=0.95, hspace=0.4)

# --- Top plot: cell rectangles + arrows ---
rect_patches = []
div_texts_top = []
arrow_lines = []
arrow_texts_top = []

div_top = compute_divergence(u_init)

# divergence rectangles + labels
for i, xc in enumerate(x_cells):
    color = plt.cm.RdBu((div_top[i] - div_top.min()) / (div_top.max() - div_top.min() + 1e-8))

    rect = Rectangle((x_faces[i], -0.2), 1, 0.4, color=color, alpha=0.8)
    ax_top.add_patch(rect)
    rect_patches.append(rect)
    txt = ax_top.text(xc, 0, f"{div_top[i]:+.2f}", ha="center", va="center", fontsize=8, color="k")
    div_texts_top.append(txt)

# velocity arrows + labels using simple line + triangle head
for xf, ui in zip(x_faces, u_init):
    # Arrow body
    line, = ax_top.plot([xf, xf + 0.3 * np.sign(ui)], [0.25, 0.25],
                        color="r" if ui >=0 else "b", lw=2)
    arrow_lines.append(line)
    txt = ax_top.text(xf, 0.33, f"{ui:+.2f}", ha="center", va="bottom", fontsize=8)
    arrow_texts_top.append(txt)

ax_top.set_xlim(-0.5, N+0.5)
ax_top.set_ylim(-0.5, 0.6)
ax_top.set_yticks([])
ax_top.set_title("1D MAC Grid Cells + Velocity Arrows")
ax_top.grid(True, linestyle='--', alpha=0.3)

# --- Bottom plot: line + bars ---
line_container_bottom, = ax_bottom.plot(x_faces, u_init, '-o', color='k', label='Velocity (u)')
div_bottom = compute_divergence(u_init)
bar_container_bottom = ax_bottom.bar(x_cells, div_bottom, width=1, alpha=0.6,
                                     color=plt.cm.RdBu((div_bottom - div_bottom.min()) / (div_bottom.max() - div_bottom.min() + 1e-8)))
ax_bottom.axhline(0, color='k', lw=0.8)
ax_bottom.set_ylim(-1.5, 1.5)
ax_bottom.set_xlim(-0.5, N+0.5)
ax_bottom.set_xlabel("x (cell index →)")
ax_bottom.set_title("Velocity Line + Divergence Bars")
ax_bottom.grid(True, linestyle='--', alpha=0.3)
ax_bottom.legend()

# --- Create sliders ---
sliders = []
for i in range(N + 1):
    ax_slider = plt.axes([0.1, 0.15 - i*0.02, 0.8, 0.015])
    slider = Slider(ax_slider, f'u{i}', -1.0, 1.0, valinit=u_init[i])
    sliders.append(slider)

# --- Update function ---
def update(val):
    u = np.array([s.val for s in sliders])
    
    # Top plot: update divergence rectangles and labels
    div_top = compute_divergence(u)
    for rect, h, txt in zip(rect_patches, div_top, div_texts_top):
        rect.set_facecolor(plt.cm.RdBu((h + 2) / 4))
        txt.set_text(f"{h:+.2f}")
    
    # Top plot: update arrows and labels
    for line, ui, txt in zip(arrow_lines, u, arrow_texts_top):
        line.set_xdata([line.get_xdata()[0], line.get_xdata()[0] + ui])
        # line.set_xdata([line.get_xdata()[0], line.get_xdata()[0] + 0.3*np.sign(ui)])
        line.set_color("b" if ui >=0 else "r")
        txt.set_text(f"{ui:+.2f}")
    
    # Bottom plot: update line and bars
    line_container_bottom.set_ydata(u)
    div_bottom = compute_divergence(u)
    for rect, h in zip(bar_container_bottom, div_bottom):
        rect.set_height(h)
        rect.set_color(plt.cm.RdBu((h + 2) / 4))
    
    fig.canvas.draw_idle()

# Connect sliders
for s in sliders:
    s.on_changed(update)

plt.show()

# %%
import numpy as np

def make_divergence_free_1d(u, dx=1.0):
    """
    Make face-centered 1D velocity u divergence-free.
    u: array of length N+1 (N cells)
    dx: cell size
    """
    N = len(u) - 1          # number of cells
    div = u[1:] - u[:-1]    # divergence at each cell

    if N <= 2:
        return u.copy()     # trivial for 1 or 2 cells

    # Solve Poisson for interior pressures (p[0]=p[N-1]=0)
    A = np.zeros((N-2, N-2))
    rhs = div[1:-1] * dx  # interior cells only

    for i in range(N-2):
        A[i,i] = -2
        if i>0: A[i,i-1] = 1
        if i<N-3: A[i,i+1] = 1

    p_interior = np.linalg.solve(A, rhs)

    # full pressure array
    p = np.zeros(N)
    p[1:-1] = p_interior

    # correct interior face velocities
    u_corr = u.copy()
    u_corr[1:N] -= (p[1:] - p[:-1]) / dx

    return u_corr
N = 8
u = np.ones(N+1) * 0.5
u[3] = -0.2

new_u = make_divergence_free_1d(u)
print("Original u:", u)
print("Corrected u:", new_u)
print("Divergence after correction:", new_u[1:] - new_u[:-1])

