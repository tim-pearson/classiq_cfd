from mac_utils import make_divergence_free_1d_iterative, plot_mac_fixed
import numpy as np

def show_before_after(u):
    """
    Plot the MAC grid for given u before and after divergence-free correction.
    """
    import matplotlib.pyplot as plt
    from mac_utils import make_divergence_free_1d
    print(np.round(u,3))
    u_corr = make_divergence_free_1d_iterative(u, max_iter=25)
    print(np.round(u_corr, 3))
    print()
    
    plot_mac_fixed(u, title="Before Correction")
    # After
    plot_mac_fixed(u_corr, title="After Correction")
    plt.show()

