import json
import os
import matplotlib.pyplot as plt
import numpy as np



def save_stats_to_json(stats, filename="vqls_stats", folder="results"):
    """
    Save a dictionary of statistics to a JSON file.
    """
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
    filepath = os.path.join(folder, filename + ".json")
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {filepath}")

def plot_classical_vs_quantum(
    classical_probs,
    quantum_probs,
    name="Classical vs Quantum probabilities",
):
    """
    Plots classical and quantum probabilities side by side for comparison.
    """
    os.makedirs("data", exist_ok=True)

    N = len(classical_probs)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    x_indices = np.arange(N)

    # Classical probabilities (blue)
    ax.bar(
        x_indices - bar_width / 2,
        classical_probs,
        width=bar_width,
        color="#1f77b4",
        alpha=0.8,
        label="Classical",
    )

    # Quantum probabilities (orange)
    ax.bar(
        x_indices + bar_width / 2,
        quantum_probs,
        width=bar_width,
        color="#ff7f0e",
        alpha=0.8,
        label="Quantum",
    )

    # Add labels and title BEFORE saving
    ax.set_xlabel("Vector space basis")
    ax.set_ylabel("Probability")
    ax.set_title(name)
    ax.set_xticks(x_indices)
    ax.legend()
    plt.tight_layout()

    # Save AFTER setting everything up
    fig.savefig(f"data/{name}.png")
    print(f"✅ Saved figure as data/{name}.png")

    plt.show()



def visualize_vqls_results(folder="data"):
    """
    Reads all JSON result files from a folder and plots each result.
    """
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found.")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not json_files:
        print(f"⚠️ No JSON files found in '{folder}'.")
        return

    print(f"Found {len(json_files)} result file(s) in '{folder}':\n")

    for filename in json_files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"📊 {filename}")
        print(f"  Iterations: {data.get('iterations', 'N/A')}")
        print(f"  Overlap: {data.get('overlap', 'N/A'):.6f}")
        print(f"  MSE: {data.get('mse', 'N/A'):.6e}")
        print(f"  Cosine similarity: {data.get('cosine_similarity', 'N/A'):.6f}")
        print()

        # Extract probability vectors
        classical_probs = np.array(data.get("classical_probs", []))
        quantum_probs = np.array(data.get("quantum_probs", []))

        # Plot and save
        name = os.path.splitext(filename)[0]
        plot_classical_vs_quantum(classical_probs, quantum_probs, name=name)

def laplacian_2d(Nx, Ny):
    N = Nx * Ny
    A = np.zeros((N, N))
    
    # Main diagonal = number of neighbors per cell
    main_diag = np.full(N, 4)
    
    # Adjust for edges/corners
    for i in range(Nx):
        for j in range(Ny):
            idx = i*Ny + j
            count = 0
            if i > 0: count += 1
            if i < Nx-1: count += 1
            if j > 0: count += 1
            if j < Ny-1: count += 1
            main_diag[idx] = count
    A[np.arange(N), np.arange(N)] = main_diag

    # Off-diagonals
    # Horizontal neighbors
    for i in range(Nx):
        for j in range(Ny-1):
            idx = i*Ny + j
            A[idx, idx+1] = -1
            A[idx+1, idx] = -1
    # Vertical neighbors
    for i in range(Nx-1):
        for j in range(Ny):
            idx = i*Ny + j
            A[idx, idx+Ny] = -1
            A[idx+Ny, idx] = -1

    return A

