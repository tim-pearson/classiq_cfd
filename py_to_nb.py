import nbformat
import sys

def split_py_into_cells(py_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell = []

    for line in lines:
        if line.strip().startswith('# %%'):
            if current_cell:
                cells.append(current_cell)
                current_cell = []
        else:
            current_cell.append(line)

    if current_cell:
        cells.append(current_cell)

    return [''.join(cell) for cell in cells]

def update_ipynb(ipynb_path, py_path):
    # Load notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Extract new code blocks from Python file
    new_code_blocks = split_py_into_cells(py_path)

    # Count existing code cells
    existing_code_cells = [c for c in nb.cells if c.cell_type == 'code']
    num_existing = len(existing_code_cells)

    # Replace existing code cells
    code_cell_idx = 0
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if code_cell_idx < len(new_code_blocks):
                cell.source = new_code_blocks[code_cell_idx]
                code_cell_idx += 1

    # Append extra code blocks as new cells
    while code_cell_idx < len(new_code_blocks):
        numbered_source = f"# Block {code_cell_idx + 1}\n" + new_code_blocks[code_cell_idx]
        new_cell = nbformat.v4.new_code_cell(source=numbered_source)
        nb.cells.append(new_cell)
        code_cell_idx += 1

    # Save updated notebook
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"✅ Updated {ipynb_path} with code from {py_path} (appended {code_cell_idx - num_existing} new cells)")

# --- Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_ipynb_from_py.py notebook.ipynb modified.py")
        sys.exit(1)

    ipynb_path = sys.argv[1]
    py_path = sys.argv[2]
    update_ipynb(ipynb_path, py_path)

