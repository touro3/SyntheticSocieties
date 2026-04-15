import json
from pathlib import Path

path = Path("notebooks/bgf_demo.ipynb")
nb = json.loads(path.read_text())

# O enumerate gera um índice (i) para cada célula
for i, cell in enumerate(nb["cells"]):
    # Cria IDs como "cell-1", "cell-2", etc.
    cell["id"] = f"cell-{i + 1}"

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
