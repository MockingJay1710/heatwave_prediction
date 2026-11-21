"""Exécuter les notebooks depuis la racine, dans l'environnement `nix develop`."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    # Afficher aussi les figures dans le notebook exécuté sans interface graphique.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    for path in args.notebooks:
        path = path.resolve()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f"Exécution : {path.name}", flush=True)
        client = NotebookClient(
            notebook, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Terminé : {path.name}", flush=True)
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream" and output.name == "stdout":
                    print(output.text, end="")


if __name__ == "__main__":
    main()
