# Inpsired by
# https://nbconvert.readthedocs.io/en/latest/execute_api.html#executing-notebooks
# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html#Quick-overview
# both accessed 2024-08-29

import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

NOTEBOOK_DIR = Path("./docs/designers/tutorials/notebooks")
for nb_path in NOTEBOOK_DIR.iterdir():
    # Load notebook
    with open(nb_path) as file:
        nb = nbformat.read(file, as_version=4)

    ep = ExecutePreprocessor(timeout=600)
    try:
        # Execute notebook (mutates the nb in-place)
        ep.preprocess(nb)
    except CellExecutionError as e:
        msg = f"Error executing the notebook {nb_path.name}.\n"
        print(msg, file=sys.stderr)
        raise e

    # Export notebook to markdown
    exporter = MarkdownExporter()
    source, resources = exporter.from_notebook_node(nb)

    # Write markdown source
    md_path = nb_path.parent.parent / (nb_path.stem + ".md")
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(source)
