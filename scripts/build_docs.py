# Inpsired by
# https://nbconvert.readthedocs.io/en/latest/execute_api.html#executing-notebooks
# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html#Quick-overview
# both accessed 2024-08-29

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

NOTEBOOK_DIR = Path("./docs/designers/tutorials/notebooks")


def build_notebooks(notebook_dir: Path, run_notebooks: bool = False) -> None:
    """Convert Jupyter notebooks to markdown files, optionally running them before
    conversion.
    """
    for nb_path in notebook_dir.iterdir():
        # Load notebook
        with open(nb_path) as file:
            nb = nbformat.read(file, as_version=4)

        if run_notebooks:
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

        # Post-process markdown text
        source = process_markdown(source)

        # Write markdown source
        md_path = nb_path.parent.parent / (nb_path.stem + ".md")
        with open(md_path, "w", encoding="utf-8") as file:
            file.write(source)


def process_markdown(markdown: str) -> str:
    """Process a markdown string to make some additions / changes.

    * Wraps output of code examples in a particular <div> tag to improve the visual formatting,
      making it appear bound to the preceding Python code example. This was worked out by
      inspecting code from the Material for MkDocs source code; see e.g.
      https://github.com/squidfunk/mkdocs-material/blob/23f18866c342968f5819495596fcb24f5dcc8b85/docs/reference/code-blocks.md?plain=1#L214-L220

    * Adds declaration to include a code copy button when entering Python code blocks
    """
    new_lines = []
    in_output_block = False
    in_python_block = False
    OUTPUT_PREFIX = " " * 4
    PYTHON_CODE_BLOCK_START = "```python\n"
    CODE_BLOCK_END = "```\n"
    for line in markdown.splitlines(keepends=True):
        # Determine whether entering/exiting a Python code block; add copy button if
        # entering.
        if line == PYTHON_CODE_BLOCK_START:
            in_python_block = True

            # Replace with code block declaration that includes a copy button
            line = "``` { .python .copy }\n"
        elif in_python_block and new_lines[-1] == CODE_BLOCK_END:
            in_python_block = False

        # Determine whether entering/exiting an output block for code and wrap with div
        # tag accordingly.
        if in_python_block:
            in_output_block = False
        elif line.startswith(OUTPUT_PREFIX) and not in_output_block:
            new_lines.append('<div class="result" markdown>\n')
            in_output_block = True
        elif not line.startswith(OUTPUT_PREFIX) and in_output_block:
            in_output_block = False
            new_lines.append("</div>\n")

        new_lines.append(line)

    return "".join(new_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build markdown docs from Jupyter notebooks."
    )
    parser.add_argument(
        "-n",
        "--no-run-notebooks",
        action="store_false",
        dest="run_notebooks",
        help=(
            "do not run each notebook before building a markdown version (default is to "
            "run the notebooks)"
        ),
    )

    args = parser.parse_args()
    build_notebooks(NOTEBOOK_DIR, args.run_notebooks)
