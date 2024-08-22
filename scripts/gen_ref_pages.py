# This file is adapted from the example script 'gen_ref_pages.py' copied from
# https://github.com/mkdocstrings/mkdocstrings/blob/0.25.2/docs/recipes.md
# (accessed 2024-08-09). It is used under the following licence.
#
#
# ISC License
#
# Copyright (c) 2019, Timothée Mazzucotelli
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

# Placeholder text for modules without documentation
PLACEHOLDER_DOCSTRING = "Documentation coming soon for this module."


def get_module_parts(py_module: Path, package_root: Path) -> tuple[str, str, str]:
    """Get the component parts of a module relative to a package.

    Examples
    --------
    >>> py_module = Path("exauq/core/modelling.py")
    >>> package_root = Path("exauq")
    >>> get_module_parts(py_module, package_root)
    ('exauq', 'core', 'modelling')
    """

    module_path = Path("exauq") / py_module.relative_to(package_root).with_suffix("")
    return tuple(module_path.parts)


def get_module_paths_for_docs(package_root: Path) -> list[Path]:
    """Get paths to modules that should appear in the docs."""

    utilities_path = Path(package_root, "utilities")
    app_path = Path(package_root, "app")

    # Return modules not in the sub-packages exauq.app and exauq.utilities
    return sorted(
        module_path
        for module_path in package_root.rglob("*.py")
        if module_path.parent not in {utilities_path, app_path}
    )


def check_if_module_is_empty(module_path: Path) -> bool:
    """
    Check if the module is empty (contains only whitespace or no content).
    Returns True if the file is empty, False otherwise.
    """
    try:
        with module_path.open("r", encoding="utf-8") as file:
            content = file.read()

            # If the file is empty or contains only whitespace, it's considered empty
            return not content.strip()

    except Exception as e:
        print(f"Error loading module {module_path}: {e}")
        return True  # Treat as empty if there's an error reading the file


nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "exauq"

for path in get_module_paths_for_docs(src):
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = get_module_parts(path, src)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        # The following two lines, which appear in the original, don't seem to be needed
        # if the mkdocs-section-index plugin is used.
        #
        # doc_path = doc_path.with_name("index.md")
        # full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)

        # Check if the module is empty
        if check_if_module_is_empty(path):
            print(PLACEHOLDER_DOCSTRING, file=fd)
        else:
            # Generate API documentation for modules with code
            print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
