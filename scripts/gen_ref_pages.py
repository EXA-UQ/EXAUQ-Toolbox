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


def get_module_parts(py_module: Path, package_root: Path):
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


nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "exauq"

for path in sorted(src.rglob("*.py")):
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
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
