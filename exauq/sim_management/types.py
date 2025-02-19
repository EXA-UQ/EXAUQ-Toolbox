"""
Provides type definitions to enhance code clarity and maintainability within the
EXAUQ toolbox. This module defines reusable type aliases that represent common
data structures and formats used across the codebase.


Type Definitions
------------------------------------------------------------------------------------
[`FilePath`][exauq.sim_management.types.FilePath]
Represents a file path, defined as a union of `str` and `PathLike` to support both
string-based and OS-native path objects.


"""

from os import PathLike
from typing import Union

FilePath = Union[str, PathLike]
"""A type to represent filepaths."""
