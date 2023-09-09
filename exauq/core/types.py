from numbers import Real
from os import PathLike
from typing import Optional, Union

from exauq.core.modelling import Input

FilePath = Union[str, bytes, PathLike]
"""A type to represent filepaths."""

Simulation = tuple[Input, Optional[Real]]
"""A type to represent a simulator input, possibly with corresponding simulator output."""
