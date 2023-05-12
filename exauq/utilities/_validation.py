"""Classes and functions to support data validation throughout the package API.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any
import numpy as np


def check_no_none_entries(x: Any, exception: Exception):
    _iter = x if isinstance(x, Iterable) else [x]
    if not all(element is not None for element in _iter):
        raise exception


def check_entries_real(x: Any, exception: Exception):
    _iter = x if isinstance(x, Iterable) else [x]
    if not all(isinstance(element, Real) for element in _iter):
        raise exception


def check_all_entries_finite(x: Any, exception: Exception):
    _iter = x if isinstance(x, Iterable) else [x]
    if not all(np.isfinite(element) for element in _iter):
        raise exception
