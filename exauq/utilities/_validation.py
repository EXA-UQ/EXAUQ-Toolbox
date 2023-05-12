"""Classes and functions to support data validation throughout the package API.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any
import numpy as np


class FiniteRealValidator(object):

    def __init__(self, x: Any):
        self._iter_x = x if isinstance(x, Iterable) else [x]

    def check_no_none_entries(self, exception: Exception):
        if not all(element is not None for element in self._iter_x):
            raise exception


    def check_entries_real(self, exception: Exception):
        if not all(isinstance(element, Real) for element in self._iter_x):
            raise exception


    def check_entries_finite(self, exception: Exception):
        if not all(np.isfinite(element) for element in self._iter_x):
            raise exception
