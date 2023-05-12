"""Classes and functions to support data validation throughout the package API.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any
import numpy as np


class FiniteRealIterableValidator(object):

    def __init__(self, x: Iterable):
        self._x = x

    def check_no_none_entries(self, exception: Exception):
        if not all(element is not None for element in self._x):
            raise exception

    def check_entries_real(self, exception: Exception):
        if not all(isinstance(element, Real) for element in self._x):
            raise exception

    def check_entries_finite(self, exception: Exception):
        if not all(np.isfinite(element) for element in self._x):
            raise exception


class FiniteRealValidator(object):
    def __init__(self, x: Any):
        self._x = x
    
    def check_not_none(self, exception: Exception):
        if self._x is None:
            raise exception

    def check_real(self, exception: Exception):
        if not isinstance(self._x, Real):
            raise exception

    def check_finite(self, exception: Exception):
        if not np.isfinite(self._x):
            raise exception
