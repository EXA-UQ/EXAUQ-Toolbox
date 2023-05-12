"""Functions for validation of data defining real numbers.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any
import numpy as np


def check_entries_not_none(x: Iterable, exception: Exception):
    check_for_each(check_not_none, x, exception)


def check_for_each(check_function, x, exception):
    try:
        for element in x:
            check_function(element, Exception())
    
    except Exception:
        raise exception


def check_not_none(x: Any, exception: Exception):
    if x is None:
        raise exception


def check_entries_real(x: Iterable, exception: Exception):
    check_for_each(check_real, x, exception)


def check_real(x: Any, exception: Exception):
    if not isinstance(x, Real):
        raise exception


def check_entries_finite(x: Iterable, exception: Exception):
    check_for_each(check_finite, x, exception)


def check_finite(x: Any, exception: Exception):
    if not np.isfinite(x):
        raise exception
