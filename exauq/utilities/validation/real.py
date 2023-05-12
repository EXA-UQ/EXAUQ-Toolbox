"""Functions for validation of data defining real numbers.
"""

from collections.abc import Iterable
from numbers import Real
from typing import (
    Any,
    Callable
    )
import numpy as np


def check_entries_not_none(x: Iterable, exception: Exception) -> None:
    """Raise the given exception if one of the elements of an iterable is
    ``None``.
    """
    check_for_each(check_not_none, x, exception)


def check_for_each(
        check_function: Callable, x: Iterable, exception: Exception
        ) -> None:
    """Apply a checking function to each element of an iterable, raising the
    given exception if the check fails for some element."""
    try:
        for element in x:
            check_function(element, Exception())
    
    except Exception:
        raise exception


def check_not_none(x: Any, exception: Exception) -> None:
    """Raise a given exception if an object is ``None``."""
    if x is None:
        raise exception


def check_entries_real(x: Iterable, exception: Exception) -> None:
    """Raise the given exception if one of the elements of an iterable is not
    a real number."""
    check_for_each(check_real, x, exception)


def check_real(x: Any, exception: Exception) -> None:
    """Raise the given exception if an object is not a real number."""
    if not isinstance(x, Real):
        raise exception


def check_entries_finite(x: Iterable, exception: Exception) -> None:
    """Raise the given exception if one of the elements of an iterable is not
    a finite number."""
    check_for_each(check_finite, x, exception)


def check_finite(x: Any, exception: Exception) -> None:
    """Raise the given exception if an object is not a finite number."""
    if not np.isfinite(x):
        raise exception
