import contextlib
from functools import wraps
from io import StringIO


def suppress_print(func):
    """This decorator is implemented to redirect the printing of unnecessary print statements from certain libraries
    (e.g the mogp-emulator) away from stdio to focus on only relevent information."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(StringIO()):
            return func(*args, **kwargs)

    return wrapper
