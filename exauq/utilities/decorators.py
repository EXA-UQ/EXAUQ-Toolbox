from io import StringIO
import contextlib
import logging


def suppress_print(func):
    """This decorator is implemented to redirect the printing of unnecessary print statements from certain libraries
    (e.g the mogp-emulator) away from stdio to focus on only relevent information."""

    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(StringIO()):
            return func(*args, **kwargs)

    return wrapper


logging.basicConfig(filename="application_output.log", level=logging.INFO)


def redirect_print_to_log(func):
    """This decorator may help form some of the logging system to be implemented later. Currently it is useful
    for testing which methods should have their print suppressed."
    """

    def wrapper(*args, **kwargs):
        log_stream = StringIO()
        with contextlib.redirect_stdout(log_stream):
            result = func(*args, **kwargs)
        logging.info(log_stream.getvalue())
        return result

    return wrapper
