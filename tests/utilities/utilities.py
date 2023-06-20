"""Functions etc. to support testing"""


def exact(string: str):
    """Turn a string into a regular expressions that defines an exact match on the
    string.
    """

    return "^" + string + "$"
