"""Functions etc. to support testing"""


def exact(string: str):
    """Turn a string into a regular expressions that defines an exact match on the
    string.
    """
    escaped = string
    for char in ["(", ")"]:
        escaped = escaped.replace(char, _escape(char))

    return "^" + escaped + "$"


def _escape(char):
    return "\\" + char
