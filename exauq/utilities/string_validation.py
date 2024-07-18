import re
from typing import Optional

ALLOWED_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
FORBIDDEN_PATTERNS = [
    "shutdown",
    "delete",
    "drop",
    "remove",
    "truncate",
    "update",
    "alter",
    "insert",
    "select",
    "grant",
]
MAX_LENGTH = 50


def validate_interface_name(interface_name: Optional[str]) -> Optional[str]:
    """
    Validates the interface name based on certain criteria.

    Parameters
    ----------
    interface_name : Optional[str]
        The interface name to be validated. Can be None.

    Returns
    -------
    Optional[str]
        The validated interface tag, or None if the input was None.

    Raises
    ------
    ValueError
        If interface_tag exceeds the maximum length or contains forbidden patterns or
        invalid characters.
    """
    if interface_name is None:
        return interface_name

    if not isinstance(interface_name, str):
        raise TypeError(
            f"Expected 'interface_tag' to be of type {str} or None but received "
            f"{type(interface_name)} instead."
        )

    if len(interface_name) > MAX_LENGTH:
        raise ValueError(
            f"Interface tag exceeds maximum length of {MAX_LENGTH} characters."
        )

    lowered_tag = interface_name.lower()
    if any(forbidden_word in lowered_tag for forbidden_word in FORBIDDEN_PATTERNS):
        raise ValueError(
            f"Interface tag contains forbidden words or patterns: {FORBIDDEN_PATTERNS}"
        )

    if not ALLOWED_PATTERN.match(interface_name):
        raise ValueError(
            "Interface tag contains invalid characters. Allowed characters are "
            "alphanumeric, hyphens, and underscores."
        )

    return interface_name
