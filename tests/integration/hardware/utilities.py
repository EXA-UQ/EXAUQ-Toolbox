import json
import pathlib
import sys
from typing import Any


def get_ssh_config_path() -> str:
    try:
        return sys.argv[1]
    except IndexError:
        print(
            f"{sys.argv[0]} error: No path to a ssh config file supplied.",
            file=sys.stderr,
        )
        sys.exit(1)


def read_ssh_config(path: str) -> dict[str, Any]:
    with open(pathlib.Path(path), mode="r") as ssh_config_file:
        return json.load(ssh_config_file)
