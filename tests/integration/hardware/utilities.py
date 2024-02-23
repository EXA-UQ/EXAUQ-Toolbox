import json
import pathlib
import sys
from typing import Any

from exauq.sim_management.hardware import RemoteServerScript


def get_command_line_args() -> str:
    try:
        return {
            "ssh_config_path": sys.argv[1],
            "remote_script_config_path": sys.argv[2],
        }
    except IndexError:
        print(
            f"{sys.argv[0]} error: Incorrect number of args supplied.",
            file=sys.stderr,
        )
        sys.exit(1)


def read_json_config(path: str) -> dict[str, Any]:
    with open(pathlib.Path(path), mode="r") as ssh_config_file:
        return json.load(ssh_config_file)


def make_remote_server_script(
    ssh_config: dict[str, Any], remote_script_config: dict[str, Any]
):
    return RemoteServerScript(
        user=ssh_config["user"],
        host=ssh_config["host"],
        program=remote_script_config["program"],
        script_path=remote_script_config["script_path"],
        config_path=remote_script_config["config_path"],
        stdout_path=remote_script_config["stdout_path"],
        key_filename=ssh_config["key_filename"],
        ssh_config_path=ssh_config["ssh_config_path"],
        use_ssh_agent=ssh_config["use_ssh_agent"],
        max_attempts=1,
    )
