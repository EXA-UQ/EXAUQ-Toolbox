import json
import pathlib
import sys
from typing import Any

from exauq.sim_management.hardware import UnixServerScriptInterface


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


def make_unix_server_script_interface(
    ssh_config: dict[str, Any], remote_script_config: dict[str, Any]
):
    return UnixServerScriptInterface(
        user=ssh_config["user"],
        host=ssh_config["host"],
        program=remote_script_config["program"],
        script_path=remote_script_config["script_path"],
        workspace_dir=remote_script_config["workspace_dir"],
        key_filename=ssh_config["key_filename"],
        ssh_config_path=ssh_config["ssh_config_path"],
        use_ssh_agent=ssh_config["use_ssh_agent"],
        max_attempts=1,
    )
