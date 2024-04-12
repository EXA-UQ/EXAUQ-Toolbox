import json
from typing import Any

from exauq.sim_management.hardware import UnixServerScriptInterface
from exauq.sim_management.types import FilePath


class InteractiveUnixServerScriptInterfaceFactory:
    def __init__(self):
        self._hardware_parameters = None

    @property
    def hardware_parameters(self) -> dict[str, Any]:
        return self._hardware_parameters

    @property
    def hardware_type(self) -> str:
        return UnixServerScriptInterface.__name__

    def make_hardware_interactively(self) -> UnixServerScriptInterface:
        host = input("Host server address: ")
        user = input("Host username: ")
        script_path = input("Path to simulator script on host: ")
        program = input("Program to run simulator script with: ")
        params = {
            "host": host,
            "user": user,
            "script_path": script_path,
            "program": program,
        }
        hardware = UnixServerScriptInterface(**params)
        params["workspace_dir"] = hardware.workspace_dir
        self._hardware_parameters = params
        return hardware

    def serialise_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="w") as f:
            json.dump(self.hardware_parameters, f, indent=4)

    def load_hardware(self, params_file: FilePath) -> UnixServerScriptInterface:
        with open(params_file, mode="r") as f:
            params = json.load(f)

        return UnixServerScriptInterface(**params)
