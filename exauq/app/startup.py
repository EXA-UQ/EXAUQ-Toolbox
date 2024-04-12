import inspect
import json
from typing import Any, Optional

from exauq.sim_management.hardware import HardwareInterface, UnixServerScriptInterface
from exauq.sim_management.types import FilePath


class HardwareInterfaceFactory:
    def __init__(self, hardware_cls: type):
        if not issubclass(hardware_cls, HardwareInterface):
            raise ValueError(
                f"{hardware_cls} does not inherit from {HardwareInterface.__name__}"
            )
        else:
            self._hardware_cls = hardware_cls

        self._hardware_parameters = None

    @property
    def hardware_parameters(self) -> Optional[dict[str, Any]]:
        return self._hardware_parameters

    @property
    def hardware_type(self) -> str:
        return self._hardware_cls.__name__

    def make_hardware_interactively(self) -> HardwareInterface:
        cls_init_params = [
            param
            for param in inspect.signature(
                self._hardware_cls.__init__
            ).parameters.values()
            if not param.name == "self"
        ]
        params = dict()
        for param in cls_init_params:
            if param.default is inspect.Parameter.empty:
                default_value = None
                value_str = input(f"{self.hardware_type} {param.name}: ").strip()
            else:
                default_value = param.default
                value_str = input(
                    f"{self.hardware_type} {param.name} (default: {default_value}): "
                ).strip()

            value = value_str if value_str else default_value
            params[param.name] = value

        self._hardware_parameters = params
        return self._hardware_cls(**params)

    def serialise_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="w") as f:
            json.dump(self.hardware_parameters, f, indent=4)

    def load_hardware(self, params_file: FilePath) -> HardwareInterface:
        with open(params_file, mode="r") as f:
            params = json.load(f)

        self._hardware_parameters = params
        return self._hardware_cls(**params)


class UnixServerScriptInterfaceFactory(HardwareInterfaceFactory):
    def __init__(self):
        super().__init__(UnixServerScriptInterface)
        self._hardware_parameters = None

    @property
    def hardware_parameters(self) -> Optional[dict[str, Any]]:
        return self._hardware_parameters

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
