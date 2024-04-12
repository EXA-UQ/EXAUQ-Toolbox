import inspect
import json
from collections import OrderedDict
from typing import Any

from exauq.sim_management.hardware import HardwareInterface, UnixServerScriptInterface
from exauq.sim_management.types import FilePath


class HardwareInterfaceFactory:
    _MISSING = inspect.Parameter.empty

    def __init__(self, hardware_cls: type):
        if not issubclass(hardware_cls, HardwareInterface):
            raise ValueError(
                f"{hardware_cls} does not inherit from {HardwareInterface.__name__}"
            )
        else:
            self._hardware_cls = hardware_cls

        self._hardware_parameters = self._get_init_params(hardware_cls)

    @staticmethod
    def _get_init_params(cls_: type) -> OrderedDict[str, Any]:
        return OrderedDict(
            (param.name, param.default)
            for param in inspect.signature(cls_.__init__).parameters.values()
            if not param.name == "self"
        )

    @property
    def hardware_parameters(self) -> OrderedDict[str, Any]:
        return self._hardware_parameters

    @property
    def hardware_type(self) -> str:
        return self._hardware_cls.__name__

    def set_param_from_str(self, param: str, value: str) -> None:
        self._hardware_parameters[param] = value

    def serialise_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="w") as f:
            json.dump(self.hardware_parameters, f, indent=4)

        return None

    def load_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="r") as f:
            params = json.load(f)

        if set(params) == set(self.hardware_parameters):
            self._hardware_parameters = OrderedDict(params)
            return None
        else:
            raise AssertionError(
                f"The deserialised parameter names do not agree with those required to initialise {self.hardware_type}."
            )

    def build_hardware(self) -> HardwareInterface:
        missings_params = {
            param
            for param, value in self._hardware_parameters.items()
            if value is self._MISSING
        }
        if missings_params:
            raise ValueError(
                f"Cannot initialise instance of {self.hardware_type} while missing params {missings_params}."
            )
        else:
            return self._hardware_cls(**self.hardware_parameters)

    @property
    def interactive_prompts(self) -> OrderedDict[str, str]:
        prompts = OrderedDict()
        for param, value in self.hardware_parameters.items():
            if value is self._MISSING:
                prompts[param] = f"{self.hardware_type} {param.name}"
            else:
                prompts[param] = f"{self.hardware_type} {param.name} (default: {value})"

        return prompts


class UnixServerScriptInterfaceFactory(HardwareInterfaceFactory):
    def __init__(self):
        super().__init__(UnixServerScriptInterface)

    def build_hardware(self) -> UnixServerScriptInterface:
        hardware = super().build_hardware()
        self._hardware_parameters["workspace_dir"] = hardware.workspace_dir
        return hardware

    @property
    def interactive_prompts(self):
        return (
            ("host", "Host server address"),
            ("user", "Host username"),
            ("script_path", "Path to simulator script on host"),
            ("program", "Program to run simulator script with"),
        )
