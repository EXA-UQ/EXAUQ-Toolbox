import inspect
import json
import pathlib
from collections import OrderedDict
from typing import Any, Callable, Optional

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

        self.hardware_parameters = self._get_init_params(hardware_cls)

    @staticmethod
    def _get_init_params(cls_: type) -> OrderedDict[str, Any]:
        return OrderedDict(
            (param.name, param.default)
            for param in inspect.signature(cls_.__init__).parameters.values()
            if not param.name == "self"
        )

    @property
    def hardware_type(self) -> str:
        return self._hardware_cls.__name__

    def set_param_from_str(self, param: str, value: str) -> None:
        self.hardware_parameters[param] = value

    def serialise_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="w") as f:
            json.dump(self.hardware_parameters, f, indent=4)

        return None

    def load_hardware_parameters(self, params_file: FilePath) -> None:
        with open(params_file, mode="r") as f:
            params = json.load(f)

        if set(params) == set(self.hardware_parameters):
            self.hardware_parameters = OrderedDict(params)
            return None
        else:
            raise AssertionError(
                f"The deserialised parameter names do not agree with those required to initialise {self.hardware_type}."
            )

    def build_hardware(self) -> HardwareInterface:
        missings_params = {
            param
            for param, value in self.hardware_parameters.items()
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
        self._parsers = {
            "host": make_str_parser(),
            "user": make_str_parser(),
            "script_path": make_posix_path_parser(),
            "program": make_str_parser(),
            "use_ssh_agent": make_bool_parser(
                required=False,
                default=self.hardware_parameters["use_ssh_agent"],
            ),
        }

    def set_param_from_str(self, param: str, value: str) -> None:
        self.hardware_parameters[param] = self._parsers[param](value)

    def build_hardware(self) -> UnixServerScriptInterface:
        hardware = super().build_hardware()
        self.hardware_parameters["workspace_dir"] = hardware.workspace_dir
        return hardware

    @property
    def interactive_prompts(self) -> OrderedDict[str, str]:
        use_ssh_agent_default = (
            "yes" if self.hardware_parameters["use_ssh_agent"] else "no"
        )
        return OrderedDict(
            (
                ("host", "Host server address"),
                ("user", "Host username"),
                ("script_path", "Path to simulator script on host"),
                ("program", "Program to run simulator script with"),
                (
                    "use_ssh_agent",
                    f"Use SSH agent? (Default '{use_ssh_agent_default}')",
                ),
            )
        )


def make_str_parser(
    required: bool = True, default: Optional[str] = None
) -> Callable[[str], Optional[str]]:

    def parse(x: str) -> Optional[str]:
        x = x.strip()
        if required and x == "":
            raise ValueError("A nonempty string must be supplied.")
        elif x == "":
            return default
        else:
            return x

    return parse


def make_posix_path_parser(
    required: bool = True, default: Optional[FilePath] = None
) -> Callable[[str], Optional[str]]:

    def parse(x: str) -> Optional[str]:
        x = make_str_parser(required=required, default="")(x)
        if x == "":
            return str(pathlib.PurePosixPath(default))
        else:
            try:
                return str(pathlib.PurePosixPath(x))
            except ValueError:
                raise ValueError(f"Could not parse '{x}' as a Posix path.")

    return parse


def make_bool_parser(
    required: bool = True, default: Optional[bool] = None
) -> Callable[[str], Optional[bool]]:

    true_values = {"true", "yes", "t", "y"}
    false_values = {"false", "no", "f", "n"}

    def parse(x: str) -> Optional[bool]:
        x = make_str_parser(required=required, default="")(x)
        if x == "":
            return default
        elif len(x.split()) > 1 or x.lower() not in (true_values | false_values):
            raise ValueError(f"Could not parse '{x}' as a boolean")
        elif x.lower() in true_values:
            return True
        else:
            return False

    return parse
