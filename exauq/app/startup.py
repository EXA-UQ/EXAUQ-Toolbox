import inspect
import json
import pathlib
from collections import OrderedDict
from typing import Any, Callable, Optional

from exauq.sim_management.hardware import HardwareInterface, UnixServerScriptInterface
from exauq.sim_management.types import FilePath


class HardwareInterfaceFactory:
    """Provides support for creating hardware interfaces interactively.

    An instance of this class creates a specific class of hardware interface, as specified
    by the `hardware_cls` parameter, with hardware parameter values stored in
    `self.hardware_parameters`. More precisely, creation of hardware interfaces is
    done by calling the `self.create_hardware` method, which initialises an instance of
    `self.hardware_cls` with arguments stored in `self.hardware_parameters`. To support
    applications that need to gather hardware parameter values interactively from a user,
    the `self.interactive_prompts` attribute specifies prompts to issue to the user and
    the `self.parsers` attribute defines functions for parsing strings into parameter
    values.

    Those wishing to create their own classes for interactively creating a certain class
    of hardware interface can derive from this class. In particular, one may wish to
    override the methods `self.serialise_hardware_parameters` and
    `self.load_hardware_parameters` concerning the serialisation and loading of parameters
    required for initialising hardware interfaces; the `self.make_parsers` method for
    creating `self.parsers` at factory initialisation; the `interactive_prompts` property
    for creating text to display to users when gathering hardware parameter values; and
    possibly the main `self.create_hardware` method for constructing a instance of the
    hardware interface from parameter values stored in `self.hardware_parameters`.

    Parameters
    ----------
    hardware_cls : type
        A class deriving from ``exauq.sim_management.hardware.HardwareInterface``.
        Instances of this class will be created by the method `self.create_hardware`.

    Attributes
    ----------

    hardware_cls : type
        The class of hardware interface that this factory instance will create.

    hardware_parameters : OrderedDict[str, Any]
        An (ordered) mapping of parameter names and values for initialising a hardware
        interface, i.e. for providing to the ``__init__`` method of `self.hardware_cls`.

    parsers : dict[str, typing.Callable[[str], Any]]
        A mapping of parameter names and fuctions for parsing strings to values of the
        parameters.

    Raises
    ------
    ValueError
        If the `hardware_cls` does not derive from
        ``exauq.sim_management.hardware.HardwareInterface``.
    """

    _MISSING = inspect.Parameter.empty

    def __init__(self, hardware_cls: type):
        if not issubclass(hardware_cls, HardwareInterface):
            raise ValueError(
                f"{hardware_cls} does not inherit from {HardwareInterface.__name__}"
            )
        else:
            self._hardware_cls = hardware_cls

        self.hardware_parameters = self._get_init_params(hardware_cls)
        self.parsers = self.make_parsers()

    @staticmethod
    def _get_init_params(cls_: type) -> OrderedDict[str, Any]:
        """Get the names and default values for the parameters of a class's ``__init__``.

        Parameters appear as names in the returned ordered dict, ordered by appearance in
        the ``__init__`` signature. If the parameter does not have a default value, then
        the returned ordered dict will store the value ``inspect.Parameter.empty`` under
        the parameter name.  The parameter ``self`` is not included in the returned dict.
        """
        return OrderedDict(
            (param.name, param.default)
            for param in inspect.signature(cls_.__init__).parameters.values()
            if not param.name == "self"
        )

    def make_parsers(self) -> dict[str, Callable[[str], Any]]:
        """Make parsers for converting strings to parameter values.

        This method is used to set `self.parsers` upon initialisation of this object. The
        keys in the returned dict are the same keys found in `self.hardware_parameters`.
        The values are functions that essentially return the same string as given them,
        except that (1) they strip any leading/trailing whitespace, and (2) they map the
        empty string to the default parameter value if applicable, or None otherwise.

        Classes deriving from `HardwareInterfaceFactory` may wish to override this method
        to provide custom parsers for parameters that are not strings (or have particular
        structure, such as file paths. Note that any implementation should ensure the
        returned dict has exactly the same keys as `self.hardware_parameters`.

        Returns
        -------
        dict[str, Callable[[str], Any]]
            A mapping of hardware interface parameter names to parsing functions.
        """
        return {
            param: make_default_parser()
            for param, value in self.hardware_parameters.items()
            if value is self._MISSING
        } | {
            param: make_default_parser(required=False, default=value)
            for param, value in self.hardware_parameters.items()
            if value is not self._MISSING
        }

    @property
    def hardware_cls(self) -> type:
        """The class of hardware interface that this factory instance will create."""
        return self._hardware_cls

    def set_param_from_str(self, param: str, value: str) -> None:
        """Set a hardware interface parameter value from a string.

        Sets the value of `param` in the `self.hardware_parameters` dict. The `value`
        string is parsed according to the corresponding function stored in `self.parsers`.

        Parameters
        ----------
        param : str
            The parameter to set.
        value : str
            A string representation of the value to set the parameter to.
        """

        self.hardware_parameters[param] = self.parsers[param](value)

    def serialise_hardware_parameters(self, params_file: FilePath) -> None:
        """Serialise parameters required for initialising a hardware interface.

        This method serialises parameter values required to initialise an instance of
        `self.hardware_cls`. The present implementation does this by serialising
        `self.hardware_parameters` as JSON. Classes deriving from
        `HardwareInterfaceFactory` may wish to override this implementation, especially if
        they require storing objects that cannot be serialised as JSON.

        Parameters
        ----------
        params_file : exauq.sim_management.types.FilePath
            Path to the file to serialise the parameters to.
        """

        with open(params_file, mode="w") as f:
            json.dump(self.hardware_parameters, f, indent=4)

        return None

    def load_hardware_parameters(self, params_file: FilePath) -> None:
        """Load parameters for initialising a hardware interface.

        Parameters are read from the given file and stored in `self.hardware_parameters`.
        In the present implementation, it is expected that the file stores the
        parameters as a JSON object defining the parameter names and corresponding values;
        this JSON is then deserialised into a Python (ordered) dict. Classes deriving from
        `HardwareInterfaceFactory` may wish to override this implementation, especially if
        they require storing objects that cannot be serialised as JSON.

        Parameters
        ----------
        params_file : FilePath
            The file to load parameters from.

        Raises
        ------
        AssertionError
            If the parameters read from the file do not agree with those required to
            initialise an instance of `self.hardware_cls` (including optional parameters).
        """

        with open(params_file, mode="r") as f:
            params = json.load(f)

        if set(params) == set(self.hardware_parameters):
            self.hardware_parameters = OrderedDict(params)
            return None
        else:
            raise AssertionError(
                "The deserialised parameter names do not agree with those required to "
                f"initialise {self._hardware_cls}."
            )

    def create_hardware(self) -> HardwareInterface:
        """Create an instance of a hardware interface based on stored parameter values.

        Returns an instance of `self.hardware_cls` by initialising with parameter values
        stored in `self.hardware_parameters`.

        Returns
        -------
        HardwareInterface
            An instance of `self.hardware_cls`, initialised from stored parameter values.

        Raises
        ------
        AssertionError
            If any parameter values in `self.hardware_params` are missing.
        """
        missings_params = {
            param
            for param, value in self.hardware_parameters.items()
            if value is self._MISSING
        }
        if missings_params:
            raise AssertionError(
                f"Cannot initialise instance of {self._hardware_cls} while missing params "
                f"{missings_params}."
            )
        else:
            return self._hardware_cls(**self.hardware_parameters)

    @property
    def interactive_prompts(self) -> OrderedDict[str, str]:
        """Prompts to be used when setting parameter values from user input.

        The current implementation assigns each parameter a prompt which gives the name of
        the hardware interface to be created and the name of the parameter to be defined,
        as well as the default value if the parameter is optional. Classes deriving from
        `HardwareInterfaceFactory` may wish to override this implementation to provide
        more user-friendly prompts.

        Returns
        -------
        OrderedDict[str, str]
            The prompts to use for each parameter. The names of the (ordered) dict are
            parameter names as stored in `self.hardware_parameters` and values are the
            corresponding prompt.
        """
        prompts = OrderedDict()
        for param, value in self.hardware_parameters.items():
            if value is self._MISSING:
                prompts[param] = f"{self._hardware_cls.__name__} {param}"
            else:
                prompts[param] = (
                    f"{self._hardware_cls.__name__} {param} (default: {value})"
                )

        return prompts


class UnixServerScriptInterfaceFactory(HardwareInterfaceFactory):
    def __init__(self):
        super().__init__(UnixServerScriptInterface)

    def make_parsers(self) -> dict[str, Callable[[str], Any]]:
        """Make parsers for converting strings to parameter values.

        This method is used to set `self.parsers` upon initialisation of this object. It
        overrides the `make_parsers` method of ``HardwareInterfaceFactory`` by replacing
        parsers for the 'script_path' and 'use_ssh_agent' parameters with POSIX path and
        boolean parsers, respectively. Other parameters are set as per the parent class
        implementation.

        Returns
        -------
        dict[str, Callable[[str], Any]]
            A mapping of hardware interface parameter names to parsing functions.
        """
        return super().make_parsers() | {
            "script_path": make_posix_path_parser(),
            "use_ssh_agent": make_bool_parser(
                required=False,
                default=self.hardware_parameters["use_ssh_agent"],
            ),
        }

    def create_hardware(self) -> UnixServerScriptInterface:
        """Create an instance of ``UnixServerScriptInterface`` from stored parameter
        values.

        Returns
        -------
        UnixServerScriptInterface
            A hardware interface representing a script on a Unix server, initialised with
            parameter values stored in `self.hardware_parameters`.

        Raises
        ------
        AssertionError
            If any parameter values in `self.hardware_params` are missing.
        """

        hardware = super().create_hardware()
        self.hardware_parameters["workspace_dir"] = hardware.workspace_dir
        return hardware

    @property
    def interactive_prompts(self) -> OrderedDict[str, str]:
        """Prompts to be used when setting parameter values from user input.

        Returns
        -------
        OrderedDict[str, str]
            The prompts to use for each parameter. The names of the (ordered) dict are
            parameter names as stored in `self.hardware_parameters` and values are the
            corresponding prompt.
        """

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


def make_default_parser(
    required: bool = True, default: Optional[Any] = None
) -> Callable[[str], Any]:
    """Make a function for parsing a string, with handling for empty or blank strings.

    The returned parser function is designed to be used for parsing user-provided text.
    The parser takes in a string, strips out any leading/trailing whitespace, and then
    does one of the following with the result:

    * Returns it if it is nonempty.
    * Raises a ValueError if it is empty and `required` is ``True``.
    * Returns a specified default value if it is empty and `required` is ``False``.

    Parameters
    ----------
    required : bool, optional
        (Default: True) Whether the returned function should raise a ValueError on strings
        that are empty or only contain whitespace.
    default : Optional[Any], optional
        (Default: None) The value for the returned function to return on strings that are
        empty or only contain whitespace, in the case where `required` is ``False``.

    Returns
    -------
    Callable[[str], Any]
        A function to parse strings with handling for strings that are empty or only
        contain whitespace, as specified by the values of `required` and `default`.
    """

    def parse(x: str) -> Any:
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
    """Make a function for parsing a POSIX-compliant path, with handling for blank
    strings.

    The returned parser function is designed to be used for parsing user-provided text.
    If the provided string defines a valid POSIX path, then the parser will return it
    as-is (after stripping out any leading/trailing whitespace). If the string is empty
    or only contains whitespace, then the specified default path is returned as a string
    if `required` is ``False`` and a ValueError is raised if `required` is ``True``.

    Parameters
    ----------
    required : bool, optional
        (Default: True) Whether the returned function should raise a ValueError on strings
        that are empty or only contain whitespace.
    default : Optional[exauq.sim_management.types.FilePath], optional
        (Default: None) The value for the returned function to return on strings that are
        empty or only contain whitespace, in the case where `required` is ``False``.

    Returns
    -------
    Callable[[str], Optional[str]]
        A function to parse strings as POSIX file paths with handling for strings that are
        empty or only contain whitespace, as specified by the values of `required` and
        `default`.
    """

    def parse(x: str) -> Optional[str]:
        x = make_default_parser(required=required, default="")(x)
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
    """Make a function for parsing a string as a bool, with handling for blank strings.

    The parser takes in a string, strips out any leading/trailing whitespace, and then
    does one of the following with the result:

    * Returns ``True`` if it is one of 'true', 'yes', 't' or 'y' after converting to lower
      case.
    * Returns ``False`` if it is one of 'false', 'no', 'f' or 'n' after converting to
      lower case.
    * Returns a specified default value if `required` is ``False`` and the string is
      empty.
    * Raises a ValueError otherwise.

    Parameters
    ----------
    required : bool, optional
        (Default: True) Whether the returned function should raise a ValueError on strings
        that are empty or only contain whitespace.
    default : Optional[exauq.sim_management.types.FilePath], optional
        (Default: None) The value for the returned function to return on strings that are
        empty or only contain whitespace, in the case where `required` is ``False``.

    Returns
    -------
    Callable[[str], Optional[str]]
        A function to parse strings as booleans with handling for strings that are
        empty or only contain whitespace, as specified by the values of `required` and
        `default`.
    """

    true_values = {"true", "yes", "t", "y"}
    false_values = {"false", "no", "f", "n"}

    def parse(x: str) -> Optional[bool]:
        x = make_default_parser(required=required, default="")(x)
        if x == "":
            return default
        elif len(x.split()) > 1 or x.lower() not in (true_values | false_values):
            raise ValueError(f"Could not parse '{x}' as a boolean")
        elif x.lower() in true_values:
            return True
        else:
            return False

    return parse
