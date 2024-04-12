import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any


def load_classes_from_file(file_path: str, base_classes: list[str]) -> dict[str, Any]:
    """
    Dynamically loads and retrieves a unique subclass for each specified base class from a given Python module.

    This function adds the directory of the specified file to the system path, dynamically imports the module,
    and then scans for classes that are subclasses of the given base classes. It enforces that only one unique subclass
    per base class is present in the module. If more than one subclass is found for any base class, a ValueError is raised.

    Parameters
    ----------
    file_path : str
        The file path of the Python script to load as a module.
    base_classes : list of str
        A list of the names of base classes to find subclasses for in the module.

    Returns
    -------
    dict of str : Any
        A dictionary where keys are the names of the base classes and values are the corresponding
        unique subclasses found in the module.

    Raises
    ------
    ValueError
        If more than one subclass is found for any of the specified base classes in the module.

    Examples
    --------
    >>> file_path = 'path_to_your_file/my_interfaces.py'
    >>> base_classes = ['HardwareInterface', 'FactoryBaseClass']
    >>> loaded_classes = load_classes_from_file(file_path, base_classes)
    >>> print(loaded_classes)
    {'HardwareInterface': <class 'module.HardwareInterfaceSubclass'>,
     'FactoryBaseClass': <class 'module.FactoryBaseClassSubclass'>}

    Notes
    -----
    The function modifies the sys.path list by appending the directory of the file_path if not already included.
    This is necessary for dynamic importing of modules not in the initial search path.
    """
    file_path = Path(file_path)

    if str(file_path.parent) not in sys.path:
        sys.path.append(str(file_path.parent))

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found_classes = {base_class: [] for base_class in base_classes}

    for name, obj in inspect.getmembers(module, inspect.isclass):
        for base_class in base_classes:
            base_class_type = getattr(module, base_class, None)
            if (
                base_class_type
                and inspect.isclass(obj)
                and issubclass(obj, base_class_type)
                and obj.__name__ != base_class
            ):
                found_classes[base_class].append(obj)

    selected_classes = {}
    for base_class, class_list in found_classes.items():
        if len(class_list) > 1:
            raise ValueError(
                f"More than one subclass of {base_class} found in {file_path}. Please ensure only one subclass per base class."
            )
        elif class_list:
            selected_classes[base_class] = class_list[0]

    return selected_classes
