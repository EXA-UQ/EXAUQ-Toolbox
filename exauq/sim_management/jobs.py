from __future__ import annotations

import re
from typing import Optional, Union

from exauq.core.modelling import Input
from exauq.utilities.string_validation import validate_interface_name


class JobId:
    """A unique identifier for a job.

    A job ID can only consist of digits. A string representation of the ID can be obtained
    using the ``str`` function.

    Parameters
    ----------
    job_id : Union[str, int, JobId]
        A non-negative integer, or a string consisting only of digits, or another instance
        of ``JobId``.
    """

    def __init__(self, job_id: Union[str, int, JobId]):
        self._job_id = self._parse(job_id)

    @staticmethod
    def _parse(job_id) -> str:
        job_id_str = str(job_id)
        if re.fullmatch("[0-9]+", job_id_str):
            return job_id_str
        else:
            raise ValueError(
                "Expected 'job_id' to define a string consisting only of digits, "
                f"but received '{str(job_id)}' instead."
            )

    def __str__(self) -> str:
        return self._job_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._job_id)})"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self._job_id == str(other)

    def __hash__(self):
        return hash(self._job_id)


class Job:
    """A job consisting of input data for a simulator.

    Parameters
    ----------
    id_ : Union[JobId, str, int]
        The ID of the job. If a string or an integer is provided then it should define a
        valid ``JobId`` instance.
    data : Input
        An input for a simulator.
    level : int
        The level of the job.
    interface_name : Optional[str]
        The interface name that the job is associated with.

    Attributes
    ----------
    id : JobId
        (Read-only) The ID of the job.
    data : Input
        (Read-only) The simulator input for the job.
    level : int
        (Read-only) The level of the job.
    interface_name : Optional[str]
        (Read-only) The interface name of the job.
    """

    def __init__(
        self,
        id_: Union[JobId, str, int],
        data: Input,
        level: int = 1,
        interface_name: Optional[str] = None,
    ) -> None:
        self._id = self._parse_id(id_)
        self._data = self._validate_data(data)
        self._level = self._validate_level(level)
        self._interface_name = validate_interface_name(interface_name)

    @staticmethod
    def _parse_id(id_) -> JobId:
        try:
            return JobId(id_)
        except ValueError:
            raise ValueError(
                f"Expected 'id_' to define a valid {JobId}, but received '{str(id_)}' "
                "instead."
            )

    @staticmethod
    def _validate_data(data) -> Input:
        if not isinstance(data, Input):
            raise TypeError(
                f"Expected 'data' to be of type {Input} but received {type(data)} instead."
            )
        else:
            return data

    @staticmethod
    def _validate_level(level: int) -> int:
        if not isinstance(level, int):
            raise TypeError(
                f"Expected 'level' to be of type {int} but received {type(level)} instead."
            )
        return level

    @property
    def id(self) -> JobId:
        """(Read-only) The ID of the job."""
        return self._id

    @property
    def data(self) -> Input:
        """(Read-only) The simulator input for the job."""
        return self._data

    @property
    def level(self) -> int:
        """(Read-only) The level of the job."""
        return self._level

    @property
    def interface_name(self) -> Optional[str]:
        """(Read-only) The interface name of the job."""
        return self._interface_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id_={repr(self.id)}, data={repr(self.data)}, level={self.level}, interface_name={self.interface_name})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.id == other.id
            and self.data == other.data
            and self.level == other.level
            and self.interface_name == other.interface_name
        )
