from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Union

from exauq.core.modelling import Input


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
    data : sequence of Input
        A batch of inputs for a simulator.

    Attributes
    ----------
    id : JobId
        (Read-only) The ID of the job.
    data : tuple[Input]
        (Read-only) The simulator inputs for the job.
    """

    def __init__(self, id_: Union[JobId, str, int], data: Sequence[Input]) -> None:
        self._id = self._parse_id(id_)
        self._data = self._validate_data(data)

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
    def _validate_data(data) -> tuple[Input]:
        if not isinstance(data, Sequence):
            raise TypeError(
                f"Expected 'data' to be a sequence of {Input} objects but received "
                f"{type(data)} instead."
            )
        elif bad_elements := [x for x in data if not isinstance(x, Input)]:
            raise ValueError(
                f"Expected each object in 'data' to be a {Input} but found object of type "
                f"{type(bad_elements[0])}."
            )
        else:
            return tuple(data)

    @property
    def id(self) -> JobId:
        """(Read-only) The ID of the job."""
        return self._id

    @property
    def data(self) -> tuple[Input]:
        """(Read-only) The simulator inputs for the job."""
        return self._data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id_={repr(self.id)}, data={repr(self.data)})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.id == other.id
            and self.data == other.data
        )
