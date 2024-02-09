import dataclasses
import re
from typing import Union

from exauq.core.modelling import Input


class JobId:
    def __init__(self, job_id: Union[str, int]):
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

    def __str__(self):
        return self._job_id

    def __repr__(self):
        return f"{self.__class__.__name__}({self._job_id})"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._job_id == str(other)


@dataclasses.dataclass(init=False)
class Job:
    def __init__(self, id_: Union[JobId, str, int], data: Input) -> None:
        self._id = JobId(id_)
        self._data = data

    @property
    def id(self) -> JobId:
        return self._id

    @property
    def data(self) -> Input:
        return self._data
