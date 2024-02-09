from typing import Union


class JobId:
    def __init__(self, job_id: Union[str, int]):
        self._job_id = job_id
