import unittest

from exauq.core.jobs import JobId
from tests.utilities.utilities import ExauqTestCase


class TestJobId(ExauqTestCase):
    def test_create_from_integer_string_like(self):
        """A job ID can be created from an object whose string representation is a
        positive integer."""

        for job_id in ["1", 99]:
            try:
                _ = JobId(job_id)
            except Exception:
                self.fail(
                    f"Should have been able to construct JobId with job_id = {job_id}"
                )


if __name__ == "__main__":
    unittest.main()
