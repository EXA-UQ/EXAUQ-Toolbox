import unittest

from exauq.core.jobs import JobId
from tests.utilities.utilities import ExauqTestCase, exact


class TestJobId(ExauqTestCase):
    def test_create_from_integer_string_like(self):
        """A job ID can be created from a non-negative integer or a string consisting only
        of digits."""

        for job_id in ["1", 99, "00001"]:
            try:
                _ = JobId(job_id)
            except Exception:
                self.fail(
                    f"Should have been able to construct JobId with job_id = {job_id}."
                )

    def test_non_integer_string_like_error(self):
        """A ValueError is raised if the supplied job ID whose string representation
        contains characters other than digits."""

        for job_id in ["", "rm -rf ~", "!", -1, "0.1", "1e10", 2j, [1]]:
            with self.subTest(job_id=job_id):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(
                        "Expected 'job_id' to define a string consisting only of digits, "
                        f"but received '{str(job_id)}' instead."
                    ),
                ):
                    _ = JobId(job_id)

    def test_string_method(self):
        """A JobID instance can be converted to a string."""

        for job_id in ["1", 99, "00001"]:
            self.assertEqual(str(job_id), str(JobId(job_id)))


if __name__ == "__main__":
    unittest.main()
