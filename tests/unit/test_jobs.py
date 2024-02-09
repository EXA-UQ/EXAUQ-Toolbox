import unittest

from exauq.core.jobs import Job, JobId
from exauq.core.modelling import Input
from tests.utilities.utilities import ExauqTestCase, exact


class TestJobId(ExauqTestCase):
    def test_create_from_integer_string_like(self):
        """A job ID can be created from a non-negative integer or a string consisting only
        of digits."""

        for job_id in ["1", 99, "00001", JobId(0)]:
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

    def test_equality(self):
        """Two JobId instances are equal if they have the same string representation.
        A JobId is not equal to an object that is not also a JobId instance."""

        self.assertEqual(JobId(0), JobId("0"))
        self.assertNotEqual(JobId("1"), "1")
        self.assertNotEqual(JobId(1), JobId("01"))


class TestJob(ExauqTestCase):
    def test_init_valid_ids(self):
        """The ID of a job can be a JobId instance or an object from which a valid JobId
        can be created."""

        for job_id in [JobId(0), "1", 99, "00001"]:
            try:
                _ = Job(id_=job_id, data=Input(0))
            except Exception:
                self.fail(
                    f"Should have been able to construct Job with job_id = {job_id}."
                )


if __name__ == "__main__":
    unittest.main()
