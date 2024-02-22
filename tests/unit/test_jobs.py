import unittest

from exauq.core.modelling import Input
from exauq.sim_management.jobs import Job, JobId
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

    def test_hashable(self):
        """JobId objects are hashable."""

        try:
            _ = {JobId(1)}
        except TypeError:
            self.fail("Expected object to be hashable")


class TestJob(ExauqTestCase):
    def test_init_valid_ids(self):
        """The ID of a job can be a JobId instance or an object from which a valid JobId
        can be created."""

        for job_id in [JobId(0), "1", 99, "00001"]:
            try:
                _ = Job(id_=job_id, data=[Input(0)])
            except Exception:
                self.fail(
                    f"Should have been able to construct Job with job_id = {job_id}."
                )

    def test_init_arg_validation(self):
        """A ValueError is raised if the id does not define a valid JobId instance.
        A TypeError is raised if the data is not a sequence.
        A ValueError is raised if the data is not a sequence of Inputs."""

        id_ = "a"
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected 'id_' to define a valid {JobId}, but received "
                f"'{str(id_)}' instead."
            ),
        ):
            _ = Job(id_=id_, data=Input(0))

        data = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'data' to be a sequence of {Input} objects but received {type(data)} instead."
            ),
        ):
            _ = Job(id_=1, data=data)

        data = [Input(0), 2]
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected each object in 'data' to be a {Input} but found object of type {type(data[1])}."
            ),
        ):
            _ = Job(id_=1, data=data)

    def test_retrieve_id_and_data_from_properties(self):
        """The supplied ID and input data can be retrieved from the properties."""

        id_ = 1
        data = (Input(0),)
        job = Job(id_, data)
        self.assertEqual(JobId(id_), job.id)

    def test_retrieve_data_as_tuple(self):
        """The supplied input data is retrieved as a tuple."""

        id_ = 1
        data = [Input(0), Input(1)]
        job = Job(id_, data)
        self.assertEqual(tuple(data), job.data)

    def test_immutable_attributes(self):
        """A Job object's attributes are immutable."""

        job = Job(id_=1, data=[Input(0)])
        with self.assertRaises(AttributeError):
            job.id = JobId(2)

        with self.assertRaises(AttributeError):
            job.data = (Input(-1),)

    def test_equality(self):
        """Two jobs are equal precisely when their IDs and input data are equal."""

        self.assertEqual(Job(id_=1, data=[Input(0)]), Job(id_="1", data=[Input(0)]))

        # Not another Job
        self.assertNotEqual(Job(id_=1, data=[Input(0)]), (JobId(1), [Input(0)]))

        # Different ID
        self.assertNotEqual(Job(id_=2, data=[Input(0)]), Job(id_=1, data=[Input(0)]))

        # Different data
        self.assertNotEqual(Job(id_=1, data=[Input(0)]), Job(id_=1, data=[Input(1)]))
        self.assertNotEqual(Job(id_=1, data=[Input(0)]), Job(id_=1, data=[Input(0, 0)]))
        self.assertNotEqual(
            Job(id_=1, data=[Input(0)]), Job(id_=1, data=[Input(0), Input(1)])
        )


if __name__ == "__main__":
    unittest.main()
