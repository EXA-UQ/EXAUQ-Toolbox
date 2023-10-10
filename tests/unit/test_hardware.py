import unittest
from unittest.mock import MagicMock, patch

from exauq.core.hardware import HardwareInterface, SSHInterface
from paramiko.ssh_exception import AuthenticationException


class TestHardwareInterface(unittest.TestCase):
    # Test that an error is raised when trying to instantiate the abstract base class
    def test_abc_instantiation(self):
        with self.assertRaises(TypeError):
            a = HardwareInterface()


class MockedSSHInterface(SSHInterface):
    def submit_job(self, job):
        pass

    def get_job_status(self, job_id):
        pass

    def get_job_output(self, job_id):
        pass

    def cancel_job(self, job_id):
        pass

    def wait_for_job(self, job_id):
        pass


class TestSSHInterface(unittest.TestCase):
    """Test cases for the SSHInterface class."""

    @patch("exauq.core.hardware.getpass.getpass", return_value="mock_password")
    @patch("exauq.core.hardware.Connection")
    def test_init_with_key_filename(self, MockConnection, MockGetpass):
        """Test that a connection to a server is established using a specified private key file
        upon initialisation."""

        interface = MockedSSHInterface("user", "host", key_filename="/path/to/key")
        MockConnection.assert_called_once_with(
            "user@host", connect_kwargs={"key_filename": "/path/to/key"}
        )

    @patch("exauq.core.hardware.getpass.getpass", return_value="mock_password")
    @patch("exauq.core.hardware.Connection")
    @patch("exauq.core.hardware.Config", return_value=MagicMock())
    def test_init_with_ssh_config_path(self, MockConfig, MockConnection, MockGetpass):
        """Test that a connection to a server is established using the specified SSH config path
        upon initialisation."""

        interface = MockedSSHInterface(
            "user", "host", ssh_config_path="/path/to/config"
        )
        MockConfig.assert_called_once_with(
            overrides={"ssh_config_path": "/path/to/config"}
        )

    @patch("exauq.core.hardware.getpass.getpass", return_value="mock_password")
    @patch("exauq.core.hardware.Connection")
    def test_init_with_ssh_agent(self, MockConnection, MockGetpass):
        """Test that a connection to a server is established using the SSH agent upon
        initialisation."""

        interface = MockedSSHInterface("user", "host", use_ssh_agent=True)
        MockConnection.assert_called_once_with("user@host")

    @patch("exauq.core.hardware.getpass.getpass", return_value="mock_password")
    def test_init_with_multiple_auth_methods(self, MockGetpass):
        """Test that initialisation raises an error when multiple authentication methods are
        specified."""

        with self.assertRaises(ValueError):
            MockedSSHInterface(
                "user",
                "host",
                key_filename="/path/to/key",
                ssh_config_path="/path/to/config",
            )

    @patch("exauq.core.hardware.getpass.getpass", return_value="mock_password")
    @patch("exauq.core.hardware.Connection", side_effect=Exception("Connection failed"))
    def test_failed_connection(self, MockConnection, MockGetpass):
        """Test that an exception is raised when a connection fails during initialization."""

        with self.assertRaises(Exception):
            MockedSSHInterface("user", "host")

    @patch(
        "exauq.core.hardware.getpass.getpass",
        side_effect=["wrong_pass1", "wrong_pass2", "wrong_pass3"],
    )
    @patch("exauq.core.hardware.Connection", side_effect=AuthenticationException())
    def test_max_attempts_authentication_exception(self, MockConnection, MockGetpass):
        """Test that an AuthenticationException is raised after the maximum number of wrong
        password attempts."""

        # Set max_attempts to 3
        with self.assertRaises(AuthenticationException):
            interface = MockedSSHInterface("user", "host", max_attempts=3)

        # Ensure getpass was called 3 times (once for each wrong password attempt)
        self.assertEqual(MockGetpass.call_count, 3)

    @patch("exauq.core.hardware.getpass.getpass", side_effect=["correct_pass"])
    @patch("exauq.core.hardware.Connection")
    def test_successful_password_auth_on_first_try(self, MockConnection, MockGetpass):
        """Test that a connection is successfully established on the first password attempt."""

        # This test is for a successful connection on the first attempt, so no exception should be raised.
        interface = MockedSSHInterface("user", "host", max_attempts=3)

        # Ensure getpass was called only once
        self.assertEqual(MockGetpass.call_count, 1)

    @patch.object(MockedSSHInterface, "_conn", create=True)
    def test_successful_check_connection(self, mock_conn):
        """Test that checking the connection succeeds without raising an exception when it's
        valid."""

        # Mock the run method to simulate a successful command execution
        mock_conn.run.return_value = True
        mock_conn.original_host = "sample_host"

        # Create an instance of SSHInterface (without triggering the actual constructor)
        interface = MockedSSHInterface.__new__(MockedSSHInterface)

        # Assign the mock connection to the instance
        interface._conn = mock_conn

        try:
            interface._check_connection()
            # If we've reached here, no exception was raised, which is what we want
        except Exception as e:
            self.fail(f"_check_connection raised an exception: {e}")

    @patch.object(MockedSSHInterface, "_conn", create=True)
    def test_failed_check_connection(self, mock_conn):
        """Test that checking a connection raises an exception when it fails."""

        # Mock the run method to raise an Exception
        mock_conn.run.side_effect = Exception("Command failed")
        mock_conn.original_host = "sample_host"

        # Create an instance of SSHInterface (without triggering the actual constructor)
        interface = MockedSSHInterface.__new__(MockedSSHInterface)

        # Assign the mock connection to the instance
        interface._conn = mock_conn

        with self.assertRaises(Exception) as context:
            interface._check_connection()

        # Check the exception message
        self.assertEqual(
            str(context.exception), "Could not connect to sample_host: Command failed"
        )

    @patch.object(MockedSSHInterface, "_conn", create=True)
    def test_entry_method(self, mock_conn):
        """Test that the __enter__ method of the context manager returns the instance itself."""

        # Create an instance of SSHInterface (without triggering the actual constructor)
        interface = MockedSSHInterface.__new__(MockedSSHInterface)
        interface._conn = mock_conn

        # Test the __enter__ method
        with interface as returned_instance:
            self.assertEqual(returned_instance, interface)

    @patch.object(MockedSSHInterface, "_conn", create=True)
    def test_exit_method(self, mock_conn):
        """Test that the __exit__ method of the context manager closes the connection."""

        # Create an instance of SSHInterface (without triggering the actual constructor)
        interface = MockedSSHInterface.__new__(MockedSSHInterface)
        interface._conn = mock_conn

        # Test the __exit__ method
        with interface:
            pass
        mock_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
