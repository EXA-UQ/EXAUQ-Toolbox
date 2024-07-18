import itertools
import unittest
from unittest.mock import MagicMock, call, patch

from paramiko.ssh_exception import AuthenticationException

from exauq.sim_management.hardware import HardwareInterface, SSHInterface


class TestHardwareInterface(unittest.TestCase):
    # Test that an error is raised when trying to instantiate the abstract base class
    def test_abc_instantiation(self):
        with self.assertRaises(TypeError):
            _ = HardwareInterface()


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

    @patch("exauq.sim_management.hardware.Connection")
    def test_init_with_key_filename(self, mock_conn):
        """Test that a connection to a server is established using a specified private key file
        upon initialisation."""

        _ = MockedSSHInterface("user", "host", "interface_01", key_filename="/path/to/key")
        mock_conn.assert_called_once_with(
            "user@host", connect_kwargs={"key_filename": "/path/to/key"}
        )

    @patch("exauq.sim_management.hardware.Connection")
    @patch("exauq.sim_management.hardware.Config", return_value=MagicMock())
    def test_init_with_ssh_config_path(self, mock_config, mock_conn):
        """Test that a connection to a server is established using the specified SSH config path
        upon initialisation."""

        _ = MockedSSHInterface("user", "host", "interface_01", ssh_config_path="/path/to/config")
        mock_config.assert_called_once_with(
            overrides={"ssh_config_path": "/path/to/config"}
        )
        mock_conn.assert_called_once_with("host", config=mock_config.return_value)

    @patch("exauq.sim_management.hardware.Connection")
    def test_init_with_ssh_agent(self, mock_conn):
        """Test that a connection to a server is established using the SSH agent upon
        initialisation."""

        _ = MockedSSHInterface("user", "host", "interface_01", use_ssh_agent=True)
        mock_conn.assert_called_once_with("user@host")

    def test_init_with_multiple_auth_methods(self):
        """Test that initialisation raises an error when multiple authentication methods are
        specified."""

        non_defaults = {
            "key_filename": "/path/to/key",
            "ssh_config_path": "/path/to/config",
            "use_ssh_agent": True,
        }
        pairwise_kwargs = map(dict, itertools.combinations(non_defaults.items(), 2))
        for kwargs in pairwise_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MockedSSHInterface("user", "host", "interface_01", **kwargs)

        with self.assertRaises(ValueError):
            MockedSSHInterface("user", "host", "interface_01", **non_defaults)

    @patch("exauq.sim_management.hardware.getpass.getpass", return_value="mock_password")
    @patch("exauq.sim_management.hardware.Connection", side_effect=Exception("Connection failed"))
    def test_failed_connection(self, mock_conn, mock_getpass):
        """Test that an exception is raised when a connection fails during initialization."""

        with self.assertRaises(Exception):
            MockedSSHInterface("user", "host")

    @patch("builtins.print")
    @patch(
        "exauq.sim_management.hardware.getpass.getpass",
        side_effect=["wrong_pass1", "wrong_pass2", "wrong_pass3"],
    )
    @patch("exauq.sim_management.hardware.Connection", side_effect=AuthenticationException())
    def test_max_attempts_authentication_exception(
        self, mock_conn, mock_getpass, mock_print
    ):
        """Test that an AuthenticationException is raised after the maximum number of wrong
        password attempts."""

        # Set max_attempts to 3
        with self.assertRaises(AuthenticationException):
            _ = MockedSSHInterface("user", "host", "interface_01", max_attempts=3)

        # Ensure getpass was called 3 times (once for each wrong password attempt)
        self.assertEqual(mock_getpass.call_count, 3)

        # Check the arguments passed to the print function on each call
        mock_print.assert_has_calls(
            [
                call("Failed to authenticate. Please try again."),
                call("Failed to authenticate. Please try again."),
                call("Maximum number of attempts exceeded."),
            ]
        )

    @patch("exauq.sim_management.hardware.getpass.getpass", side_effect=["correct_pass"])
    @patch("exauq.sim_management.hardware.Connection")
    def test_successful_password_auth_on_first_try(self, mock_conn, mock_getpass):
        """Test that a connection is successfully established on the first password attempt."""

        # This test is for a successful connection on the first attempt, so no exception should be raised.
        _ = MockedSSHInterface("user", "host", "interface_01", max_attempts=3)

        # Ensure getpass was called only once
        self.assertEqual(mock_getpass.call_count, 1)

        # Ensure Connection was called using correct password
        mock_conn.assert_called_once_with(
            "user@host", connect_kwargs={"password": "correct_pass"}
        )

    @patch("exauq.sim_management.hardware.Connection.run", return_value=True)
    def test_successful_check_connection(self, mock_conn):
        """Test checking a connection when it's successful."""

        try:
            _ = MockedSSHInterface("user", "host", "interface_01", key_filename="")
            # If we've reached here, no exception was raised, which is what we want
        except Exception as e:
            self.fail(f"_check_connection raised an exception: {e}")

    @patch(
        "exauq.sim_management.hardware.Connection.run", side_effect=Exception("Command failed")
    )
    def test_failed_check_connection(self, mock_conn):
        """Test checking a connection when it fails."""

        with self.assertRaises(Exception) as context:
            _ = MockedSSHInterface("user", "sample_host", "interface_01", key_filename="")

        # Check the exception message
        self.assertEqual(
            str(context.exception), "Could not connect to sample_host: Command failed"
        )

    @patch("exauq.sim_management.hardware.Connection")
    def test_entry_method(self, mock_conn):
        """Test that the __enter__ method of the context manager returns the instance itself."""

        interface = MockedSSHInterface("user", "sample_host", "interface_01", key_filename="")

        with interface as result:
            self.assertIs(interface, result)

    @patch("exauq.sim_management.hardware.Connection")
    def test_exit_method(self, mock_conn):
        """Test that the __exit__ method of the context manager closes the connection."""

        interface = MockedSSHInterface("user", "sample_host", "interface_01", key_filename="")

        # Test the __exit__ method
        with interface:
            pass

        interface._conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
