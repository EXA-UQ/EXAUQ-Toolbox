import importlib
import inspect
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional

import click

from cli_rough.utils.state_management import save_state_to_file
from exauq.core.modelling import SimulatorDomain
from exauq.sim_management.hardware import HardwareInterface, SSHInterface
from exauq.sim_management.simulators import Simulator


def get_additional_params(cls, base_params):
    """Get parameters of the class constructor excluding those of the base class."""
    cls_params = inspect.signature(cls.__init__).parameters
    additional_params = {
        name: param for name, param in cls_params.items() if name not in base_params
    }
    return additional_params


@click.group(name="simulation")
def simulation_group():
    """Simulation management commands."""
    pass


@simulation_group.command(name="interface")
@click.pass_context
def init_interface(ctx, file_path, class_name):
    """Initialise a HardwareInterface subclass"""
    pass


@simulation_group.command(name="ssh_interface")
@click.option(
    "--file_path",
    prompt=True,
    type=click.Path(exists=True),
    help="Full file path to the SSHInterface subclass",
)
@click.option("--class_name", prompt=True, help="Name of the SSHInterface subclass")
@click.option("--user", prompt=True, help="SSH Username")
@click.option("--host", prompt=True, help="SSH Hostname or IP address")
@click.option(
    "--key_filename",
    type=click.Path(exists=True),
    help="Path to the SSH private key file",
    default=None,
)
@click.option(
    "--ssh_config_path",
    type=click.Path(exists=True),
    help="Path to the SSH config file",
    default=None,
)
@click.option(
    "--use_ssh_agent",
    is_flag=True,
    help="Use SSH agent for authentication",
    default=False,
)
@click.option("--max_attempts", type=int, help="Max authentication attempts", default=3)
@click.pass_context
def init_ssh_interface(
    ctx,
    file_path,
    class_name,
    user,
    host,
    key_filename,
    ssh_config_path,
    use_ssh_agent,
    max_attempts,
):
    """Initialize an SSHInterface subclass with user-specified parameters."""
    ssh_base_params = [
        "self",
        "user",
        "host",
        "key_filename",
        "ssh_config_path",
        "use_ssh_agent",
        "max_attempts",
    ]

    directory, filename = os.path.split(file_path)
    module_name, _ = os.path.splitext(filename)

    # Add the directory to sys.path
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        # Import the module by name and access the class
        module = import_module(module_name)
        cls = getattr(module, class_name)

        if not issubclass(cls, SSHInterface):
            raise ValueError(f"{class_name} is not a subclass of SSHInterface")

        additional_params = get_additional_params(cls, ssh_base_params)

        # Prompt for additional parameters
        additional_values = {}
        for name, param in additional_params.items():
            if param.default is inspect.Parameter.empty:
                value = click.prompt(f"Please enter value for '{name}'")
            else:
                value = click.prompt(
                    f"Please enter value for '{name}' (default: {param.default})",
                    default=param.default,
                )
            additional_values[name] = value

        # Instantiate the class with both common and additional parameters
        instance = cls(
            user=user,
            host=host,
            key_filename=key_filename,
            ssh_config_path=ssh_config_path,
            use_ssh_agent=use_ssh_agent,
            max_attempts=max_attempts,
            **additional_values,
        )
        ctx.obj["interface"] = instance

        click.echo("SSH Interface subclass initialised successfully.")

    except Exception as e:
        click.echo(f"Failed to initialise SSH interface subclass: {e}")

    # save_state_to_file(ctx.obj)


@simulation_group.command(name="simulator")
@click.option(
    "--log_file",
    type=click.Path(),
    default="simulations.csv",
    help='Path to the simulation log file. Defaults to "simulations.csv".',
)
@click.pass_context
def init_simulator(ctx, log_file):
    """Initialise the Simulator with the specified domain and interface."""
    domain = ctx.obj["simulator_domain"]
    interface = ctx.obj["interface"]

    missing_components = []
    if domain is None:
        missing_components.append("SimulatorDomain")
    if interface is None:
        missing_components.append("HardwareInterface")

    if missing_components:
        missing_str = " and ".join(missing_components)
        click.echo(
            f"{missing_str} {'is' if len(missing_components) == 1 else 'are'} not initialised. Please initialise {'it' if len(missing_components) == 1 else 'them'} first."
        )
        return

    try:
        # Ensure the log file path is absolute
        log_file = str(Path(log_file).resolve())
        simulator = Simulator(
            domain=domain, interface=interface, simulations_log_file=log_file
        )
        ctx.obj["simulator"] = simulator
        click.echo("Simulator initialized successfully.")
    except Exception as e:
        click.echo(f"Failed to initialize Simulator: {e}")


@simulation_group.command(name="submit")
@click.argument("input_parameters", nargs=-1, type=float)
@click.pass_context
def submit_simulation(ctx, input_parameters):
    """Submit a new simulation with specified INPUT_PARAMETERS."""
    simulator = ctx.obj["simulator"]
    # Assuming your simulator has a method like this:
    result = simulator.compute(input_parameters)
    if result is not None:
        click.echo(
            f"Simulation submitted with input parameters {input_parameters}. Result: {result}"
        )
    else:
        click.echo(
            f"Simulation submitted with input parameters {input_parameters}. Awaiting results."
        )


@simulation_group.command(name="status")
@click.pass_context
def get_job_status(ctx):
    """Get the status of the Simulation with job id"""
    pass


@simulation_group.command(name="list")
@click.pass_context
def list_jobs(ctx):
    """Display list of simulator jobs"""
    pass


@simulation_group.command(name="monitor")
@click.pass_context
def monitor_jobs(ctx):
    """launch live simulator job monitor"""
    pass


@simulation_group.command(name="monitor")
@click.pass_context
def get_result(ctx):
    """Retrieve result for simulator job"""
    pass
