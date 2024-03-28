import click
import pickle

from cli.simulation.commands import simulation_group

from cli.utils.state_management import load_state_from_file, save_state_to_file

from exauq.core.modelling import SimulatorDomain





def parse_bounds(bounds_str):
    """Parse the bounds string into a list of tuple pairs."""
    bounds = []
    for dim_bounds in bounds_str.split(':'):
        lower, upper = map(float, dim_bounds.split(','))
        bounds.append((lower, upper))
    return bounds


@click.group()
@click.pass_context
def cli(ctx):
    """Exauq-Toolbox CLI: A suite of tools for uncertainty quantification and analysis."""
    ctx.ensure_object(dict)
    ctx.obj['simulator_domain'] = None
    ctx.obj['interface'] = None
    ctx.obj['simulator'] = None

    loaded_state = load_state_from_file()
    if loaded_state:
        ctx.obj = loaded_state


@cli.command(name="domain")
@click.option('--bounds',
              help='Optional bounds for the simulator domain specified as "lower1,upper1:lower2,upper2,..."')
@click.pass_context
def init_sim_domain(ctx, bounds):
    """Initialise the SimulatorDomain"""
    if bounds:
        try:
            parsed_bounds = parse_bounds(bounds)
            domain = SimulatorDomain(bounds=parsed_bounds)
            ctx.obj['simulator_domain'] = domain
            click.echo("Simulator domain initialized successfully.")
            # save_state_to_file(ctx.obj)
        except ValueError as e:
            click.echo(f"Error parsing bounds: {e}")
    else:
        dim = click.prompt("Enter the number of dimensions for the simulator domain", type=int)
        bounds = []
        for i in range(dim):
            lower = click.prompt(f"Enter lower bound for dimension {i + 1}", type=float)
            upper = click.prompt(f"Enter upper bound for dimension {i + 1}", type=float)
            bounds.append((lower, upper))

        try:
            domain = SimulatorDomain(bounds=bounds)
            ctx.obj['simulator_domain'] = domain
            click.echo("Simulator domain initialized successfully.")
            # save_state_to_file(ctx.obj)
        except ValueError as e:
            click.echo(f"Failed to initialize simulator domain: {e}")


# Add the simulation command group to the main CLI
cli.add_command(simulation_group, name='simulation')

if __name__ == '__main__':
    cli()

