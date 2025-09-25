"""The main module containing the swiftsim-utils CLI tool."""

from swiftsim_utils.modes import MODE_MODULES
from swiftsim_utils.multi_mode_args import MultiModeCLIArgs
from swiftsim_utils.params import load_parameters
from swiftsim_utils.profile import load_swift_profile


def main(argv: list[str] | None = None) -> None:
    """Run the SWIFT-utils command-line interface.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    # Load the config first, we may need it later and this will cache it for
    # future use.
    _ = load_swift_profile()

    # Parse the multi-mode command-line arguments
    multi_args = MultiModeCLIArgs(argv)

    # Execute each mode in sequence
    for mode_name, args in multi_args.modes:
        # Load the parameters if they exist (for modes that use them)
        _ = load_parameters(getattr(args, "params", None))

        # Get the mode module and execute it
        mode_module = MODE_MODULES[mode_name]
        mode_module.run(args)
