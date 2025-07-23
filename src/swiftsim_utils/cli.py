"""The main module containing the swiftsim-utils CLI tool."""

from swiftsim_utils.cmd_args import SwiftUtilsArgs
from swiftsim_utils.config import config_swift_utils


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the swiftsim-utils CLI tool.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    # Parse the command-line arguments
    args = SwiftUtilsArgs(argv).args

    # Run the appropriate command based on the mode
    if args.mode == "config":
        config_swift_utils()
