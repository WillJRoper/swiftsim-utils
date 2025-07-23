"""The main module containing the swiftsim-utils CLI tool."""

from swiftsim_utils.cmd_args import SwiftUtilsArgs
from swiftsim_utils.config import config_swift_utils, load_swift_config


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the swiftsim-utils CLI tool.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    # Parse the command-line arguments
    args = SwiftUtilsArgs(argv).args

    # Load the config, we may need it later
    config = load_swift_config()

    # Run the appropriate command based on the mode
    if args.mode == "config":
        config_swift_utils()
    elif args.mode == "version":
        print(f"SWIFT-utils version: {args.version}")
