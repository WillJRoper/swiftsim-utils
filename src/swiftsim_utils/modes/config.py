"""A module containing tools for configuring SWIFT."""

import argparse
from pathlib import Path

from swiftsim_utils.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for configuring SWIFT itself."""
    # Show the configuration options (equivalent to running
    # `./configure --help`).
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show the available configuration options.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the config mode."""
    # Are we just showing the config options?
    if args.show:
        show_config_options(args.swift_dir)
    else:
        config_swiftsim(
            opts=" ".join(args.options),
            swift_dir=args.swift_dir,
        )


def config_swiftsim(opts: str, swift_dir: Path | None = None) -> None:
    """Configure SWIFT itself.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the SWIFT configuration script with the provided options.

    Args:
        opts: The options to pass to the SWIFT configuration script.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    _run_command_in_swift_dir(f"./configure {opts}", swift_dir)


def show_config_options(swift_dir: Path | None = None) -> None:
    """Show the configuration options for SWIFT.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the SWIFT configuration script with the --help option.

    Args:
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    _run_command_in_swift_dir("./configure --help", swift_dir)
