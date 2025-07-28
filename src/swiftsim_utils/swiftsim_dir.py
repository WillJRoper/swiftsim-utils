"""A module containing tools for managing the SWIFT simulation directory."""

import os
from pathlib import Path

from swiftsim_utils.config import load_swift_config


def get_swiftsim_dir(swift_dir: Path | None = None) -> Path:
    """Get the SWIFT simulation directory.

    This will attempt to read the SWIFT directory from the configuration file
    if one is not passed. It will then check if the directory exists, raising
    an error if it does not.

    Args:
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Load the SWIFT configuration
    config = load_swift_config()

    # If we haven't been given a SWIFT directory, use the one from the config
    if swift_dir is None:
        swift_dir = config.swiftsim_dir

    # If we still don't have a SWIFT directory, raise an error
    if swift_dir is None:
        raise ValueError(
            "SWIFT directory not passed (--swift-dir) and "
            "not found in the configuration file."
        )

    # Does it actually exist?
    if not swift_dir.exists():
        raise FileNotFoundError(f"SWIFT directory does not exist: {swift_dir}")

    return swift_dir


def _run_command_in_swift_dir(command: str, swift_dir: Path) -> None:
    """Run a command in the SWIFT directory.

    This will navigate to the SWIFT directory and run the provided command.

    Args:
        command: The command to run in the SWIFT directory.
        swift_dir: The path to the SWIFT directory.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Cache the current working directory
    original_cwd = os.getcwd()

    # Navigate to the SWIFT directory
    os.chdir(swift_dir)

    # Run the SWIFT configuration script with the provided options
    os.system(command)

    # Return to the original working directory
    os.chdir(original_cwd)


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
