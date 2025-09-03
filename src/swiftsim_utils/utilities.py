"""A module containing generic utility functions for SWIFTSim-CLI."""

import os
from pathlib import Path

ascii_art = (
    r"    ______       _________________        _______   ____",
    r"   / ___/ |     / /  _/ ___/_  __/ ____  / ___/ /  /  _/",
    r"   \__ \| | /| / // // /_   / /   /___/ / /  / /   / /",
    r"  ___/ /| |/ |/ // // __/  / /         / /__/ /__ / /",
    r" /____/ |__/|__/___/_/    /_/         /____/____/___/",
)


def run_command_in_dir(command: str, directory: Path) -> None:
    """Run a command in a specified directory.

    This function changes the current working directory to the specified
    directory, runs the command, and then returns to the original directory.

    Args:
        command: The command to run.
        directory: The directory in which to run the command.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    # Cache the current working directory
    original_cwd = os.getcwd()

    # Try to change to the specified directory and run the command, make sure
    # we always return to the original directory
    try:
        os.chdir(directory)
        os.system(command)
    finally:
        os.chdir(original_cwd)


def make_directory(path: Path) -> None:
    """Create a directory if it does not exist.

    Args:
        path: The path to the directory to create.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileExistsError(f"Path {path} exists and is not a directory.")
