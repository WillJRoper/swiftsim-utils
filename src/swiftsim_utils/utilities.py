"""A module containing generic utility functions for SWIFTSim-CLI."""

import os
from pathlib import Path


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


def create_output_path(
    output_path: str | None = None,
    prefix: str | None = None,
    base_filename: str = "output.png",
) -> Path:
    """Create and validate output path for saving files.

    Args:
        output_path: Optional path to save the file. If None, uses
           current directory.
        prefix: Optional prefix to add to the filename.
        base_filename: Base filename to use (default: "output.png").

    Returns:
        Path: Complete path to the output file.

    Raises:
        ValueError: If the output path is not a directory.
    """
    from pathlib import Path

    # Create the output path
    if output_path is not None:
        path = Path(output_path)
    else:
        path = Path.cwd()

    # Ensure the output directory exists and is a directory
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path {path} exists but is not a directory.")

    # Create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Create the output filename with optional prefix
    filename = f"{prefix + '_' if prefix else ''}{base_filename}"
    output_file = path / filename

    return output_file
