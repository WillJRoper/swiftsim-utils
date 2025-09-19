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
    output_dir: str | None = None,
) -> Path:
    """Create and validate output path for saving files.

    Args:
        output_path: Optional path to save the file. If None, uses
           current directory.
        prefix: Optional prefix to add to the filename.
        base_filename: Base filename to use (default: "output.png").
        output_dir: An optional directory to hold the outputs.

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
    if output_dir is not None:
        output_file = path / output_dir / filename
    else:
        output_file = path / filename

    return output_file


def create_ascii_table(headers, rows, title=None):
    """Create a formatted ASCII table.

    Args:
        headers: List of column headers
        rows: List of row data (each row is a list)
        title: Optional table title

    Returns:
        String containing the formatted table
    """
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Create separator line
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"

    # Build table
    table_lines = []

    # Title
    if title:
        total_width = len(separator)
        table_lines.append("+" + "=" * (total_width - 2) + "+")
        title_line = f"| {title:^{total_width - 4}} |"
        table_lines.append(title_line)

    table_lines.append(separator)

    # Headers
    header_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {str(header):^{col_widths[i] - 2}} |"
    table_lines.append(header_line)
    table_lines.append(separator)

    # Rows
    for row in rows:
        row_line = "|"
        for i, cell in enumerate(row):
            # Right-align numbers, left-align text
            cell_str = str(cell)
            if i == 0:  # First column (names) - left align
                row_line += f" {cell_str:<{col_widths[i] - 2}} |"
            else:  # Other columns (numbers) - right align
                row_line += f" {cell_str:>{col_widths[i] - 2}} |"
        table_lines.append(row_line)

    table_lines.append(separator)

    return "\n".join(table_lines)
