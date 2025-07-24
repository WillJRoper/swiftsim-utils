"""A module containing the parameter reader and parameter holder."""

from pathlib import Path

import yaml

PARAMS: dict | None = None


def _clean_yaml_text(text: str, spaces_per_tab: int = 4) -> str:
    """Replace all tab characters with spaces in the YAML text.

    Args:
        text:          Raw text from the YAML file.
        spaces_per_tab: Number of spaces to use for each tab.

    Returns:
        A new string where every tab character is replaced by the
        specified number of spaces.
    """
    return text.replace("	", " " * spaces_per_tab)


def _parse_parameters(param_file: Path) -> dict:
    """Parse parameters from a YAML file.

    Args:
        param_file: Path to the parameter YAML file.

    Returns:
        A dict of parsed parameters.

    Raises:
        IOError:   If the file cannot be read.
        ValueError: If YAML parsing fails, with context on filename.
    """
    # Read raw content using a context manager
    try:
        with open(param_file, "r", encoding="utf-8") as file:
            raw = file.read()
    except Exception as e:
        raise IOError(f"Could not read parameter file '{param_file}': {e}")

    # Clean leading tabs
    cleaned = _clean_yaml_text(raw)

    # Parse YAML safely
    try:
        return yaml.safe_load(cleaned) or {}
    except yaml.YAMLError as e:
        # Add context and re-raise
        raise ValueError(f"Error parsing YAML in '{param_file}': {e}") from e


def load_parameters(param_file: Path | None = None) -> dict:
    """Load and cache parameters from a YAML file.

    Args:
        param_file: Optional Path to the parameter YAML file.

    Returns:
        A dict of loaded parameters (cached after first load).

    Raises:
        FileNotFoundError: If the path is provided but does not exist.
        IOError:           If the file cannot be read.
        ValueError:        If parsing fails.
    """
    global PARAMS

    if PARAMS is not None:
        return PARAMS

    if param_file is None:
        return {}

    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    PARAMS = _parse_parameters(param_file)
    return PARAMS
