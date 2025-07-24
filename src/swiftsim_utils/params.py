"""A module containing the parameter reader and parameter holder."""

from functools import lru_cache
from pathlib import Path

import yaml


def _parse_parameters(param_file: Path) -> dict:
    """Parse parameters from a YAML file.

    Args:
        param_file: Path to the parameter file.

    Returns:
        dict: Parsed parameters.
    """
    with open(param_file, "r") as file:
        params = yaml.safe_load(file)
    return params


@lru_cache(maxsize=1)
def load_parameters(param_file: Path | None = None) -> dict:
    """Load parameters from a YAML file.

    Args:
        param_file: Path to the parameter file.

    Returns:
        dict: Loaded parameters.
    """
    # If no parameter file is provided, just exit with an empty dict
    if param_file is None:
        return {}

    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    return _parse_parameters(param_file)
