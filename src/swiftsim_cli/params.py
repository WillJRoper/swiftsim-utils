"""A module containing the parameter reader and parameter holder."""

from pathlib import Path

from ruamel.yaml import YAML

# Configure YAML for round-trip comment preservation and consistent formatting
yaml = YAML()
yaml.default_flow_style = False
yaml.indent(mapping=4, sequence=4, offset=2)
yaml.width = 80
yaml.allow_unicode = True


def _clean_yaml_text(text: str, spaces_per_tab: int = 4) -> str:
    """Replace all tab characters with spaces in the YAML text.

    Args:
        text:          Raw text from the YAML file.
        spaces_per_tab: Number of spaces to use for each tab.

    Returns:
        A new string where every tab character is replaced by the
        specified number of spaces.
    """
    return text.expandtabs(spaces_per_tab)


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
        with param_file.open("r", encoding="utf-8") as file:
            raw = file.read()
    except Exception as e:
        raise IOError(f"Could not read parameter file '{param_file}': {e}")

    # Clean leading tabs
    cleaned = _clean_yaml_text(raw)

    # Parse YAML safely
    try:
        return yaml.load(cleaned) or {}
    except Exception as e:
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
    # Avoid circular import
    from swiftsim_cli.profile import load_swift_profile

    # If we haven't been passed a path, get the parameter file path from the
    # profile
    if param_file is None:
        profile = load_swift_profile()
        param_file = getattr(profile, "template_params", None)
        if param_file is None:
            raise ValueError(
                "No parameter file specified and no default set in profile."
            )

    # Ensure we have a Path object
    if not isinstance(param_file, Path):
        param_file = Path(param_file)

    # Ensure the path exists
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    return _parse_parameters(param_file)
