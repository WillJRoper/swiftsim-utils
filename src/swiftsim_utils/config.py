"""A module containing tools for permenantly configuring SWIFT-utils."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import ValidationError, Validator


@dataclass(slots=True)
class SwiftCLIConfig:
    """Configuration for SWIFT-utils.

    Attributes:
        swiftsim_dir: Path to the SWIFTSim repository.
        data_dir: Path to the directory containing additional data files.
    """

    swiftsim_dir: Path
    data_dir: Path


class PathExistsValidator(Validator):
    """Validator to check if a given path exists."""

    def validate(self, document):
        """Validate that the path exists."""
        p = Path(document.text).expanduser()
        if not p.exists():
            raise ValidationError(
                message="Path does not exist.",
                cursor_position=len(document.text),
            )


def get_cli_configuration(
    default_swift: str | None = None,
    default_data: str | None = None,
) -> SwiftCLIConfig:
    """Interactively collect config values for SWIFT-utils.

    Args:
        default_swift: Optional default SWIFTSim directory.
        default_data: Optional default data directory.

    Returns:
        SwiftCLIConfig: Collected configuration.
    """
    path_completer = PathCompleter(expanduser=True, only_directories=True)

    swift_repo = prompt(
        "SWIFTSim directory: ",
        default=(default_swift or ""),
        completer=path_completer,
        validator=PathExistsValidator(),
    ).strip()

    data_dir = prompt(
        "Extra data directory: ",
        default=(default_data or swift_repo),
        completer=path_completer,
        validator=PathExistsValidator(),
    ).strip()

    # Convert paths to absolute paths (relative paths are useless in the
    # configuration file if we then run elsewhere)
    swift_repo = Path(swift_repo).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()

    return SwiftCLIConfig(
        Path(swift_repo).expanduser(), Path(data_dir).expanduser()
    )


def config_swift_utils() -> None:
    """Configure SWIFT-utils by collecting user input."""
    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Ensure the directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # If the config file already exists, load it to use as defaults
    if config_file.exists():
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:  # Handle case where the file is empty
            config_data = {}
        default_swift = config_data.get("swiftsim_dir", None)
        default_data = config_data.get("data_dir", None)
    else:
        default_swift = None
        default_data = None

    # Collect configuration interactively
    config = get_cli_configuration(
        default_swift=default_swift,
        default_data=default_data,
    )

    # Convert the dataclass to a dictionary
    data = asdict(config)

    # Convert all Paths to strings for YAML serialization
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Write the configuration to the YAML file
    with open(config_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("Configuration saved to", config_file)


def _load_swift_config() -> SwiftCLIConfig:
    """Load the SWIFT-utils configuration from the config file.

    Returns:
        SwiftCLIConfig: The loaded configuration.
    """
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # If we don't have a config yet return an empty one
    if config_data is None:
        return SwiftCLIConfig(None, None)

    return SwiftCLIConfig(
        Path(config_data["swiftsim_dir"]),
        Path(config_data["data_dir"]),
    )


@lru_cache(maxsize=1)
def load_swift_config() -> SwiftCLIConfig:
    """Load the SWIFT-utils configuration.

    This function caches the result to avoid reloading the configuration
    multiple times.

    Returns:
        SwiftCLIConfig: The loaded configuration.
    """
    return _load_swift_config()
