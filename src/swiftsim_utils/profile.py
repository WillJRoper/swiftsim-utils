"""A module containing tools for permenantly configuring SWIFT-utils."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import ValidationError, Validator


@dataclass()
class SWIFTCLIProfile:
    """Configuration for SWIFT-utils.

    Attributes:
        swiftsim_dir: Path to the SWIFTSim repository.
        data_dir: Path to the directory containing additional data files.
    """

    swiftsim_dir: Path
    data_dir: Path
    branch: str = "master"
    softening_coeff: float = 0.04
    softening_pivot_z: float = 2.7


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
) -> SWIFTCLIProfile:
    """Interactively collect config values for SWIFT-utils.

    Args:
        default_swift: Optional default SWIFTSim directory.
        default_data: Optional default data directory.

    Returns:
        SWIFTCLIProfile: Collected configuration.
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

    swift_branch = prompt(
        "SWIFTSim branch: ",
        default="master",
    ).strip()

    # Get the softening coefficient and pivot redshift, with a default values
    softening_coeff = prompt(
        "Softening coefficient in epsilon"
        " = x * mean_separation (default x=0.04): ",
        default="0.04",
    ).strip()
    softening_pivot_z = prompt(
        "Softening pivot redshift (used for calculate maximal softening"
        " lengths, default z=2.7): ",
        default="2.7",
    ).strip()

    # Convert paths to absolute paths (relative paths are useless in the
    # configuration file if we then run elsewhere)
    swift_repo = Path(swift_repo).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()

    return SWIFTCLIProfile(
        Path(swift_repo).expanduser(),
        Path(data_dir).expanduser(),
        swift_branch,
        float(softening_coeff),
        float(softening_pivot_z),
    )


@lru_cache(maxsize=None)
def _load_swift_config(key=None) -> SWIFTCLIProfile:
    """Load the SWIFT-utils configuration from the config file.

    Returns:
        SWIFTCLIProfile: The loaded configuration.
    """
    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Return a dummy if the config file does not yet exist
    if not config_file.exists():
        return SWIFTCLIProfile(None, None, 0.04, 2.7)

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # If we don't have a config yet return an empty one
    if config_data is None:
        return SWIFTCLIProfile(None, None, 0.04, 2.7)

    # If key is None return the current config
    if key is None:
        config_data = config_data["Current"]
    else:
        config_data = config_data.get(key, {})

    return SWIFTCLIProfile(
        Path(config_data["swiftsim_dir"]),
        Path(config_data["data_dir"]),
        config_data.get("branch", "master"),
        float(config_data.get("softening_coeff", 0.04)),
        float(config_data.get("softening_pivot_z", 2.7)),
    )


def _load_all_profiles() -> dict[str, dict]:
    """Load all profiles from the SWIFT-utils configuration file.

    Returns:
        dict[str, dict]: A dictionary of all profiles in the configuration
            file.
    """
    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Load existing config handling edge cases
    if config_file.exists():
        with open(config_file, "r") as f:
            all_config_data = yaml.safe_load(f)
        if all_config_data is None:  # Handle case where the file is empty
            all_config_data = {}
    else:
        all_config_data = {}

    return all_config_data


def load_swift_config() -> SWIFTCLIProfile:
    """Load the SWIFT-utils configuration.

    This function caches the result to avoid reloading the configuration
    multiple times.

    Returns:
        SWIFTCLIProfile: The loaded configuration.
    """
    return _load_swift_config()


def _save_swift_config(config: SWIFTCLIProfile, key: str = "Current") -> None:
    """Save the SWIFT-utils configuration to the config file.

    Args:
        config: The configuration to save.
        key: The key under which to save the configuration. Defaults to
             "Current".
    """
    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Ensure the directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    all_config_data = _load_all_profiles()

    # Set the contents under the given key
    all_config_data[key] = asdict(config)

    # Convert all Paths to strings for YAML serialization
    for k, v in all_config_data.items():
        for _k, _v in v.items():
            if isinstance(_v, Path):
                all_config_data[k][_k] = str(_v)

    # Write back to the config file
    with open(config_file, "w") as f:
        yaml.safe_dump(all_config_data, f)

    # Clear the cached config
    _load_swift_config.cache_clear()


def update_current_config_value(key: str, value: str | float | int) -> None:
    """Update a single value in the current configuration.

    Args:
        key: The key to update.
        value: The new value.
    """
    # Get the current config
    config = load_swift_config()

    # Update the value
    setattr(config, key, value)

    # Save the updated config
    _save_swift_config(config, "Current")
