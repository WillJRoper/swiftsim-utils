"""A module containing tools for permenantly configuring SWIFT-utils."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import ValidationError, Validator

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


@dataclass()
class SwiftCLIConfig:
    """Configuration for SWIFT-utils.

    Attributes:
        swiftsim_dir: Path to the SWIFTSim repository.
        data_dir: Path to the directory containing additional data files.
    """

    swiftsim_dir: Path
    data_dir: Path
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

    return SwiftCLIConfig(
        Path(swift_repo).expanduser(),
        Path(data_dir).expanduser(),
        float(softening_coeff),
        float(softening_pivot_z),
    )


def _load_swift_config() -> SwiftCLIConfig:
    """Load the SWIFT-utils configuration from the config file.

    Returns:
        SwiftCLIConfig: The loaded configuration.
    """
    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Return a dummy if the config file does not yet exist
    if not config_file.exists():
        return SwiftCLIConfig(None, None, 0.04, 2.7)

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # If we don't have a config yet return an empty one
    if config_data is None:
        return SwiftCLIConfig(None, None, 0.04, 2.7)

    return SwiftCLIConfig(
        Path(config_data["swiftsim_dir"]),
        Path(config_data["data_dir"]),
        float(config_data.get("softening_coeff", 0.04)),
        float(config_data.get("softening_pivot_z", 2.7)),
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
