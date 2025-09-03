"""Init mode for configuring SWIFTsim-CLI settings."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml
from swiftsim_utils.config import get_cli_configuration


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'init' mode."""
    # No other arguments, this will just walk the user through the config.
    pass


def run(args: argparse.Namespace) -> None:
    """Execute the init mode."""
    config_swift_utils()


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