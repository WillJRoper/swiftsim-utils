"""Profile mode for configuring SWIFTsim-CLI settings and profiles."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from swiftsim_utils.config import (
    _load_all_profiles,
    _load_swift_config,
    _save_swift_config,
    get_cli_configuration,
)
from swiftsim_utils.utilities import ascii_art


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the profile mode."""
    # initial profile
    parser.add_argument(
        "--init",
        "-i",
        action="store_true",
        help="Run the initial configuration to setup the default profile.",
        default=False,
    )

    # Clear
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Wipe all SWIFT-CLI profiles.",
        default=False,
    )

    # Save current as default
    parser.add_argument(
        "--override",
        "-o",
        action="store_true",
        help="Save the current profile as the default profile.",
        default=False,
    )

    # New profile with a given name
    parser.add_argument(
        "--new",
        "-n",
        type=str,
        help="Create a new profile with the given name.",
        default=None,
    )

    # Switch to a different profile
    parser.add_argument(
        "--switch",
        "-s",
        type=str,
        help="Switch to the profile with the given name.",
        default=None,
    )

    # Display the current profile
    parser.add_argument(
        "--show",
        "-S",
        action="store_true",
        help="Display the current profile.",
        default=False,
    )

    # List all profiles
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available profiles.",
        default=False,
    )


def run(args: argparse.Namespace) -> None:
    """Execute the init mode."""
    if args.init:
        initial_profile_config()
    elif args.clear:
        clear_swift_config()
    elif args.override:
        save_current_config_as_default()
    elif args.new is not None:
        new_config(args.new)
    elif args.switch is not None:
        switch_config(args.switch)
    if args.show:
        display_config()
    if args.list:
        list_configs()


def initial_profile_config() -> None:
    """Configure SWIFT-utils by collecting user input."""
    # Print the ASCII art
    print("\nWelcome to SWIFTsim-CLI!\n")
    print("\n".join(ascii_art))
    print()
    print("Let's set up your configuration...\n")
    print()

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

    # Save the configuration under a name and also store it as the current
    config = {"Current": data, "Default": data}

    # Write the configuration to the YAML file
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Configuration saved to", config_file)


def clear_swift_config() -> None:
    """Clear the SWIFT-utils configuration.

    This will delete the configuration file if it exists.
    """
    # Make sure the user is sure they want to do this
    confirm = input(
        "Are you sure you want to clear all SWIFT-CLI profiles?"
        " This action cannot be undone. (y/N): "
    )
    if confirm.lower() != "y":
        print("Aborting.")
        return

    # Define the path to the config file
    config_file = Path.home() / ".swiftsim-utils" / "config.yaml"

    # Delete the config file if it exists
    if config_file.exists():
        config_file.unlink()

    print("All SWIFT-CLI profiles cleared.")
    _load_swift_config.cache_clear()


def save_current_config_as_default() -> None:
    """Save the current configuration as the default configuration."""
    config = _load_swift_config()
    _save_swift_config(config, "Default")


def new_config(key: str) -> None:
    """Create a new configuration with the given key.

    Args:
        key: The key under which to save the new configuration.
    """
    # Ensure the key exists and is not already in the config
    exiting_profiles = _load_all_profiles()
    if len(key) == 0:
        raise ValueError("Profile name cannot be empty.")
    if key in exiting_profiles:
        raise ValueError(f"Profile '{key}' already exists.")

    # Otherwise, get the new config and save it
    config = get_cli_configuration()
    _save_swift_config(config, key)


def switch_config(key: str) -> None:
    """Switch the current configuration to the one with the given key.

    Args:
        key: The key of the configuration to switch to.
    """
    # Make sure the key exists
    exiting_profiles = _load_all_profiles()
    if key not in exiting_profiles:
        raise ValueError(f"Profile '{key}' does not exist.")

    # Load the config and save it as current
    config = _load_swift_config(key)
    _save_swift_config(config, "Current")


def display_config() -> None:
    """Display the current SWIFT-utils configuration."""
    # Print a header
    print()
    print("\n".join(ascii_art))
    print()
    print(" Current SWIFT-utils configuration:\n")

    # Get the current config
    config = _load_swift_config()

    # Print the config values in a nice table format
    head = f"{'Key':<20} | {'Value'}"
    print(head)
    print("-" * len(head))
    for field in asdict(config).keys():
        value = getattr(config, field)
        print(f"{field:<20} | {value}")
    print()


def list_configs() -> None:
    """List all available SWIFT-utils profiles."""
    # Print a header
    print(" Available SWIFT-utils profiles:\n")

    # Get all profiles
    profiles = _load_all_profiles()

    # Print the profile names
    if len(profiles) == 0:
        print("No profiles found.")
    else:
        for profile in profiles.keys():
            print(f" - {profile}")
    print()
