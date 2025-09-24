"""Profile mode for configuring SWIFTsim-CLI settings and profiles."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from swiftsim_utils.profile import (
    _load_all_profiles,
    _load_swift_profile,
    _save_swift_profile,
    get_cli_profiles,
)
from swiftsim_utils.utilities import ascii_art, create_ascii_table


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
        initial_profile_profile()
    elif args.clear:
        clear_swift_profile()
    elif args.override:
        save_current_profile_as_default()
    elif args.new is not None:
        new_profile(args.new)
    elif args.switch is not None:
        switch_profile(args.switch)
    if args.show:
        display_profile()
    if args.list:
        list_profiles()


def initial_profile_profile() -> None:
    """Configure SWIFT-utils by collecting user input."""
    # Print the ASCII art
    print("\nWelcome to SWIFTsim-CLI!\n")
    print("\n".join(ascii_art))
    print()
    print("Let's set up your profile...\n")
    print()

    # Define the path to the profile file
    profile_file = Path.home() / ".swiftsim-utils" / "profiles.yaml"

    # Ensure the directory exists
    profile_file.parent.mkdir(parents=True, exist_ok=True)

    # If the profile file already exists, load it to use as defaults
    if profile_file.exists():
        with open(profile_file, "r") as f:
            profile_data = yaml.safe_load(f)
        if profile_data is None:  # Handle case where the file is empty
            profile_data = {}
        default_swift = profile_data.get("swiftsim_dir", None)
        default_data = profile_data.get("data_dir", None)
    else:
        default_swift = None
        default_data = None

    # Collect profileuration interactively
    profiles = get_cli_profiles(
        default_swift=default_swift,
        default_data=default_data,
    )

    # Convert the dataclass to a dictionary
    data = asdict(profiles)

    # Convert all Paths to strings for YAML serialization
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Save the profileuration under a name and also store it as the current
    profile = {"Current": data, "Default": data}

    # Write the profileuration to the YAML file
    with open(profile_file, "w") as f:
        yaml.dump(profile, f, default_flow_style=False)

    print("profileuration saved to", profile_file)


def clear_swift_profile() -> None:
    """Clear the SWIFT-utils profileuration.

    This will delete the profileuration file if it exists.
    """
    # Make sure the user is sure they want to do this
    confirm = input(
        "Are you sure you want to clear all SWIFT-CLI profiles?"
        " This action cannot be undone. (y/N): "
    )
    if confirm.lower() != "y":
        print("Aborting.")
        return

    # Define the path to the profile file
    profile_file = Path.home() / ".swiftsim-utils" / "profile.yaml"

    # Delete the profile file if it exists
    if profile_file.exists():
        profile_file.unlink()

    print("All SWIFT-CLI profiles cleared.")
    _load_swift_profile.cache_clear()


def save_current_profile_as_default() -> None:
    """Save the current profileuration as the default profileuration."""
    profile = _load_swift_profile()
    _save_swift_profile(profile, "Default")


def new_profile(key: str) -> None:
    """Create a new profileuration with the given key.

    Args:
        key: The key under which to save the new profileuration.
    """
    # Ensure the key exists and is not already in the profile
    exiting_profiles = _load_all_profiles()
    if len(key) == 0:
        raise ValueError("Profile name cannot be empty.")
    if key in exiting_profiles:
        raise ValueError(f"Profile '{key}' already exists.")

    # Otherwise, get the new profile and save it
    profile = get_cli_profileuration()
    _save_swift_profile(profile, key)


def switch_profile(key: str) -> None:
    """Switch the current profileuration to the one with the given key.

    Args:
        key: The key of the profileuration to switch to.
    """
    # Make sure the key exists
    exiting_profiles = _load_all_profiles()
    if key not in exiting_profiles:
        raise ValueError(f"Profile '{key}' does not exist.")

    # Load the profile and save it as current
    profile = _load_swift_profile(key)
    _save_swift_profile(profile, "Current")

    # Report that we've switched and print the profile
    print("Switched to:")
    display_profile(print_header=False, title=f"PROFILE: {key}")


def display_profile(
    print_header: bool = True,
    title: str = "CURRENT PROFILE",
) -> None:
    """Display the current SWIFT-utils profile.

    Args:
        print_header: Whether to print the ASCII art header.
        title: Title to display above the profile table.
    """
    # Print a header
    if print_header:
        print()
        print("\n".join(ascii_art))
        print()
        print(" Current SWIFT-utils profile:\n")

    # Get the current profile
    profile = _load_swift_profile()

    # Print the profile values in a nice table format
    headers = ["Key", "Value"]
    rows = []
    for field in asdict(profile).keys():
        value = getattr(profile, field)
        rows.append([field, str(value)])

    print(create_ascii_table(headers, rows, title=title))


def list_profiles() -> None:
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
