"""Profile mode for configuring SWIFTsim-CLI settings and profiles."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from swiftsim_utils.profile import (
    PROFILE_FILE,
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

    # Edit the current profile
    parser.add_argument(
        "--edit",
        "-e",
        help="Edit a profile.",
        type=str,
        default=None,
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
    elif args.edit is not None:
        edit_profile(args.edit)
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

    # Ensure the directory exists
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # If the profile file already exists, load it to use as defaults
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, "r") as f:
            profile_data = yaml.safe_load(f)
        if profile_data is None:  # Handle case where the file is empty
            profile_data = {}
        default_swift = profile_data.get("swiftsim_dir", None)
        default_data = profile_data.get("data_dir", None)
    else:
        default_swift = None
        default_data = None

    # Collect profiles
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

    # Save the profile under a name and also store it as the current
    profile = {"Current": data, "Default": data}

    # Write the profile to the YAML file
    with open(PROFILE_FILE, "w") as f:
        yaml.dump(profile, f, default_flow_style=False)

    print("Profile saved to", PROFILE_FILE)


def clear_swift_profile() -> None:
    """Clear the SWIFT-utils profiles.

    This will delete the profiles file if it exists.
    """
    # Make sure the user is sure they want to do this
    confirm = input(
        "Are you sure you want to clear all SWIFT-CLI profiles?"
        " This action cannot be undone. (y/N): "
    )
    if confirm.lower() != "y":
        print("Aborting...")
        return

    # Delete the profile file if it exists
    if PROFILE_FILE.exists():
        PROFILE_FILE.unlink()

    print("All SWIFT-CLI profiles cleared.")
    _load_swift_profile.cache_clear()


def save_current_profile_as_default() -> None:
    """Save the current profile as the default profile."""
    profile = _load_swift_profile()
    _save_swift_profile(profile, "Default")


def new_profile(key: str) -> None:
    """Create a new profile with the given key.

    Args:
        key: The key under which to save the new profile.
    """
    # Ensure the key exists and is not already in the profile
    exiting_profiles = _load_all_profiles()
    if len(key) == 0:
        raise ValueError("Profile name cannot be empty.")
    if key in exiting_profiles:
        raise ValueError(f"Profile '{key}' already exists.")

    # Unpack the default profile to use as defaults
    default_profile = exiting_profiles.get("Default", {})
    default_swift = default_profile.get("swiftsim_dir", None)
    default_data = default_profile.get("data_dir", None)

    # Otherwise, get the new profile and save it
    profile = get_cli_profiles(
        default_swift=default_swift,
        default_data=default_data,
    )
    _save_swift_profile(profile, key)


def switch_profile(key: str) -> None:
    """Switch the current profile to the one with the given key.

    Args:
        key: The key of the profile to switch to.
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
            # Skip Current
            if profile == "Current":
                continue
            print(f" - {profile}")
    print()


def edit_profile(key: str) -> None:
    """Edit the a SWIFT-utils profile interactively.

    Args:
        key: The key of the profile to edit.
    """
    profile = _load_swift_profile(key)
    print(f"Editing profile '{key}'. Press Enter to keep existing values.\n")
    updated_profile = get_cli_profiles(
        default_swift=str(profile.swiftsim_dir)
        if profile.swiftsim_dir
        else None,
        default_data=str(profile.data_dir) if profile.data_dir else None,
        default_branch=profile.branch if profile.branch else "master",
        default_softening_coeff=profile.softening_coeff
        if profile.softening_coeff
        else 0.04,
        default_softening_pivot_z=profile.softening_pivot_z
        if profile.softening_pivot_z
        else 2.7,
    )
    _save_swift_profile(updated_profile, key=key)
    print("\nProfile updated successfully.")
