"""A module containing tools for permanently configuring SWIFT-utils."""

from dataclasses import asdict, dataclass  # asdict may be used elsewhere
from functools import lru_cache
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator

# Define the path to the profile file
PROFILE_FILE = Path.home() / ".swiftsim-utils" / "profiles.yaml"


@dataclass()
class SWIFTCLIProfile:
    """A configuration for SWIFT-utils.

    This contains all the information defining a SWIFT profile which can be
    used in SWIFT-CLI's various modes.

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


# ---------- prompt_toolkit styling and key bindings ----------

PTK_STYLE = Style.from_dict(
    {
        "completion-menu": "bg:#2b2e3b",
        "completion-menu.completion": "bg:#2b2e3b #c0caf5",
        "completion-menu.completion.current": "bg:#3b4252 #ffffff",
        "scrollbar.background": "bg:#3b4252",
        "scrollbar.button": "bg:#ffffff",
        "prompt": "bold",
    }
)

KB = KeyBindings()


@KB.add("enter")
def _(event):
    """Enter accepts highlighted completion if menu is open, otherwise submits."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None and buf.complete_state:
        comp = buf.complete_state.current_completion
        if comp is not None:
            buf.apply_completion(comp)
    else:
        buf.validate_and_handle()


@KB.add("tab")
def _(event):
    """Tab opens/cycles completions only if the current prompt has a completer."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None:
        if buf.complete_state:
            buf.complete_next()
        else:
            buf.start_completion(select_first=True)


def get_cli_profiles(
    default_swift: str | None = None,
    default_data: str | None = None,
) -> SWIFTCLIProfile:
    """Interactively collect profile values for SWIFT-utils.

    Path prompts (first two) have completion + validation. While a completion
    menu is visible on those prompts:
      - Enter applies the highlighted completion and continues editing.
      - Enter submits only when the completion menu is not open.
    Other prompts behave normally and submit on Enter.
    """
    path_completer = PathCompleter(expanduser=True, only_directories=True)

    # Separate sessions so completers/validators do not leak.
    path_session = PromptSession(
        style=PTK_STYLE,
        key_bindings=KB,
        complete_while_typing=False,
        reserve_space_for_menu=8,
    )
    text_session = PromptSession(
        style=PTK_STYLE,
        complete_while_typing=False,
    )

    # --- Path prompts with validation and completion ---
    swift_repo = path_session.prompt(
        [("class:prompt", "SWIFTSim directory: ")],
        default=(default_swift or ""),
        completer=path_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    data_dir = path_session.prompt(
        [("class:prompt", "Extra data directory: ")],
        # preserve fallback: use default_data if provided, otherwise swift_repo
        default=(default_data or swift_repo),
        completer=path_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    # --- Plain prompts (no completer) ---
    swift_branch = text_session.prompt(
        [("class:prompt", "SWIFTSim branch: ")],
        default="master",
    ).strip()

    softening_coeff = text_session.prompt(
        [
            (
                "class:prompt",
                "Softening coefficient in epsilon = x * mean_separation (default x=0.04): ",
            )
        ],
        default="0.04",
    ).strip()

    softening_pivot_z = text_session.prompt(
        [
            (
                "class:prompt",
                "Softening pivot redshift (used to calculate maximal softening lengths, default z=2.7): ",
            )
        ],
        default="2.7",
    ).strip()

    # Convert to absolute paths
    swift_repo = Path(swift_repo).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()

    return SWIFTCLIProfile(
        swift_repo,
        data_dir,
        swift_branch,
        float(softening_coeff),
        float(softening_pivot_z),
    )


@lru_cache(maxsize=None)
def _load_swift_profile(key=None) -> SWIFTCLIProfile:
    """Load the SWIFT-utils profile from the profile file.

    Returns:
        SWIFTCLIProfile: The loaded profile.
    """
    # Return a dummy if the profile file does not yet exist
    if not PROFILE_FILE.exists():
        return SWIFTCLIProfile(None, None, 0.04, 2.7)

    with open(PROFILE_FILE, "r") as f:
        profile_data = yaml.safe_load(f)

    # If we don't have a profile yet return an empty one
    if profile_data is None:
        return SWIFTCLIProfile(None, None, 0.04, 2.7)

    # If key is None return the current profile
    if key is None:
        profile_data = profile_data["Current"]
    else:
        profile_data = profile_data.get(key, {})

    return SWIFTCLIProfile(
        Path(profile_data["swiftsim_dir"]),
        Path(profile_data["data_dir"]),
        profile_data.get("branch", "master"),
        float(profile_data.get("softening_coeff", 0.04)),
        float(profile_data.get("softening_pivot_z", 2.7)),
    )


def _load_all_profiles() -> dict[str, dict]:
    """Load all profiles from the SWIFT-utils profile file.

    Returns:
        dict[str, dict]: A dictionary of all profiles in the profile
            file.
    """
    # Load existing profile handling edge cases
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, "r") as f:
            all_profile_data = yaml.safe_load(f)
        if all_profile_data is None:  # Handle case where the file is empty
            all_profile_data = {}
    else:
        all_profile_data = {}

    return all_profile_data


def load_swift_profile() -> SWIFTCLIProfile:
    """Load the SWIFT-utils profile.

    This function caches the result to avoid reloading the profile
    multiple times.

    Returns:
        SWIFTCLIProfile: The loaded profile.
    """
    return _load_swift_profile()


def _save_swift_profile(
    profile: SWIFTCLIProfile, key: str = "Current"
) -> None:
    """Save the SWIFT-utils profile to the profile file.

    Args:
        profile: The profile to save.
        key: The key under which to save the profile. Defaults to
             "Current".
    """
    # Load existing profile
    all_profile_data = _load_all_profiles()

    # Set the contents under the given key
    all_profile_data[key] = asdict(profile)

    # Convert all Paths to strings for YAML serialization
    for k, v in all_profile_data.items():
        for _k, _v in v.items():
            if isinstance(_v, Path):
                all_profile_data[k][_k] = str(_v)

    # Write back to the profile file
    with open(PROFILE_FILE, "w") as f:
        yaml.safe_dump(all_profile_data, f)

    # Clear the cached profile
    _load_swift_profile.cache_clear()


def update_current_profile_value(key: str, value: str | float | int) -> None:
    """Update a single value in the current profile.

    Args:
        key: The key to update.
        value: The new value.
    """
    # Get the current profile
    profile = load_swift_profile()

    # Update the value
    setattr(profile, key, value)

    # Save the updated profile
    _save_swift_profile(profile, "Current")
