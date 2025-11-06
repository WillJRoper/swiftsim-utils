"""A module containing tools for permanently configuring SWIFT-utils."""

import subprocess
from dataclasses import (  # asdict may be used elsewhere
    asdict,
    dataclass,
)
from functools import lru_cache
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from ruamel.yaml import YAML

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
        branch: Branch name for SWIFT repository.
        softening_coeff: Softening coefficient in units of mean separation.
        softening_pivot_z: Maximal softening pivot redshift.
        parameter_file: Path to parameter file for cosmology.

        Cosmology parameters:
        h: Reduced Hubble constant.
        a_begin: Initial scale-factor of the simulation.
        a_end: Final scale factor of the simulation.
        Omega_m: Matter density parameter.
        Omega_lambda: Dark-energy density parameter.
        Omega_b: Baryon density parameter.
        Omega_r: Radiation density parameter.
        w_0: Dark-energy equation-of-state parameter at z=0.
        w_a: Dark-energy equation-of-state time evolution parameter.
        T_nu_0: Present-day neutrino temperature in internal units.
        N_ur: Number of ultra-relativistic species.
        N_nu: Integer number of massive neutrinos.
        M_nu_eV: Comma-separated list of neutrino masses in eV.
        deg_nu: Comma-separated list of neutrino degeneracies.
    """

    swiftsim_dir: Path | None
    data_dir: Path | None
    branch: str = "master"
    softening_coeff: float = 0.04
    softening_pivot_z: float = 2.7
    parameter_file: Optional[Path] = None

    # Cosmology parameters
    h: float = 0.6777
    a_begin: float = 0.0078125
    a_end: float = 1.0
    Omega_m: float = 0.307
    Omega_lambda: float = 0.693
    Omega_b: float = 0.0482519
    Omega_r: float = 0.0
    w_0: float = -1.0
    w_a: float = 0.0
    T_nu_0: float = 1.9514
    N_ur: float = 1.0196
    N_nu: int = 2
    M_nu_eV: str = "0.05, 0.01"
    deg_nu: str = "1.0, 1.0"


def load_cosmology_from_parameter_file(parameter_file_path: Path) -> dict:
    """Load cosmology parameters from a SWIFT parameter file.

    Args:
        parameter_file_path: Path to the SWIFT parameter YAML file.

    Returns:
        dict: Loaded cosmology parameters with defaults for missing values.

    Raises:
        FileNotFoundError: If the parameter file doesn't exist.
        ValueError: If the parameter file is malformed or missing cosmology
            section.
    """
    if not parameter_file_path.exists():
        raise FileNotFoundError(
            f"Parameter file not found: {parameter_file_path}"
        )

    # Read file and clean tabs
    with open(parameter_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Replace tabs with spaces (same approach as params.py)
    cleaned_text = raw_text.expandtabs(4)

    yaml = YAML()
    yaml.preserve_quotes = True
    params = yaml.load(cleaned_text)

    if params is None or "Cosmology" not in params:
        return {}  # Return empty dict if no cosmology section

    cosmo_params = params["Cosmology"]

    # Return dictionary with values from file, falling back to defaults
    # handled by SWIFTCLIProfile
    return {
        "h": cosmo_params.get("h", 0.6777),
        "a_begin": cosmo_params.get("a_begin", 0.0078125),
        "a_end": cosmo_params.get("a_end", 1.0),
        "Omega_m": cosmo_params.get("Omega_m", 0.307),
        "Omega_lambda": cosmo_params.get("Omega_lambda", 0.693),
        "Omega_b": cosmo_params.get("Omega_b", 0.0482519),
        "Omega_r": cosmo_params.get("Omega_r", 0.0),
        "w_0": cosmo_params.get("w_0", -1.0),
        "w_a": cosmo_params.get("w_a", 0.0),
        "T_nu_0": cosmo_params.get("T_nu_0", 1.9514),
        "N_ur": cosmo_params.get("N_ur", 1.0196),
        "N_nu": cosmo_params.get("N_nu", 2),
        "M_nu_eV": cosmo_params.get("M_nu_eV", "0.05, 0.01"),
        "deg_nu": cosmo_params.get("deg_nu", "1.0, 1.0"),
    }


def get_default_parameter_file(swiftsim_dir: Path) -> Path:
    """Get the default parameter file path.

    Args:
        swiftsim_dir: Path to the SWIFTSim repository.

    Returns:
        Path: Path to the default parameter_example.yml file.
    """
    return swiftsim_dir / "examples" / "parameter_example.yml"


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
    """Enter accepts current completion if menu is open, otherwise submits."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None and buf.complete_state:
        comp = buf.complete_state.current_completion
        if comp is not None:
            buf.apply_completion(comp)
    else:
        buf.validate_and_handle()


@KB.add("tab")
def _(event):
    """Tab opens/cycles completions, if the current prompt has a completer."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None:
        if buf.complete_state:
            buf.complete_next()
        else:
            buf.start_completion(select_first=True)


def get_cli_profiles(
    default_swift: str | None = None,
    default_data: str | None = None,
    default_branch: str = "master",
    default_softening_coeff: float = 0.04,
    default_softening_pivot_z: float = 2.7,
    default_parameter_file: str | None = None,
) -> SWIFTCLIProfile:
    """Interactively collect profile values for SWIFT-utils.

    Path prompts (first two) have completion + validation. While a completion
    menu is visible on those prompts:
      - Enter applies the highlighted completion and continues editing.
      - Enter submits only when the completion menu is not open.
    Other prompts behave normally and submit on Enter.
    """
    # Ensure all inputs are either str or None
    default_swift = (
        default_swift if default_swift is None else str(default_swift)
    )
    default_data = default_data if default_data is None else str(default_data)
    default_branch = str(default_branch)
    default_softening_coeff_str = str(default_softening_coeff)
    default_softening_pivot_z_str = str(default_softening_pivot_z)
    default_parameter_file = (
        default_parameter_file
        if default_parameter_file is None
        else str(default_parameter_file)
    )

    # Path completer for directory paths
    path_completer = PathCompleter(expanduser=True, only_directories=True)
    # File path completer for parameter files
    file_completer = PathCompleter(expanduser=True)

    # Separate sessions so completers/validators do not leak.
    path_session: PromptSession[str] = PromptSession(
        style=PTK_STYLE,
        key_bindings=KB,
        complete_while_typing=False,
        reserve_space_for_menu=8,
    )
    text_session: PromptSession[str] = PromptSession(
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
        default=default_branch,
    ).strip()

    # Convert to absolute path to determine default parameter file
    swift_repo_path = Path(swift_repo).expanduser().resolve()
    default_param_path = get_default_parameter_file(swift_repo_path)

    parameter_file = path_session.prompt(
        [("class:prompt", "Parameter file: ")],
        default=(default_parameter_file or str(default_param_path)),
        completer=file_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    softening_coeff = text_session.prompt(
        [
            (
                "class:prompt",
                "Softening (in units of mean separation): ",
            )
        ],
        default=default_softening_coeff_str,
    ).strip()

    softening_pivot_z = text_session.prompt(
        [
            (
                "class:prompt",
                "Maximal softening pivot redshift: ",
            )
        ],
        default=default_softening_pivot_z_str,
    ).strip()

    # Convert to absolute paths
    swift_repo_path = Path(swift_repo).expanduser().resolve()
    data_dir_path = Path(data_dir).expanduser().resolve()
    param_path = Path(parameter_file).expanduser().resolve()

    # Load cosmology parameters from parameter file
    cosmology_params = load_cosmology_from_parameter_file(param_path)
    print(f"Loaded cosmology parameters from {param_path}")

    return SWIFTCLIProfile(
        swiftsim_dir=swift_repo_path,
        data_dir=data_dir_path,
        branch=swift_branch,
        softening_coeff=float(softening_coeff),
        softening_pivot_z=float(softening_pivot_z),
        parameter_file=param_path,
        **cosmology_params,  # Unpack cosmology parameters as individual fields
    )


@lru_cache(maxsize=None)
def _load_swift_profile(key=None) -> SWIFTCLIProfile:
    """Load the SWIFT-utils profile from the profile file.

    Returns:
        SWIFTCLIProfile: The loaded profile.
    """
    # Return a dummy if the profile file does not yet exist
    if not PROFILE_FILE.exists():
        return SWIFTCLIProfile(
            swiftsim_dir=None,
            data_dir=None,
            branch="master",
            softening_coeff=0.04,
            softening_pivot_z=2.7,
            parameter_file=None,
        )

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(PROFILE_FILE, "r") as f:
        profile_data = yaml.load(f)

    # If we don't have a profile yet return an empty one
    if profile_data is None:
        return SWIFTCLIProfile(
            swiftsim_dir=None,
            data_dir=None,
            branch="master",
            softening_coeff=0.04,
            softening_pivot_z=2.7,
            parameter_file=None,
        )

    # If key is None return the current profile
    if key is None:
        profile_data = profile_data["Current"]
    else:
        profile_data = profile_data.get(key, {})

    # Load parameter file path (with backward compatibility)
    param_file = (
        profile_data.get("parameter_file")
        or profile_data.get("template_parameter_file")
        or profile_data.get("template_params")
    )
    param_file_path = Path(param_file) if param_file else None

    return SWIFTCLIProfile(
        swiftsim_dir=Path(profile_data["swiftsim_dir"]),
        data_dir=Path(profile_data["data_dir"]),
        branch=profile_data.get("branch", "master"),
        softening_coeff=float(profile_data.get("softening_coeff", 0.04)),
        softening_pivot_z=float(profile_data.get("softening_pivot_z", 2.7)),
        parameter_file=param_file_path,
        # Load cosmology parameters directly as individual fields
        h=profile_data.get("h", 0.6777),
        a_begin=profile_data.get("a_begin", 0.0078125),
        a_end=profile_data.get("a_end", 1.0),
        Omega_m=profile_data.get("Omega_m", 0.307),
        Omega_lambda=profile_data.get("Omega_lambda", 0.693),
        Omega_b=profile_data.get("Omega_b", 0.0482519),
        Omega_r=profile_data.get("Omega_r", 0.0),
        w_0=profile_data.get("w_0", -1.0),
        w_a=profile_data.get("w_a", 0.0),
        T_nu_0=profile_data.get("T_nu_0", 1.9514),
        N_ur=profile_data.get("N_ur", 1.0196),
        N_nu=profile_data.get("N_nu", 2),
        M_nu_eV=profile_data.get("M_nu_eV", "0.05, 0.01"),
        deg_nu=profile_data.get("deg_nu", "1.0, 1.0"),
    )


def _load_all_profiles() -> dict[str, dict]:
    """Load all profiles from the SWIFT-utils profile file.

    Returns:
        dict[str, dict]: A dictionary of all profiles in the profile
            file.
    """
    # Load existing profile handling edge cases
    if PROFILE_FILE.exists():
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(PROFILE_FILE, "r") as f:
            all_profile_data = yaml.load(f)
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
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(PROFILE_FILE, "w") as f:
        yaml.dump(all_profile_data, f)

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


def get_current_git_branch(swift_dir: Path) -> str | None:
    """Get the current git branch from a SWIFT repository.

    Args:
        swift_dir: Path to the SWIFT repository.

    Returns:
        str | None: The current branch name, or None if unable to determine.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=swift_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,  # Prevent hanging on slow/stuck operations
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        # Git operation took too long
        return None
    except subprocess.CalledProcessError:
        # Git command failed (not a repo, detached HEAD, etc.)
        return None
    except FileNotFoundError:
        # Git executable not found
        return None
    except Exception:
        # Unexpected error
        return None


def sync_profile_with_repo() -> None:
    """Sync the current profile's branch with the actual git repository.

    This updates the profile if the git branch has changed outside of
    swift-cli.
    """
    profile = load_swift_profile()

    # Only sync if we have a swift directory
    if profile.swiftsim_dir is None:
        return

    # Get the actual current branch
    actual_branch = get_current_git_branch(profile.swiftsim_dir)

    # If we got a branch and it's different from profile, update
    if actual_branch and actual_branch != profile.branch:
        update_current_profile_value("branch", actual_branch)
