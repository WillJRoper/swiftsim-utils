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

from swiftsim_cli.params import load_parameters

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
        branch: The branch of the SWIFTSim repository to use.
        template_params: The path to a template parameter file, if None
            defaults to swiftsim_dir/examples/parameter_example.yml.
        h: The Hubble parameter (optional).
        Omega_cdm: The cold dark matter density parameter (optional).
        Omega_b: The baryon density parameter (optional).
        Omega_m: The total matter density parameter (optional).
        Omega_Lambda: The dark energy density parameter (optional).
        softening_coeff: The softening length in units of the mean
            inter-particle separation.
        softening_pivot_z: The pivot redshift for the softening length.
    """

    swiftsim_dir: Path
    data_dir: Path
    branch: str = "master"
    template_params: Path = None
    h: float = None
    Omega_cdm: float = None
    Omega_b: float = None
    Omega_m: float = None
    Omega_Lambda: float = None
    softening_coeff: float = 0.04
    softening_pivot_z: float = 2.7

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure paths are Path objects
        if self.swiftsim_dir is not None:
            self.swiftsim_dir = Path(self.swiftsim_dir)
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)
        if self.template_params is not None:
            self.template_params = Path(self.template_params)

        # If the template_param is not set, default
        # to swiftsim_dir/examples/parameter_example.yml
        if (
            self.template_params is None or str(self.template_params) == "."
        ) and self.swiftsim_dir is not None:
            self.template_params = (
                self.swiftsim_dir / "examples" / "parameter_example.yml"
            )

        # Validate that the template_params file exists
        if (
            self.template_params is not None
            and not self.template_params.exists()
        ):
            raise FileNotFoundError(
                f"Template parameter file {self.template_params}"
                " does not exist."
            )


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
    default_softening_coeff = str(default_softening_coeff)
    default_softening_pivot_z = str(default_softening_pivot_z)

    # Path completer for directory paths
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

    print("\nFirst we need to collect some location information...\n")

    # The SWIFT repo location
    swift_repo = path_session.prompt(
        [("class:prompt", "SWIFTSim directory: ")],
        default=(default_swift or ""),
        completer=path_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    # The directory where SWIFT will put extra data dependencies
    data_dir = path_session.prompt(
        [("class:prompt", "SWIFT data directory: ")],
        default=(default_data or swift_repo),
        completer=path_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    # The SWIFTSim branch to use
    swift_branch = text_session.prompt(
        [("class:prompt", "Which SWIFT git branch?: ")],
        default=default_branch,
    ).strip()

    print(
        "\nFor some operations we'll need a template SWIFT parameter file. "
        "This will be used to extract default parameters and generate "
        "parameter files for new simulations.\n"
        "By default this will use the complete example parameter file that "
        "comes with SWIFT.\n"
    )

    # An optional template parameter file path
    template_params = path_session.prompt(
        [
            (
                "class:prompt",
                "Template parameter file: ",
            )
        ],
        default=f"{swift_repo}/examples/parameter_example.yml",
        completer=path_completer,
        validator=PathExistsValidator(),
        validate_while_typing=False,
    ).strip()

    # If the user entered an empty string, set to None
    if template_params == "":
        template_params = None

    print(
        "\nNext you can optionally customise the default cosmology "
        "(by default these are taken from the template parameter file)\n"
    )

    # Get the cosmology parameters we just loaded to use as defaults
    cosmo_params = load_parameters(template_params).get("Cosmology", {})

    # Take missing defaults from D3A from Flamingo (Schaye et al. 2023)
    if "h" not in cosmo_params:
        cosmo_params["h"] = 0.681
    if "Omega_cdm" not in cosmo_params:
        cosmo_params["Omega_cdm"] = 0.2574
    if "Omega_b" not in cosmo_params:
        cosmo_params["Omega_b"] = 0.0486
    if "Omega_m" not in cosmo_params:
        cosmo_params["Omega_m"] = 0.306
    if "Omega_Lambda" not in cosmo_params:
        cosmo_params["Omega_lambda"] = 0.694

    # The little-h (reduced Hubble parameter)
    h = text_session.prompt(
        [("class:prompt", "Default Hubble parameter: ")],
        default=str(cosmo_params.get("h", "")),
    ).strip()

    # Omega_cdm
    omega_cdm = text_session.prompt(
        [("class:prompt", "Omega_cdm: ")],
        default=str(cosmo_params.get("Omega_cdm", "")),
    ).strip()

    # Omega_b
    omega_b = text_session.prompt(
        [("class:prompt", "Omega_b: ")],
        default=str(cosmo_params.get("Omega_b", "")),
    ).strip()

    # Omega_m
    omega_m = text_session.prompt(
        [("class:prompt", "Omega_m (Omega_b + Omega_cdm): ")],
        default=str(cosmo_params.get("Omega_m", "")),
    ).strip()

    # Omega_Lambda
    omega_lambda = text_session.prompt(
        [("class:prompt", "Omega_Lambda: ")],
        default=str(cosmo_params.get("Omega_lambda", "")),
    ).strip()

    print(
        "\nFinally, we can customise the gravitational softening "
        "definitions...\n"
    )

    # The softening length in units of the mean inter-particle separation
    softening_coeff = text_session.prompt(
        [
            (
                "class:prompt",
                "Softening (in units of mean separation): ",
            )
        ],
        default=default_softening_coeff,
    ).strip()

    # The pivot redshift for the softening length
    softening_pivot_z = text_session.prompt(
        [
            (
                "class:prompt",
                "Maximal softening pivot redshift: ",
            )
        ],
        default=default_softening_pivot_z,
    ).strip()

    # Convert to absolute paths
    swift_repo = Path(swift_repo).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    if template_params is not None:
        template_params = Path(template_params).expanduser().resolve()

    return SWIFTCLIProfile(
        swift_repo,
        data_dir,
        swift_branch,
        template_params,
        float(h),
        float(omega_cdm),
        float(omega_b),
        float(omega_m),
        float(omega_lambda),
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
        return SWIFTCLIProfile(
            None,  # swiftsim_dir
            None,  # data_dir
            "master",  # branch
            None,  # template_params
            0.681,  # h from Flamingo (Schaye et al. 2023)
            0.2574,  # Omega_cdm from Flamingo (Schaye et al. 2023)
            0.0486,  # Omega_b from Flamingo (Schaye et al. 2023)
            0.306,  # Omega_m from Flamingo (Schaye et al. 2023)
            0.694,  # Omega_Lambda from Flamingo (Schaye et al. 2023)
            0.04,  # softening_coeff
            2.7,  # softening_pivot_z
        )

    with open(PROFILE_FILE, "r") as f:
        profile_data = yaml.safe_load(f)

    # If we don't have a profile yet return an empty one
    if profile_data is None:
        return SWIFTCLIProfile(
            None,
            None,
            "master",
            None,
            0.681,
            0.2574,
            0.0486,
            0.306,
            0.694,
            0.04,
            2.7,
        )

    # If key is None return the current profile
    if key is None:
        profile_data = profile_data["Current"]
    else:
        profile_data = profile_data.get(key, {})

    return SWIFTCLIProfile(
        Path(profile_data["swiftsim_dir"]),
        Path(profile_data["data_dir"]),
        profile_data.get("branch", "master"),
        Path(profile_data.get("template_params", "")),
        float(profile_data.get("h", 0.681)),
        float(profile_data.get("Omega_cdm", 0.2574)),
        float(profile_data.get("Omega_b", 0.0486)),
        float(profile_data.get("Omega_m", 0.306)),
        float(profile_data.get("Omega_Lambda", 0.694)),
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
