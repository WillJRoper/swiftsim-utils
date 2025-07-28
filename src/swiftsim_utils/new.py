"""A module containing the machinery for creating a new SWIFT run."""

import os
import warnings
from pathlib import Path

import h5py
from ruamel.yaml import YAML

from swiftsim_utils.config import load_swift_config
from swiftsim_utils.swiftsim_dir import get_swiftsim_dir

# Configure YAML for round-trip comment preservation and consistent formatting
yaml = YAML()
yaml.default_flow_style = False
yaml.indent(mapping=4, sequence=4, offset=2)
yaml.width = 80
yaml.allow_unicode = True


def derive_params_from_ics(
    inicond_file: Path,
    params: dict,
) -> None:
    """Derive parameters from the initial conditions file.

    Args:
        inicond_file: The initial conditions file to read.
        params: The parameter dictionary to update.

    Returns:
        dict: The updated parameters dictionary with derived values.
    """
    # Get the config
    config = load_swift_config()

    # Read the initial conditions file
    with h5py.File(inicond_file, "r") as hdf:
        # Get the box size from the initial conditions
        box_size = hdf["Header"].attrs["BoxSize"][:]

        # How many particles do we have?
        ngas = hdf["Header"].attrs["NumPart_Total"][:][0]
        ndm = hdf["Header"].attrs["NumPart_Total"][:][1]

        # Get the redshift from the initial conditions
        if "Redshift" in hdf["Header"].attrs:
            redshift = hdf["Header"].attrs["Redshift"]
        else:
            redshift = None

        # Derive a_begin from the redshift
        a_begin = 1 / (1 + redshift)

        # If we have a Units group them get the unit system
        units = {}
        if "Units" in hdf:
            for key, value in hdf["Units"].attrs.items():
                units[key] = value
        else:
            # Warn the user that they need to set the units manually
            warnings.warn(
                "No 'Units' group found in the initial conditions file. You "
                "will need to set the units manually in the parameter file.",
                UserWarning,
            )

        # Compute the mean separation of each particle type
        # (assuming a cubic box)
        mean_separation_dm = box_size[0] / (ndm ** (1 / 3))
        mean_separation_gas = (
            (box_size[0] / (ngas ** (1 / 3)))
            if ngas > 0
            else mean_separation_dm
        )  # if we have no gas in ics itll be generated from dm

        # Compute the softening lengths and their maximal values
        dm_soft = config.softening_coeff * mean_separation_dm
        gas_soft = config.softening_coeff * mean_separation_gas
        max_dm_soft = dm_soft / (1 + config.softening_pivot_z)
        max_gas_soft = gas_soft / (1 + config.softening_pivot_z)

    print("Setting softening lengths to:")
    print(
        f"  Dark Matter: {dm_soft:.3e} (max (@{config.softening_pivot_z}):"
        f" {max_dm_soft:.3e})"
    )
    print(
        f"  Baryons:     {gas_soft:.3e} (max (@{config.softening_pivot_z}):"
        f" {max_gas_soft:.3e})"
    )

    # Update the parameters dictionary with the derived values
    params["Gravity"]["comoving_DM_softening"] = dm_soft
    params["Gravity"]["max_physical_DM_softening"] = max_dm_soft
    params["Gravity"]["comoving_baryon_softening"] = gas_soft
    params["Gravity"]["max_physical_baryon_softening"] = max_gas_soft
    if redshift is not None:
        params["Cosmology"]["a_begin"] = a_begin
    if len(units) > 0:
        if "Unit mass in cgs (U_M)" in units:
            params["InternalUnitSystem"]["UnitMass_in_cgs"] = units[
                "Unit mass in cgs (U_M)"
            ]
        else:
            warnings.warn(
                "No 'Unit mass in cgs (U_M)' found in the initial"
                " conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit length in cgs (U_L)" in units:
            params["InternalUnitSystem"]["UnitLength_in_cgs"] = units[
                "Unit length in cgs (U_L)"
            ]
        else:
            warnings.warn(
                "No 'Unit length in cgs (U_L)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit time in cgs (U_t)" in units:
            params["InternalUnitSystem"]["UnitTime_in_cgs"] = units[
                "Unit time in cgs (U_t)"
            ]
        else:
            warnings.warn(
                "No 'Unit time in cgs (U_t)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit current in cgs (U_I)" in units:
            params["InternalUnitSystem"]["UnitCurrent_in_cgs"] = units[
                "Unit current in cgs (U_I)"
            ]
        else:
            warnings.warn(
                "No 'Unit current in cgs (U_I)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit temperature in cgs (U_T)" in units:
            params["InternalUnitSystem"]["UnitTemp_in_cgs"] = units[
                "Unit temperature in cgs (U_T)"
            ]
        else:
            warnings.warn(
                "No 'Unit temperature in cgs (U_T)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit velocity in cgs (U_V)" in units:
            params["InternalUnitSystem"]["UnitVelocity_in_cgs"] = units[
                "Unit velocity in cgs (U_V)"
            ]
        else:
            warnings.warn(
                "No 'Unit velocity in cgs (U_V)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )

    return params


def apply_overrides(params: dict, overide_params: dict | None = None) -> None:
    """Apply overrides to the parameters dictionary.

    Args:
        params: The parameter dictionary to update.
        overide_params: Optional dictionary of parameters to override in the
            parameter file. Keys should be in the format 'PARENTKEY:KEY=VALUE'.
    """
    # Loop over the override parameters
    for key, value in overide_params.items():
        # Split parent and child keys, if we have that structure, otherwise
        # something has gone wrong
        split_key = key.split(":")
        if len(split_key) > 2:
            raise ValueError(
                f"Invalid parameter key '{key}'. "
                "Keys should be in the format 'parent:child'."
            )
        elif len(split_key) == 2:
            parent, child = key.split(":")
        else:
            raise ValueError(
                f"Invalid parameter key '{key}'. "
                "Keys should be provided in the format: "
                "'PARENTKEY:KEY=VALUE'."
            )

        # If the parent key is not in the parameters, raise an error
        if parent not in params:
            raise ValueError(
                f"Parent key '{parent}' not found in parameters. "
                "Available keys: " + ", ".join(params.keys())
            )

        # Ensure the child key exists in the parent
        if child not in params[parent]:
            raise ValueError(
                f"Key '{child}' not found in parent '{parent}'. "
                "Available keys: " + ", ".join(params[parent].keys())
            )

        # Set the value in the parameters dictionary
        params[parent][child] = value


def make_new_parameter_file(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
) -> None:
    """Create a new parameter file for the SWIFT run.

    Args:
        output_dir: The directory where the new parameter file will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        overide_params: Optional dictionary of parameters to override in the
            parameter file.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Read the example parameter file into a CommentedMap (preserves comments)
    example_param_file = swift_dir / "examples" / "parameter_example.yml"
    with example_param_file.open("r") as f:
        raw = f.read()

    # Convert tabs to spaces to satisfy YAML spec (tabs are not allowed
    # in indentation)
    raw = raw.expandtabs(4)
    params = yaml.load(raw)

    # Update the parameters with the initial conditions file
    params["InitialConditions"]["file_name"] = str(inicond_file)

    # Set various directories
    params["Restarts"]["subdir"] = str(output_dir / "restart")
    params["Snapshots"]["subdir"] = str(output_dir / "snapshots")

    # Apply any overrides provided by the user
    if overide_params is not None:
        apply_overrides(params, overide_params)

    # Write the new parameters to the output directory
    output_param_file = output_dir / "params.yml"
    with output_param_file.open("w") as f:
        yaml.dump(params, f)

    return params


def make_new_run_dir(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    dmo: bool = False,
) -> None:
    """Create a new SWIFT run directory with a parameter file.

    Args:
        output_dir: The directory where the new run will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        dmo: If True, the parameters will be modified for a DMO run.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the new parameters file
    params = make_new_parameter_file(output_dir, inicond_file, swift_dir)

    # Make some directories we will use
    restart_dir = params["Restarts"]["subdir"]
    snapshots_dir = params["Snapshots"]["subdir"]
    os.makedirs(restart_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
