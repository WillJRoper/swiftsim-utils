"""A module defining machinery related to cosmology in SWIFT."""

from functools import lru_cache

from astropy.cosmology import FlatLambdaCDM, z_at_value

from swiftsim_utils.params import load_parameters


def _get_cosmology() -> FlatLambdaCDM:
    """Get the cosmology parameters from the SWIFT configuration.

    Returns:
        FlatLambdaCDM: An astropy cosmology object with the parameters
        defined in the SWIFT configuration.
    """
    # Get the parameters
    params = load_parameters()

    # If we have no parameters we can't set up a cosmology
    if len(params) == 0:
        return None

    # Extract cosmology parameters
    cosmo_params = params["Cosmology"]

    # Create a FlatLambdaCDM cosmology object
    cosmo = FlatLambdaCDM(
        H0=cosmo_params["h"] * 100,
        Om0=cosmo_params["Omega_m"],
        Ob0=cosmo_params["Omega_b"],
    )

    return cosmo


@lru_cache(maxsize=1)
def get_cosmology() -> FlatLambdaCDM | None:
    """Get the cached cosmology object.

    If no parameters are defined, this will return None.

    Returns:
        FlatLambdaCDM: The cached cosmology object.
    """
    return _get_cosmology()


def convert_redshift_to_time(redshift: float) -> float:
    """Convert redshift to time in seconds.

    Args:
        redshift (float): The redshift value to convert.

    Returns:
        float: The time in seconds corresponding to the given redshift.
    """
    cosmo = get_cosmology()
    return cosmo.age(redshift)


def convert_time_to_redshift(time: float) -> float:
    """Convert time in seconds to redshift.

    Args:
        time (float): The time in seconds to convert.

    Returns:
        float: The redshift corresponding to the given time.
    """
    cosmo = get_cosmology()
    return z_at_value(cosmo.age, time)


def convert_scale_factor_to_redshift(scale_factor: float) -> float:
    """Convert scale factor to redshift.

    Args:
        scale_factor (float): The scale factor to convert.

    Returns:
        float: The redshift corresponding to the given scale factor.
    """
    return 1.0 / scale_factor - 1.0


def convert_redshift_to_scale_factor(redshift: float) -> float:
    """Convert redshift to scale factor.

    Args:
        redshift (float): The redshift value to convert.

    Returns:
        float: The scale factor corresponding to the given redshift.
    """
    return 1.0 / (1.0 + redshift)
