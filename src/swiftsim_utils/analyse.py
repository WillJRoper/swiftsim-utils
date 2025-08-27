"""A module containing tools for analysing SWIFT runs."""

import numpy as np


def _analyse_timestep_files(
    files: list[str],
    labels: list[str],
    plot_time: bool = True,
) -> None:
    """Plot the timestep files of one or more SWIFT runs.

    Args:
        files: List of file paths to the timestep files.
        labels: List of labels for the runs.
        plot_time: Whether to plot against time or scale factor. If True, plot
            against time, otherwise plot against scale factor.

    Raises:
        ValueError: If the number of files and labels do not match.
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    # Load the data from the files
    data = [np.loadtxt(f, skiprows=13) for f in files]

    # Are we plotting against time or scale factor?
    time_index = 1 if plot_time else 2
    wall_clock_index = 12
    deadtime_index = -1

    # Extract the x and y columns
    x = [d[:, time_index] for d in data]
    y = np.cumsum([d[:, wall_clock_index] for d in data])
    deadtime = np.cumsum([d[:, deadtime_index] for d in data])
