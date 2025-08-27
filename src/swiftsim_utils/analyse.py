"""A module containing tools for analysing SWIFT runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyse_timestep_files(
    files: list[str],
    labels: list[str],
    plot_time: bool = True,
    output_path: str | None = None,
    prefix: str = None,
    show_plot: bool = True,
) -> None:
    """Plot the timestep files of one or more SWIFT runs.

    Args:
        files: List of file paths to the timestep files.
        labels: List of labels for the runs.
        plot_time: Whether to plot against time or scale factor. If True, plot
            against time, otherwise plot against scale factor.
        output_path: Optional path to save the plot. If None, the plot is saved
            to the current directory.
        prefix: Optional prefix to add to the output filename if saving.
            If empty, defaults to 'timestep_analysis.png'.
        show_plot: Whether to display the plot.

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
    y = [np.cumsum(d[:, wall_clock_index]) for d in data]
    deadtime = [np.cumsum(d[:, deadtime_index]) for d in data]

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Colors for the plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

    # Main plot - absolute times
    for i, (xi, yi, dt, label, color) in enumerate(
        zip(x, y, deadtime, labels, colors)
    ):
        # Plot wall clock time (solid lines)
        ax1.plot(
            xi,
            yi,
            "-",
            color=color,
            label=f"{label} (wall clock)",
            linewidth=2,
        )
        # Plot dead time (dashed lines with alpha) - make more visible
        ax1.plot(xi, dt, "--", color=color, alpha=0.6, linewidth=2)

    # Set labels and title for main plot
    x_label = "Time [Internal Units]" if plot_time else "Scale factor"
    ax1.set_ylabel("Wallclock Time [s]")
    # Only show legend for wall clock lines (solid lines only)
    handles, legend_labels = ax1.get_legend_handles_labels()
    wall_clock_handles = [
        h for h, l in zip(handles, legend_labels) if "(wall clock)" in l
    ]
    wall_clock_labels = [l for l in legend_labels if "(wall clock)" in l]
    ax1.legend(
        wall_clock_handles,
        wall_clock_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax1.grid(True, alpha=0.3)

    # Deadtime percentage plot
    for i, (xi, yi, dt, label, color) in enumerate(
        zip(x, y, deadtime, labels, colors)
    ):
        # Calculate deadtime percentage: (deadtime / total_time) * 100
        deadtime_percentage = (dt / yi) * 100

        # Plot deadtime percentage
        ax2.plot(
            xi,
            deadtime_percentage,
            "-",
            color=color,
            label=f"{label}",
            linewidth=2,
        )

    # Set labels and formatting for deadtime percentage plot
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Deadtime [%]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_ylim(0, None)  # Start y-axis at 0 for percentage

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for legends

    # Create the output path
    path = None
    if output_path is not None:
        path = Path(output_path)
    else:
        path = Path.cwd()

    # Ensure the output directory exists and is a directory
    if not path.is_dir():
        raise ValueError(f"Output path {path} is not a directory.")
    path.mkdir(parents=True, exist_ok=True)

    # Create the output filename
    filename = f"{prefix + '_' if prefix else ''}timestep_analysis.png"
    output_file = path / filename

    # Save the figure if an output path is provided
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    plt.close()
