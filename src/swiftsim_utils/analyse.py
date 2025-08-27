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
        # Plot dead time (dashed lines with alpha)
        ax1.plot(xi, dt, "--", color=color, alpha=0.6, linewidth=1.5)

    # Set labels and title for main plot
    x_label = "Time [Internal Units]" if plot_time else "Scale factor"
    ax1.set_ylabel("Wallclock Time [s]")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Relative comparison plot (letterbox style)
    if len(files) > 1:
        # Use first file as reference
        x_ref = x[0]
        y_ref = y[0]
        dt_ref = deadtime[0]

        for i in range(1, len(files)):
            # Interpolate reference data to match current x points
            y_ref_interp = np.interp(x[i], x_ref, y_ref)
            dt_ref_interp = np.interp(x[i], x_ref, dt_ref)

            # Calculate relative differences
            y_rel = (y[i] - y_ref_interp) / y_ref_interp * 100
            dt_rel = (deadtime[i] - dt_ref_interp) / dt_ref_interp * 100

            # Plot relative differences
            ax2.plot(
                x[i],
                y_rel,
                "-",
                color=colors[i],
                label=f"{labels[i]} vs {labels[0]}",
                linewidth=2,
            )
            ax2.plot(
                x[i], dt_rel, "--", color=colors[i], alpha=0.6, linewidth=1.5
            )

    # Set labels and formatting for relative plot
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Relative Difference [%]")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

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
