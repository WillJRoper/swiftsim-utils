"""Analyse mode for analysing SWIFT runs."""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from swiftsim_utils.utilities import create_output_path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'analyse' mode with subparsers."""
    # Create subparsers for different analysis types
    subparsers = parser.add_subparsers(
        dest="analysis_type",
        help="Type of analysis to perform",
        required=True,
    )

    # Timestep analysis subparser
    timestep_parser = subparsers.add_parser(
        "timesteps", help="Analyse timestep files"
    )

    timestep_parser.add_argument(
        "files",
        nargs="+",
        help="List of timestep files to analyse and produce a plot for.",
        type=Path,
    )

    timestep_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        required=True,
        help="List of labels for the runs being analysed.",
        type=str,
    )

    timestep_parser.add_argument(
        "--plot-time",
        action="store_true",
        help="Plot against time (default is scale factor).",
        default=False,
    )

    timestep_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    timestep_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    timestep_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    # Gravity checks analysis subparser
    gravity_parser = subparsers.add_parser(
        "gravity-checks", help="Analyse gravity check files"
    )

    gravity_parser.add_argument(
        "files",
        nargs="+",
        help="List of gravity check files to analyse (either exact "
        "or SWIFT files).",
        type=Path,
    )

    gravity_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        required=True,
        help="List of labels for the runs being analysed.",
        type=str,
    )

    gravity_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    gravity_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    gravity_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    gravity_parser.add_argument(
        "--min-error",
        type=float,
        help="Minimum error for binning (default: 1e-7).",
        default=1e-7,
    )

    gravity_parser.add_argument(
        "--max-error",
        type=float,
        help="Maximum error for binning (default: 3e-1).",
        default=3e-1,
    )

    gravity_parser.add_argument(
        "--num-bins",
        type=int,
        help="Number of bins for histogram (default: 64).",
        default=64,
    )

    # Gravity error map subparser
    error_map_parser = subparsers.add_parser(
        "gravity-error-map",
        help="Create hexbin error maps for gravity check files",
    )

    error_map_parser.add_argument(
        "files",
        nargs="+",
        help="List of gravity check files to analyse (either exact or"
        " SWIFT files).",
        type=Path,
    )

    error_map_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        required=True,
        help="List of labels for the runs being analysed.",
        type=str,
    )

    error_map_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    error_map_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    error_map_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    error_map_parser.add_argument(
        "--resolution",
        "--gridsize",
        "-r",
        type=int,
        help="Resolution (gridsize) for hexbin plot (default: 100).",
        default=100,
    )


def run_timestep(args: argparse.Namespace) -> None:
    """Execute the timestep analysis."""
    analyse_timestep_files(
        files=args.files,
        labels=args.labels,
        plot_time=args.plot_time,
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
    )


def run_gravity_checks(args: argparse.Namespace) -> None:
    """Execute the gravity checks analysis."""
    analyse_force_checks(
        files=args.files,
        labels=args.labels,
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        min_error=args.min_error,
        max_error=args.max_error,
        num_bins=args.num_bins,
    )


def run_gravity_error_maps(args: argparse.Namespace) -> None:
    """Execute the gravity error map analysis."""
    analyse_gravity_error_maps(
        files=args.files,
        labels=args.labels,
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        resolution=args.resolution,
    )


def run(args: argparse.Namespace) -> None:
    """Execute the analyse mode based on the selected subcommand."""
    if args.analysis_type == "timesteps":
        run_timestep(args)
    elif args.analysis_type == "gravity-checks":
        run_gravity_checks(args)
    elif args.analysis_type == "gravity-error-map":
        run_gravity_error_maps(args)
    else:
        raise ValueError(f"Unknown analysis type: {args.analysis_type}")


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
    print(prefix)
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    # Are we plotting against time or scale factor?
    time_index = 1 if plot_time else 2
    wall_clock_index = 12
    deadtime_index = -1

    # Loop over the lines in the file and extract the relevant data
    x = []
    y = []
    deadtime = []
    for file in files:
        xi, yi, dti = [], [], []
        with open(file, "r") as f:
            for line in f:
                # Ensure we aren't reading a comment line
                if line.startswith("#"):
                    continue

                # If the line doesn't start with an empty space, its not a
                # data line
                if not line[0].isspace():
                    continue

                # Split the line into parts
                parts = line.split()

                # Ensure we found 15 columns
                if len(parts) != 15:
                    continue

                # Ensure all 15 columns are numbers
                try:
                    [float(part) for part in parts]
                except ValueError:
                    print(
                        "Failed to parse line:",
                        line,
                        "in file:",
                        file,
                        "(this is probably fine)",
                    )
                    continue

                xi.append(float(parts[time_index]))
                yi.append(float(parts[wall_clock_index]))
                dti.append(float(parts[deadtime_index]))

        # Convert to numpy arrays and compute cumulative sums in hours
        x.append(np.array(xi))
        y.append(np.cumsum(np.array(yi)) / (1000 * 60 * 60))
        deadtime.append(np.cumsum(np.array(dti)) / (1000 * 60 * 60))

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

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
            linewidth=2,
        )
        # Plot dead time (dashed lines with alpha) - make more visible
        ax1.plot(xi, dt, "--", color=color, alpha=0.6, linewidth=2)

    # Set labels and title for main plot
    x_label = "Time [Internal Units]" if plot_time else "Scale factor"
    ax1.set_ylabel("Time [hrs]")

    # Create custom legend with black lines showing line styles
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Wallclock Time",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label="Dead Time",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="best")

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
    ax2.set_ylabel("Dead Time [%]")
    ax2.legend(loc="best")
    ax2.set_ylim(0, None)  # Start y-axis at 0 for percentage

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Create the output path
    output_file = create_output_path(
        output_path, prefix, "timestep_analysis.png"
    )

    # Save the figure if an output path is provided
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    plt.close()


def _find_counterpart_file(filepath: str) -> tuple[str, str]:
    """Find the counterpart file (exact <-> swift) for a given file."""
    file_path = Path(filepath)
    filename = file_path.name
    directory = file_path.parent

    # Check if this is an exact file
    if "exact" in filename:
        # This is an exact file, find the corresponding SWIFT file
        exact_file = filepath

        # Extract step number
        step_match = re.search(r"step(\d{4})", filename)
        if not step_match:
            raise ValueError(f"Could not extract step number from {filename}")
        step_num = step_match.group(1)

        # Look for corresponding SWIFT files with different orders
        swift_pattern = (
            directory / f"gravity_checks_swift_step{step_num}_order*.dat"
        )
        swift_files = list(glob.glob(str(swift_pattern)))

        if not swift_files:
            raise FileNotFoundError(
                f"No SWIFT files found for step {step_num}"
            )

        # Take the first one found
        swift_file = swift_files[0]

    else:
        # This should be a SWIFT file, find the corresponding exact file
        swift_file = filepath

        # Extract step number
        step_match = re.search(r"step(\d{4})", filename)
        if not step_match:
            raise ValueError(f"Could not extract step number from {filename}")
        step_num = step_match.group(1)

        # Look for exact file - try different patterns
        exact_patterns = [
            directory / f"gravity_checks_exact_step{step_num}.dat",
            directory / f"gravity_checks_exact_periodic_step{step_num}.dat",
        ]

        exact_file = None
        for pattern in exact_patterns:
            if pattern.exists():
                exact_file = str(pattern)
                break

        if exact_file is None:
            raise FileNotFoundError(f"No exact file found for step {step_num}")

    return exact_file, swift_file


def analyse_force_checks(
    files: list[str],
    labels: list[str],
    output_path: str | None = None,
    prefix: str = None,
    show_plot: bool = True,
    min_error: float = 1e-7,
    max_error: float = 3e-1,
    num_bins: int = 64,
) -> None:
    """Plot the force check analysis for one or more SWIFT runs.

    Args:
        files: List of file paths to either exact or SWIFT force files.
               The function will automatically find the counterpart files.
        labels: List of labels for the runs.
        output_path: Optional path to save the plot. If None, the plot is saved
            to the current directory.
        prefix: Optional prefix to add to the output filename if saving.
            If empty, defaults to 'gravity_checks.png'.
        show_plot: Whether to display the plot.
        min_error: Minimum error for binning (default: 1e-7).
        max_error: Maximum error for binning (default: 3e-1).
        num_bins: Number of bins for histogram (default: 64).

    Raises:
        ValueError: If the number of files and labels do not match.
        FileNotFoundError: If counterpart files cannot be found.
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    # Set up parameters for plotting
    params = {
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "font.size": 11,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": False,
        "figure.figsize": (12, 9),
        "figure.subplot.left": 0.06,
        "figure.subplot.right": 0.99,
        "figure.subplot.bottom": 0.06,
        "figure.subplot.top": 0.99,
        "figure.subplot.wspace": 0.14,
        "figure.subplot.hspace": 0.14,
        "lines.markersize": 6,
        "lines.linewidth": 3.0,
    }
    plt.rcParams.update(params)

    # Construct the bins
    bin_edges = np.linspace(
        np.log10(min_error), np.log10(max_error), num_bins + 1
    )
    bin_size = (np.log10(max_error) - np.log10(min_error)) / num_bins
    bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_edges = 10**bin_edges
    bins = 10**bins

    # Colors for the plots
    cols = ["#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677"]

    # Create the figure with subplots
    fig = plt.figure()
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)

    # Process each file and find its counterpart
    for i, (input_file, label) in enumerate(zip(files, labels)):
        try:
            exact_file, swift_file = _find_counterpart_file(input_file)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {input_file}: {e}")
            continue

        # Read exact data
        exact_data = np.loadtxt(exact_file)
        exact_ids = exact_data[:, 0]
        exact_pos = exact_data[:, 1:4]
        exact_a = exact_data[:, 4:7]
        exact_pot = exact_data[:, 7]

        # Sort exact data
        sort_index = np.argsort(exact_ids)
        exact_ids = exact_ids[sort_index]
        exact_pos = exact_pos[sort_index, :]
        exact_a = exact_a[sort_index, :]
        exact_pot = exact_pot[sort_index]
        exact_a_norm = np.sqrt(
            exact_a[:, 0] ** 2 + exact_a[:, 1] ** 2 + exact_a[:, 2] ** 2
        )

        # Read SWIFT data
        swift_data = np.loadtxt(swift_file)
        swift_ids = swift_data[:, 0]
        swift_pos = swift_data[:, 1:4]
        swift_a_grav = swift_data[:, 4:7]
        swift_pot = swift_data[:, 7]

        # Sort SWIFT data
        sort_index = np.argsort(swift_ids)
        swift_ids = swift_ids[sort_index]
        swift_pos = swift_pos[sort_index, :]
        swift_a_grav = swift_a_grav[sort_index, :]
        swift_pot = swift_pot[sort_index]

        # Compute errors
        diff = exact_a - swift_a_grav
        diff_pot = exact_pot - swift_pot

        # Correct for different normalization of potential
        exact_pot_corrected = exact_pot - np.mean(diff_pot)
        diff_pot = exact_pot_corrected - swift_pot

        norm_diff = np.sqrt(
            diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
        )
        norm_error = norm_diff / exact_a_norm
        error_x = np.abs(diff[:, 0]) / exact_a_norm
        error_y = np.abs(diff[:, 1]) / exact_a_norm
        error_z = np.abs(diff[:, 2]) / exact_a_norm
        error_pot = np.abs(diff_pot) / np.abs(exact_pot_corrected)

        # Bin the errors
        norm_error_hist, _ = np.histogram(
            norm_error, bins=bin_edges, density=False
        )
        norm_error_hist = norm_error_hist / (np.size(norm_error) * bin_size)

        error_x_hist, _ = np.histogram(error_x, bins=bin_edges, density=False)
        error_x_hist = error_x_hist / (np.size(norm_error) * bin_size)

        error_y_hist, _ = np.histogram(error_y, bins=bin_edges, density=False)
        error_y_hist = error_y_hist / (np.size(norm_error) * bin_size)

        error_z_hist, _ = np.histogram(error_z, bins=bin_edges, density=False)
        error_z_hist = error_z_hist / (np.size(norm_error) * bin_size)

        error_pot_hist, _ = np.histogram(
            error_pot, bins=bin_edges, density=False
        )
        error_pot_hist = error_pot_hist / (np.size(norm_error) * bin_size)

        # Calculate statistics
        norm_median = np.median(norm_error)
        median_x = np.median(error_x)
        median_y = np.median(error_y)
        median_z = np.median(error_z)
        median_pot = np.median(error_pot)

        norm_per99 = np.percentile(norm_error, 99)
        per99_x = np.percentile(error_x, 99)
        per99_y = np.percentile(error_y, 99)
        per99_z = np.percentile(error_z, 99)
        per99_pot = np.percentile(error_pot, 99)

        # Use colors from the color list, cycling if necessary
        color = cols[i % len(cols)]

        # Plot results
        ax1.semilogx(bins, error_x_hist, color=color, label=label)
        ax1.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_x:.5f} 99%→{per99_x:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax2.semilogx(bins, error_y_hist, color=color, label=label)
        ax2.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_y:.5f} 99%→{per99_y:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax3.semilogx(bins, error_z_hist, color=color, label=label)
        ax3.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_z:.5f} 99%→{per99_z:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax4.semilogx(bins, norm_error_hist, color=color, label=label)
        ax4.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{norm_median:.5f} 99%→{norm_per99:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax5.semilogx(bins, error_pot_hist, color=color, label=label)
        ax5.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_pot:.5f} 99%→{per99_pot:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        print(f"Processed: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")
        print(
            f"  Norm error - median: {norm_median:.5f}, 99%: {norm_per99:.5f}"
        )
        print()

    # Set up axes labels and limits
    ax1.set_xlabel("δax/|a_exact|")
    ax1.set_xlim(min_error, max_error)
    ax1.set_ylim(0, 1.75)

    ax2.set_xlabel("δay/|a_exact|")
    ax2.set_xlim(min_error, max_error)
    ax2.set_ylim(0, 1.75)

    ax3.set_xlabel("δaz/|a_exact|")
    ax3.set_xlim(min_error, max_error)
    ax3.set_ylim(0, 1.75)

    ax4.set_xlabel("|δa|/|a_exact|")
    ax4.set_xlim(min_error, max_error)
    ax4.set_ylim(0, 2.5)
    ax4.legend(loc="upper left", fontsize=8)

    ax5.set_xlabel("δφ/φ_exact")
    ax5.set_xlim(min_error, max_error)
    ax5.set_ylim(0, 1.75)

    # Save the figure
    png_file = create_output_path(output_path, prefix, "gravity_checks.png")

    fig.savefig(png_file, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {png_file}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    plt.close(fig)


def analyse_gravity_error_maps(
    files: list[str],
    labels: list[str],
    output_path: str | None = None,
    prefix: str = None,
    show_plot: bool = True,
    resolution: int = 100,
    error_thresh: float = 1e-2,
) -> None:
    """Create hexbin error maps for gravity check files.

    Args:
        files: List of file paths to either exact or SWIFT force files.
               The function will automatically find the counterpart files.
        labels: List of labels for the runs.
        output_path: Optional path to save the plot. If None, the plot is saved
            to the current directory.
        prefix: Optional prefix to add to the output filename if saving.
            If empty, defaults to 'gravity_error_map.png'.
        show_plot: Whether to display the plot.
        resolution: Resolution (gridsize) for hexbin plot (default: 100).

    Raises:
        ValueError: If the number of files and labels do not match.
        FileNotFoundError: If counterpart files cannot be found.
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    # Process each file and create individual error maps
    for i, (input_file, label) in enumerate(zip(files, labels)):
        try:
            exact_file, swift_file = _find_counterpart_file(input_file)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {input_file}: {e}")
            continue

        # Read exact data
        exact_data = np.loadtxt(exact_file)
        exact_ids = exact_data[:, 0]
        exact_pos = exact_data[:, 1:4]
        exact_a = exact_data[:, 4:7]

        # Sort exact data
        sort_index = np.argsort(exact_ids)
        exact_ids = exact_ids[sort_index]
        exact_pos = exact_pos[sort_index, :]
        exact_a = exact_a[sort_index, :]
        exact_a_norm = np.sqrt(
            exact_a[:, 0] ** 2 + exact_a[:, 1] ** 2 + exact_a[:, 2] ** 2
        )

        # Read SWIFT data
        swift_data = np.loadtxt(swift_file)
        swift_ids = swift_data[:, 0]
        swift_pos = swift_data[:, 1:4]
        swift_a_grav = swift_data[:, 4:7]  # a_swift columns

        # Sort SWIFT data
        sort_index = np.argsort(swift_ids)
        swift_ids = swift_ids[sort_index]
        swift_pos = swift_pos[sort_index, :]
        swift_a_grav = swift_a_grav[sort_index, :]

        # Compute errors
        diff = exact_a - swift_a_grav
        norm_diff = np.sqrt(
            diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
        )
        norm_error = norm_diff / exact_a_norm

        # Create individual error map for this file
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Gravity Error Maps - {label}", fontsize=16)

        # X-Y projection
        hb1 = ax1.hexbin(
            exact_pos[:, 0],
            exact_pos[:, 1],
            C=norm_error,
            gridsize=resolution,
            cmap="viridis",
            norm=LogNorm(vmin=1e-7, vmax=1e-1),
        )
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_title("X-Y Projection")
        ax1.set_aspect("equal")
        cb1 = plt.colorbar(hb1, ax=ax1)
        cb1.set_label("|δa|/|a_exact|")

        # X-Z projection
        hb2 = ax2.hexbin(
            exact_pos[:, 0],
            exact_pos[:, 2],
            C=norm_error,
            gridsize=resolution,
            cmap="viridis",
            norm=LogNorm(vmin=1e-7, vmax=1e-1),
        )
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Z Position")
        ax2.set_title("X-Z Projection")
        ax2.set_aspect("equal")
        cb2 = plt.colorbar(hb2, ax=ax2)
        cb2.set_label("|δa|/|a_exact|")

        # Y-Z projection
        hb3 = ax3.hexbin(
            exact_pos[:, 1],
            exact_pos[:, 2],
            C=norm_error,
            gridsize=resolution,
            cmap="viridis",
            norm=LogNorm(vmin=1e-7, vmax=1e-1),
        )
        ax3.set_xlabel("Y Position")
        ax3.set_ylabel("Z Position")
        ax3.set_title("Y-Z Projection")
        ax3.set_aspect("equal")
        cb3 = plt.colorbar(hb3, ax=ax3)
        cb3.set_label("|δa|/|a_exact|")

        # Error vs distance from center
        # Assuming box center at [0.5, 0.5, 0.5]
        center = np.array([0.5, 0.5, 0.5])
        distance_from_center = np.sqrt(
            np.sum((exact_pos - center) ** 2, axis=1)
        )

        hb4 = ax4.hexbin(
            distance_from_center,
            norm_error,
            gridsize=resolution,
            cmap="viridis",
            norm=LogNorm(vmin=1e-7, vmax=1e-1),
        )
        ax4.set_xlabel("Distance from Center")
        ax4.set_ylabel("|δa|/|a_exact|")
        ax4.set_title("Error vs Distance from Center")
        ax4.set_yscale("log")
        cb4 = plt.colorbar(hb4, ax=ax4)
        cb4.set_label("Particle Density")

        plt.tight_layout()

        # Save the figure
        safe_label = label.replace(" ", "_").replace("/", "_")
        png_file = create_output_path(
            output_path, prefix, f"gravity_error_map_{safe_label}.png"
        )

        fig.savefig(png_file, dpi=200, bbox_inches="tight")
        print(f"Error map saved to {png_file}")

        # Show the plot if requested
        if show_plot:
            plt.show()
        plt.close()

        # Create individual error map for this file with error threshold based
        # colors
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Gravity Error Maps - {label}", fontsize=16)

        # Remove the axis from the 4th subplot
        ax4.axis("off")

        # Create mask for errors above and below threshold
        mask = norm_error < error_thresh

        # X-Y projection
        ax1.scatter(
            exact_pos[mask, 0],
            exact_pos[mask, 1],
            c="blue",
            s=1,
            label=f"|δa|/|a_exact| < {error_thresh:.1e}",
        )
        ax1.scatter(
            exact_pos[~mask, 0],
            exact_pos[~mask, 1],
            c="red",
            s=1,
            label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
        )
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_title("X-Y Projection")
        ax1.set_aspect("equal")
        ax1.legend(markerscale=5)

        # X-Z projection
        ax2.scatter(
            exact_pos[mask, 0],
            exact_pos[mask, 2],
            c="blue",
            s=1,
            label=f"|δa|/|a_exact| < {error_thresh:.1e}",
        )
        ax2.scatter(
            exact_pos[~mask, 0],
            exact_pos[~mask, 2],
            c="red",
            s=1,
            label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
        )
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Z Position")
        ax2.set_title("X-Z Projection")
        ax2.set_aspect("equal")

        # Y-Z projection
        ax3.scatter(
            exact_pos[mask, 1],
            exact_pos[mask, 2],
            c="blue",
            s=1,
            label=f"|δa|/|a_exact| < {error_thresh:.1e}",
        )
        ax3.scatter(
            exact_pos[~mask, 1],
            exact_pos[~mask, 2],
            c="red",
            s=1,
            label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
        )
        ax3.set_xlabel("Y Position")
        ax3.set_ylabel("Z Position")
        ax3.set_title("Y-Z Projection")
        ax3.set_aspect("equal")

        plt.tight_layout()

        # Save the figure
        safe_label = label.replace(" ", "_").replace("/", "_")
        png_file = create_output_path(
            output_path, prefix, f"gravity_binary_error_map_{safe_label}.png"
        )

        fig.savefig(png_file, dpi=200, bbox_inches="tight")
        print(f"Error map saved to {png_file}")

        # Show the plot if requested
        if show_plot:
            plt.show()
        plt.close()

        print(f"Processed error map for: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")
        print(
            f"  Error range: {np.min(norm_error):.2e} to"
            f" {np.max(norm_error):.2e}"
        )
        print()
