"""Analyse mode for analysing SWIFT runs."""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from swiftsim_cli.src_parser import (
    generate_timer_nesting_database,
)
from swiftsim_cli.utilities import create_ascii_table, create_output_path


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
        default=None,
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

    error_map_parser.add_argument(
        "--thresh",
        type=float,
        help="Error threshold for hexbin color scale (default: 1e-2).",
        default=1e-2,
    )

    log_parser = subparsers.add_parser(
        "log",
        help="Analyse timing information from SWIFT log files."
        " To get the most from this mode SWIFT should be run with "
        "-v 1 for verbose output.",
    )

    log_parser.add_argument(
        "log_file",
        help="SWIFT log file to analyse.",
        type=Path,
    )

    log_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    log_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    log_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    log_parser.add_argument(
        "--top-n",
        type=int,
        help="Number of top functions to show in detailed plots "
        "(default: 20).",
        default=20,
    )

    log_parser.add_argument(
        "--regenerate-nesting",
        action="store_true",
        help="Force regeneration of timer nesting database from source code.",
        default=False,
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
        error_thresh=args.thresh,
    )


def run_swift_log_timing(args: argparse.Namespace) -> None:
    """Execute the SWIFT log timing analysis."""
    analyse_swift_log_timings(
        log_file=str(args.log_file),
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        top_n=args.top_n,
        regenerate_nesting=args.regenerate_nesting,
    )


def run(args: argparse.Namespace) -> None:
    """Execute the analyse mode based on the selected subcommand."""
    if args.analysis_type == "timesteps":
        run_timestep(args)
    elif args.analysis_type == "gravity-checks":
        run_gravity_checks(args)
    elif args.analysis_type == "gravity-error-map":
        run_gravity_error_maps(args)
    elif args.analysis_type == "log":
        run_swift_log_timing(args)
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

        norm_per90 = np.percentile(norm_error, 90)
        per90_x = np.percentile(error_x, 90)
        per90_y = np.percentile(error_y, 90)
        per90_z = np.percentile(error_z, 90)
        per90_pot = np.percentile(error_pot, 90)

        # Use colors from the color list, cycling if necessary
        color = cols[i % len(cols)]

        # Plot results
        ax1.semilogx(bins, error_x_hist, color=color, label=label)
        ax1.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_x:.5f} 90%→{per90_x:.5f} 99%→{per99_x:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax2.semilogx(bins, error_y_hist, color=color, label=label)
        ax2.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_y:.5f} 90%→{per90_y:.5f} 99%→{per99_y:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax3.semilogx(bins, error_z_hist, color=color, label=label)
        ax3.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_z:.5f} 90%→{per90_z:.5f} 99%→{per99_z:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax4.semilogx(bins, norm_error_hist, color=color, label=label)
        ax4.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{norm_median:.5f} 90%→{norm_per90:.5f} 99%→{norm_per99:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        ax5.semilogx(bins, error_pot_hist, color=color, label=label)
        ax5.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_pot:.5f} 90%→{per90_pot:.5f} 99%→{per99_pot:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        print(f"Processed: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")
        print(
            f"  Norm error - median: {norm_median:.5f}, 90%: {norm_per90:.5f},"
            f" 99%: {norm_per99:.5f}",
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
        error_thresh: Error threshold for hexbin color scale (default: 1e-2).

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


def analyse_swift_log_timings(
    log_file: str,
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    top_n: int = 20,
    regenerate_nesting: bool = False,
) -> None:
    """Analyse SWIFT timing logs and emit full reports.

    This function is a drop-in analysis routine that restores and expands the
    outputs from your original analysis, while fixing nested timer
    double-counting using a pre-built YAML database of timing sites.

    It:
      * Parses step lines and task/category sections from the log (as before),
      * Uses `src_parser` helpers to map log lines to specific timing sites
        and to compute **exclusive** times (inclusive minus nested children),
      * Produces the same plots/tables you had originally (top by total time,
        top by calls, distributions, task categories over time, pies, counts,
        cumulative curves), plus an **Inclusive vs Exclusive** comparison,
      * Estimates per-step **untimed** time when a step total is available.

    Notes:
      * All site matching / nested-time logic is delegated to `src_parser`.
      * Display names use `function [file:line]` for uniqueness and
        traceability (based on timer_id from YAML).
      * The step "Wall-clock time [ms]" is parsed as the **last float** on
        the step line. If you have the exact column index, you can replace
        that one-liner.

    Args:
      log_file: Path to the SWIFT log file to analyse.
      output_path: Directory where figures are saved. If None, saves to CWD.
      prefix: Optional filename prefix and output subdirectory prefix.
      show_plot: Whether to display plots interactively.
      top_n: Number of top timers shown in multi-item plots/tables.
      regenerate_nesting: Force regeneration of timer nesting database from
      source.

    Returns:
      None. Figures are written to disk; a textual summary is printed to
            stdout.
    """
    # Local imports (keeps this function self-contained)
    import re
    from collections import Counter, defaultdict

    # Load timer nesting relationships
    from pathlib import Path
    from typing import Dict, List

    import matplotlib.pyplot as plt
    import numpy as np
    from ruamel.yaml import YAML
    from tqdm.auto import tqdm

    # Heavy lifting lives in src_parser (per your constraint).
    from swiftsim_cli.src_parser import (
        compile_site_patterns,
        load_timer_db,
        scan_log_instances_by_step,
    )

    def load_timer_nesting(auto_generate=True, force_regenerate=False):
        """Load timer nesting relationships.

        Args:
            auto_generate: If True, automatically generate nesting DB if it
                doesn't exist
            force_regenerate: If True, regenerate even if file exists
        """
        nesting_file = Path.home() / ".swiftsim-utils" / "timer_nesting.yaml"

        # Check if we need to generate the database
        should_generate = force_regenerate or (
            auto_generate and not nesting_file.exists()
        )

        if should_generate:
            try:
                print(
                    "Auto-generating timer nesting database "
                    "from source code..."
                )

                # Try to get SWIFT source directory from profile
                from swiftsim_cli.profile import load_swift_profile

                try:
                    profile = load_swift_profile()
                    swift_src = profile.get("swift_src")
                except Exception:
                    swift_src = "/Users/willroper/Research/SWIFT/swiftsim/src"

                # Generate the nesting database
                nesting_data = generate_timer_nesting_database(swift_src)

                # Save to file
                yaml_writer = YAML()
                yaml_writer.default_flow_style = False
                nesting_file.parent.mkdir(exist_ok=True)
                with open(nesting_file, "w") as f:
                    yaml_writer.dump(nesting_data, f)

                print(
                    "Generated nesting database with "
                    f"{len(nesting_data.get('nesting', {}))} functions"
                )
                return nesting_data.get("nesting", {})

            except Exception as e:
                print(
                    f"Warning: Failed to auto-generate nesting database: {e}"
                )
                # Fall back to empty if generation fails
                return {}

        # Load existing file
        if not nesting_file.exists():
            return {}

        yaml_safe = YAML(typ="safe")
        with open(nesting_file, "r", encoding="utf-8") as f:
            data = yaml_safe.load(f) or {}

        return data.get("nesting", {})

    print(f"Analyzing SWIFT log:  {log_file}")

    # Output directory naming identical to your original approach
    out_dir = (
        "runtime_analysis" if prefix is None else f"{prefix}_runtime_analysis"
    )

    print("Loading timer database and compiling patterns...")
    # Load DB + compile patterns
    timer_db = load_timer_db()
    compiled = compile_site_patterns(timer_db)
    nesting_db = load_timer_nesting(
        auto_generate=True, force_regenerate=regenerate_nesting
    )
    print(
        f"Loaded {len(timer_db)} timer definitions and compiled "
        f"{len(compiled)} patterns"
    )
    print(f"Loaded nesting relationships for {len(nesting_db)} functions")

    # Scan log to collect per-step timer instances (encounter order)
    print("Scanning log file for timer instances...")
    instances_by_step, _ = scan_log_instances_by_step(
        log_file, compiled, timer_db
    )

    total_instances = sum(
        len(instances) for instances in instances_by_step.values()
    )
    print(
        f"Found {total_instances} timer instances across "
        f"{len(instances_by_step)} steps"
    )

    # Dynamically classify timers: function timer = max time per function,
    # others = operations
    def classify_timers_by_max(instances_by_step, timer_db, nesting_db):
        """Classify timers dynamically with smart function timer detection.

        Uses the nesting database to guide classification when available,
        falling back to heuristics for functions not in the nesting database.
        For functions with multiple operations listed in the nesting database,
        looks for a generic function timer pattern. If not found, creates a
        synthetic function timer as the sum of all operations.

        Args:
            instances_by_step: Dictionary mapping step numbers to timer
                instances
            timer_db: Dictionary of timer definitions by timer ID
            nesting_db: Dictionary of nesting relationships by function name

        Returns:
            Tuple of (function_timer_ids, synthetic_function_timers) where:
            - function_timer_ids: Set of timer IDs classified as function
              timers
            - synthetic_function_timers: Dict of function_name -> total_time
              for functions needing synthetic timers
        """
        # Collect all timer times by function
        timer_totals_by_function = defaultdict(lambda: defaultdict(float))

        for inst_list in instances_by_step.values():
            for inst in inst_list:
                func_name = timer_db[inst.timer_id].function
                timer_totals_by_function[func_name][inst.timer_id] += (
                    inst.time_ms
                )

        # For each function, determine the function timer intelligently
        function_timer_ids = set()

        # For functions without explicit function timer
        synthetic_function_timers = {}

        for func_name, timer_totals in timer_totals_by_function.items():
            if not timer_totals:  # Skip if no timers
                continue

            # Check if nesting database has guidance for this function
            if func_name in nesting_db and nesting_db[func_name].get(
                "nested_operations"
            ):
                # Nesting database indicates this function should have
                # multiple operations. Look for a timer that matches the
                # function_timer pattern
                function_timer_pattern = nesting_db[func_name].get(
                    "function_timer", ""
                )
                function_timer_found = False

                # Try to find a timer matching the function timer pattern
                for tid in timer_totals.keys():
                    timer_label = timer_db[tid].label_text
                    # Simple pattern matching - if function timer pattern is
                    # "took %.3f %s." then look for timers with just "took"
                    # without specific operation descriptions
                    if (
                        function_timer_pattern
                        and "took" in function_timer_pattern
                        and "took" in timer_label
                    ):
                        # Check if this is a generic "took" timer (not a
                        # specific operation)
                        # Specific operations usually have descriptive text
                        # before "took"
                        words_before_took = timer_label.split("took")[
                            0
                        ].strip()
                        if (
                            not words_before_took
                            or len(words_before_took.split()) <= 2
                        ):
                            # This looks like a generic function timer
                            function_timer_ids.add(tid)
                            function_timer_found = True
                            break

                if not function_timer_found:
                    # No function timer found, create synthetic one from sum
                    # of operations
                    total_time = sum(timer_totals.values())
                    synthetic_function_timers[func_name] = total_time
                    # All existing timers remain as operations
            else:
                # No nesting database guidance, fall back to heuristic
                if len(timer_totals) == 1:
                    # Only one timer - it's the function timer
                    function_timer_ids.add(list(timer_totals.keys())[0])
                else:
                    # Multiple timers - check if max timer represents the
                    # whole function
                    sorted_timers = sorted(
                        timer_totals.items(), key=lambda x: x[1], reverse=True
                    )
                    max_timer_id, max_time = sorted_timers[0]
                    other_timers_sum = sum(
                        time for tid, time in sorted_timers[1:]
                    )

                    # Use a more sophisticated heuristic:
                    # Only treat max timer as function timer if it's
                    # significantly larger
                    # (at least 2x) than the sum of others, indicating item
                    # encompasses them
                    ratio_threshold = 2.0
                    if max_time > ratio_threshold * other_timers_sum:
                        # Max timer is significantly larger than sum of
                        # others - it's the function timer
                        function_timer_ids.add(max_timer_id)
                    else:
                        # No single dominant timer - function timer is sum of
                        # all operations. We'll create a synthetic function
                        # timer entry
                        total_time = sum(timer_totals.values())
                        synthetic_function_timers[func_name] = total_time

        # Update timer_db with dynamic classification
        for tid, timer_def in timer_db.items():
            if tid in function_timer_ids:
                timer_def.timer_type = "function"
            else:
                timer_def.timer_type = "operation"

        return function_timer_ids, synthetic_function_timers

    print(
        "Dynamically classifying timers based on maximum time per function..."
    )
    function_timer_ids, synthetic_function_timers = classify_timers_by_max(
        instances_by_step, timer_db, nesting_db
    )
    print(f"Identified {len(function_timer_ids)} explicit function timers")
    print(
        f"Identified {len(synthetic_function_timers)} synthetic function"
        " timers (sum of operations)"
    )

    # Aggregate timing statistics by timer site and timer type
    print("Aggregating timing statistics by timer site and type...")
    function_times_by_tid: Dict[str, List[float]] = defaultdict(list)
    operation_times_by_tid: Dict[str, List[float]] = defaultdict(list)
    call_counts_by_tid: Counter = Counter()

    for inst_list in instances_by_step.values():
        for inst in inst_list:
            if inst.timer_type == "function":
                function_times_by_tid[inst.timer_id].append(inst.time_ms)
            else:  # operation
                operation_times_by_tid[inst.timer_id].append(inst.time_ms)
            call_counts_by_tid[inst.timer_id] += 1

    # Utility to render a readable name
    def display_name(tid: str) -> str:
        if tid.startswith("SYNTHETIC:"):
            func_name = tid[10:]  # Remove "SYNTHETIC:" prefix
            return f"{func_name} [SYNTHETIC:sum_of_operations]"
        td = timer_db[tid]
        return f"{td.function} [{tid}]"

    # Build stats for function and operation timers
    def build_stats(values: List[float]) -> Dict[str, float | int]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return dict(
                total_time=0.0,
                mean_time=0.0,
                median_time=0.0,
                std_time=0.0,
                max_time=0.0,
                min_time=0.0,
                call_count=0,
            )
        return dict(
            total_time=float(np.sum(arr)),
            mean_time=float(np.mean(arr)),
            median_time=float(np.median(arr)),
            std_time=float(np.std(arr)),
            max_time=float(np.max(arr)),
            min_time=float(np.min(arr)),
            call_count=int(arr.size),
        )

    function_stats = {
        tid: build_stats(vals) for tid, vals in function_times_by_tid.items()
    }

    # Add synthetic function timers (sum of operations for functions
    # without explicit function timer)
    for func_name, total_time in synthetic_function_timers.items():
        # Create a synthetic timer ID for this function
        synthetic_tid = f"SYNTHETIC:{func_name}"
        # Add to function stats as a single call with the total time
        function_stats[synthetic_tid] = {
            "total_time": total_time,
            "mean_time": total_time,
            "median_time": total_time,
            "std_time": 0.0,
            "max_time": total_time,
            "min_time": total_time,
            "call_count": 1,
        }
    operation_stats = {
        tid: build_stats(vals) for tid, vals in operation_times_by_tid.items()
    }

    # Sort by total time for each timer type
    sorted_functions = sorted(
        function_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )
    sorted_operations = sorted(
        operation_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )

    # For legacy compatibility, combine all timers for some plots
    all_stats = {**function_stats, **operation_stats}
    sorted_all = sorted(
        all_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )
    sorted_by_calls = sorted(
        all_stats.items(),
        key=lambda x: x[1]["call_count"],
        reverse=True,
    )

    print("Parsing task categories, step info, and additional log data...")
    step_info: List[Dict[str, float | int]] = []
    task_category_times: Dict[str, List[float]] = defaultdict(list)
    task_counts: Dict[str, List[int]] = defaultdict(list)
    step_totals: Dict[
        int, float
    ] = {}  # wall-clock ms per step (last float on step line)
    step_seen_order: List[int] = []

    # Regex mirrors your original, plus helpers for categories & counts
    step_line_re = re.compile(
        r"^\s*(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)"
    )
    task_line_re = re.compile(
        r"\*\*\*\s+([^:]+):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s%?\)"
    )
    counts_re = re.compile(r"task counts are \[(.*)\]")
    last_float_re = re.compile(r"([-\d.]+)(?:\s*)$")

    # Read the log file and count lines for progress
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    print(
        f"Processing {len(lines)} log lines for task categories and "
        "step info..."
    )

    for raw in tqdm(lines, desc="Parsing log lines", unit="line"):
        line = raw.strip()

        # Step info rows (capture first 5 numeric columns as in your original)
        if sm := step_line_re.match(line):
            try:
                step_num = int(sm.group(1))
                step_seen_order.append(step_num)
                step_info.append(
                    dict(
                        step=step_num,
                        time_1=float(sm.group(2)),
                        time_2=float(sm.group(3)),
                        time_3=float(sm.group(4)),
                        time_4=float(sm.group(5)),
                    )
                )
                # Also capture step total wall-clock ms as last float on
                # the line
                if wm := last_float_re.search(line):
                    try:
                        step_totals[step_num] = float(wm.group(1))
                    except ValueError:
                        pass
            except ValueError:
                # Skip malformed lines (keep behavior similar to your
                # warning pattern)
                pass
            continue

        # Task category summary rows
        if tm := task_line_re.search(line):
            category = tm.group(1).strip()
            if category.lower() != "total":
                try:
                    time_ms = float(tm.group(2))
                    task_category_times[category].append(time_ms)
                except ValueError:
                    pass
            continue

        # Task counts rows
        if cm := counts_re.search(line):
            parts = cm.group(1).split()
            for token in parts:
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        task_counts[k].append(int(v))
                    except ValueError:
                        pass
            continue

    # Note: Untimed calculation removed since we no longer use nesting

    print("Generating analysis plots...")
    plots_created: List[str] = []

    # Helper: light label truncation
    def trunc(s: str, n: int) -> str:
        return s if len(s) <= n else (s[:n] + "...")

    # 1) Top function timers by total time (log scale)
    if sorted_functions:
        fig, ax = plt.subplots(figsize=(12, 8))
        top = sorted_functions[:top_n]
        names = [display_name(tid) for tid, _ in top]
        totals = [st["total_time"] for _, st in top]
        bars = ax.barh(range(len(names)), totals)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([trunc(n, 60) for n in names], fontsize=10)
        ax.set_xlabel("Total Time (ms)")
        ax.set_title(f"Top {top_n} Function Timers by Total Time")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 1.01,
                b.get_y() + b.get_height() / 2,
                f"{totals[i]:.1f}ms",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        p = create_output_path(
            output_path,
            prefix,
            "01_function_timers_by_total_time.png",
            out_dir,
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 1b) Top operation timers by total time (log scale)
    if sorted_operations:
        fig, ax = plt.subplots(figsize=(12, 8))
        top = sorted_operations[:top_n]
        names = [display_name(tid) for tid, _ in top]
        totals = [st["total_time"] for _, st in top]
        bars = ax.barh(range(len(names)), totals)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([trunc(n, 60) for n in names], fontsize=10)
        ax.set_xlabel("Total Time (ms)")
        ax.set_title(f"Top {top_n} Operation Timers by Total Time")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 1.01,
                b.get_y() + b.get_height() / 2,
                f"{totals[i]:.1f}ms",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        p = create_output_path(
            output_path,
            prefix,
            "01b_operation_timers_by_total_time.png",
            out_dir,
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 2) Top timers by **call count** (as before; still useful)
    if sorted_by_calls:
        fig, ax = plt.subplots(figsize=(12, 8))
        top = sorted_by_calls[:top_n]
        names = [display_name(tid) for tid, _ in top]
        calls = [
            all_stats.get(tid, {"call_count": 0})["call_count"]
            for tid, _ in top
        ]
        bars = ax.barh(range(len(names)), calls)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([trunc(n, 60) for n in names], fontsize=10)
        ax.set_xlabel("Call Count")
        ax.set_title(f"Top {top_n} Timers by Call Count")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, max(calls) * 1.1 if calls else 1)
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 1.01,
                b.get_y() + b.get_height() / 2,
                f"{int(calls[i])}",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "02_timers_by_call_count.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 3) Major task categories over time (scatter), same as your original idea
    if task_category_times:
        fig, ax = plt.subplots(figsize=(12, 6))
        major_categories = [
            "gravity",
            "dead time",
            "drift",
            "time integration",
        ]
        for category in major_categories:
            if category in task_category_times:
                times = task_category_times[category]
                steps = range(len(times))
                ax.scatter(
                    steps, times, marker=".", s=10, alpha=0.7, label=category
                )
        ax.set_xlabel("Step Number")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Major Task Categories Over Time")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "03_task_categories_over_time.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 4) Timing distribution (violin) for top function timers
    violin_data, violin_labels = [], []
    for tid, _ in sorted_functions[:15]:
        vals = function_times_by_tid.get(tid, [])
        if len(vals) > 1:
            violin_data.append(vals)
            violin_labels.append(trunc(display_name(tid), 25))
    if violin_data:
        fig, ax = plt.subplots(figsize=(14, 8))
        parts = ax.violinplot(
            violin_data,
            range(len(violin_data)),
            showmeans=True,
            showmedians=True,
        )
        for pc in parts.get("bodies", []):
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(violin_labels)))
        ax.set_xticklabels(violin_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Timing Distribution for Top Function Timers")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "04_timing_distribution.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 5) Task category pie (sum over observed entries)
    if task_category_times:
        fig, ax = plt.subplots(figsize=(10, 8))
        cat_totals = {
            cat: sum(times)
            for cat, times in task_category_times.items()
            if times
        }
        if cat_totals:
            items = sorted(
                cat_totals.items(), key=lambda x: x[1], reverse=True
            )
            items = items[:8]
            labels = [k for k, _ in items]
            sizes = [v for _, v in items]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%", startangle=90
            )
            ax.set_title("Time Distribution by Task Category")
            for at in autotexts:
                at.set_color("white")
                at.set_fontweight("bold")
            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "05_task_category_pie.png", out_dir
            )
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
        plt.close()

    # 6) Top task categories over time (log y)
    if task_category_times:
        fig, ax = plt.subplots(figsize=(12, 8))
        cat_totals = {
            c: sum(t)
            for c, t in task_category_times.items()
            if c.lower() != "total"
        }
        top_cats = sorted(
            cat_totals.items(), key=lambda x: x[1], reverse=True
        )[:6]
        for c, _ in top_cats:
            times = task_category_times[c]
            steps = range(len(times))
            ax.scatter(steps, times, marker=".", s=10, alpha=0.7, label=c)
        ax.set_xlabel("Step Number")
        ax.set_ylabel("Time (ms)")
        ax.set_yscale("log")
        ax.set_title("Top Task Categories Over Time")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "06_top_tasks_over_time.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 7) "Efficiency" plot — average time per call for function timers
    if sorted_functions:
        fig, ax = plt.subplots(figsize=(12, 8))
        eff = []
        for tid, s in sorted_functions[:20]:
            calls = s["call_count"]
            val = (s["total_time"] / calls) if calls > 0 else 0.0
            eff.append((tid, val))
        eff.sort(key=lambda x: x[1], reverse=True)
        names = [trunc(display_name(tid), 35) for tid, _ in eff]
        values = [v for _, v in eff]
        bars = ax.barh(range(len(names)), values)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Average Time per Call (ms)")
        ax.set_title("Function Timer Efficiency (Time per Call)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, max(values) * 1.1 if values else 1)
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 1.01,
                b.get_y() + b.get_height() / 2,
                f"{values[i]:.2f}ms",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "07_function_efficiency.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 8) Cumulative time analysis for all timers
    if sorted_all:
        fig, ax = plt.subplots(figsize=(10, 6))
        top = sorted_all[:25]
        cum = np.cumsum([s["total_time"] for _, s in top])
        pct = 100 * cum / cum[-1] if cum[-1] > 0 else cum
        ax.plot(range(1, len(pct) + 1), pct, "ro-", linewidth=2, markersize=4)
        ax.set_xlabel("Number of Top Timers")
        ax.set_ylabel("Cumulative % of Total Time")
        ax.set_title("Cumulative Time Distribution")
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=80,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="80% threshold",
        )
        ax.axhline(
            y=90,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="90% threshold",
        )
        ax.legend(fontsize=11)
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "08_cumulative_time_analysis.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 9) Task counts over time (log y), excluding 'skipped'/'none' if present
    if task_counts:
        fig, ax = plt.subplots(figsize=(14, 8))
        task_max = {}
        for tname, counts in task_counts.items():
            if counts and tname not in ("skipped", "none") and max(counts) > 0:
                task_max[tname] = max(counts)
        top_types = sorted(task_max.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        for tname, _ in top_types:
            counts = task_counts[tname]
            steps = range(len(counts))
            ax.scatter(steps, counts, marker=".", s=10, alpha=0.7, label=tname)
        ax.set_xlabel("Step Number")
        ax.set_ylabel("Task Count")
        ax.set_yscale("log")
        ax.set_title("Top Task Types Over Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "09_task_counts_over_time.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 10) Function vs Operation timers comparison
    if sorted_functions and sorted_operations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Function timers subplot
        top_func = sorted_functions[: min(top_n // 2, 10)]
        func_names = [trunc(display_name(tid), 40) for tid, _ in top_func]
        func_totals = [st["total_time"] for _, st in top_func]
        ax1.barh(range(len(func_names)), func_totals, color="skyblue")
        ax1.set_yticks(range(len(func_names)))
        ax1.set_yticklabels(func_names, fontsize=9)
        ax1.set_xlabel("Total Time (ms)")
        ax1.set_title("Top Function Timers")
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3, axis="x")

        # Operation timers subplot
        top_op = sorted_operations[: min(top_n // 2, 10)]
        op_names = [trunc(display_name(tid), 40) for tid, _ in top_op]
        op_totals = [st["total_time"] for _, st in top_op]
        ax2.barh(range(len(op_names)), op_totals, color="lightcoral")
        ax2.set_yticks(range(len(op_names)))
        ax2.set_yticklabels(op_names, fontsize=9)
        ax2.set_xlabel("Total Time (ms)")
        ax2.set_title("Top Operation Timers")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "10_function_vs_operation_timers.png", out_dir
        )
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 11) Hierarchical call graph visualization
    if nesting_db:
        try:
            import networkx as nx

            # Build the call graph
            G = nx.DiGraph()

            # Add nodes and edges from nesting database
            for func_name, data in nesting_db.items():
                if (
                    func_name in function_stats
                ):  # Only include functions that actually ran
                    func_time = function_stats[func_name]["total_time"]
                    G.add_node(func_name, time=func_time, type="function")

                    # Add edges to nested functions
                    for nested_func in data.get("nested_functions", []):
                        if nested_func in function_stats:
                            nested_time = function_stats[nested_func][
                                "total_time"
                            ]
                            G.add_node(
                                nested_func, time=nested_time, type="function"
                            )
                            G.add_edge(func_name, nested_func)

            if len(G.nodes()) > 1:
                fig, ax = plt.subplots(figsize=(16, 12))

                # Use hierarchical layout
                try:
                    pos = nx.spring_layout(G, k=3, iterations=50)
                except Exception:
                    pos = nx.circular_layout(G)

                # Node sizes based on timing (log scale)
                times = [G.nodes[node].get("time", 1) for node in G.nodes()]
                max_time = max(times) if times else 1
                node_sizes = [
                    1000 * (time / max_time) ** 0.5 for time in times
                ]

                # Draw the graph
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    node_size=node_sizes,
                    node_color="lightblue",
                    alpha=0.7,
                    ax=ax,
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edge_color="gray",
                    arrows=True,
                    arrowsize=20,
                    alpha=0.6,
                    ax=ax,
                )

                # Add labels
                labels = {
                    node: f"{node}\\n{G.nodes[node].get('time', 0):.1f}ms"
                    for node in G.nodes()
                }
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

                ax.set_title(
                    "Function Call Graph (Node size ∝ execution time)",
                    fontsize=14,
                )
                ax.axis("off")

                plt.tight_layout()
                p = create_output_path(
                    output_path,
                    prefix,
                    "11_hierarchical_call_graph.png",
                    out_dir,
                )
                plt.savefig(p, dpi=200, bbox_inches="tight")
                plots_created.append(p)
                if show_plot:
                    plt.show()
                plt.close()
        except ImportError:
            print("NetworkX not available, skipping call graph visualization")
        except Exception as e:
            print(f"Failed to create call graph: {e}")

    # 12) Hierarchical timing breakdown (treemap style using matplotlib)
    if nesting_db and function_stats:
        try:
            import squarify

            # Get top-level functions (those not called by others)
            all_nested = set()
            for func_data in nesting_db.values():
                all_nested.update(func_data.get("nested_functions", []))

            top_level_funcs = []
            for func_name in function_stats.keys():
                if func_name not in all_nested and func_name in nesting_db:
                    top_level_funcs.append(func_name)

            if top_level_funcs:
                # Create treemap data
                treemap_data = []
                treemap_labels = []

                for func_name in sorted(
                    top_level_funcs,
                    key=lambda x: function_stats[x]["total_time"],
                    reverse=True,
                )[:10]:  # Top 10
                    func_time = function_stats[func_name]["total_time"]
                    treemap_data.append(func_time)
                    treemap_labels.append(f"{func_name}\\n{func_time:.1f}ms")

                if treemap_data:
                    fig, ax = plt.subplots(figsize=(14, 10))

                    # Create treemap
                    colors = plt.cm.Set3(np.linspace(0, 1, len(treemap_data)))
                    squarify.plot(
                        sizes=treemap_data,
                        label=treemap_labels,
                        color=colors,
                        alpha=0.8,
                        ax=ax,
                    )

                    ax.set_title(
                        "Top-Level Function Timing Breakdown (Treemap)",
                        fontsize=14,
                    )
                    ax.axis("off")

                    plt.tight_layout()
                    p = create_output_path(
                        output_path, prefix, "12_timing_treemap.png", out_dir
                    )
                    plt.savefig(p, dpi=200, bbox_inches="tight")
                    plots_created.append(p)
                    if show_plot:
                        plt.show()
                    plt.close()
        except ImportError:
            print("Squarify not available, skipping treemap visualization")
        except Exception as e:
            print(f"Failed to create treemap: {e}")

    # 13) Nesting depth analysis
    if nesting_db and function_stats:
        # Calculate nesting depths
        def calculate_depth(func_name, visited=None):
            if visited is None:
                visited = set()
            if func_name in visited:
                return 0
            visited.add(func_name)

            nested_funcs = nesting_db.get(func_name, {}).get(
                "nested_functions", []
            )
            if not nested_funcs:
                return 1

            max_depth = 0
            for nested in nested_funcs:
                if nested in nesting_db:
                    depth = calculate_depth(nested, visited.copy())
                    max_depth = max(max_depth, depth)

            return max_depth + 1

        depth_data = {}
        for func_name in function_stats.keys():
            if func_name in nesting_db:
                depth = calculate_depth(func_name)
                depth_data[func_name] = {
                    "depth": depth,
                    "time": function_stats[func_name]["total_time"],
                }

        if depth_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Depth distribution
            depths = [data["depth"] for data in depth_data.values()]
            ax1.hist(
                depths,
                bins=range(1, max(depths) + 2),
                alpha=0.7,
                edgecolor="black",
            )
            ax1.set_xlabel("Nesting Depth")
            ax1.set_ylabel("Number of Functions")
            ax1.set_title("Function Nesting Depth Distribution")
            ax1.grid(True, alpha=0.3)

            # Time vs depth scatter
            depths = [data["depth"] for data in depth_data.values()]
            times = [data["time"] for data in depth_data.values()]
            ax2.scatter(depths, times, alpha=0.6)
            ax2.set_xlabel("Nesting Depth")
            ax2.set_ylabel("Total Time (ms)")
            ax2.set_title("Execution Time vs Nesting Depth")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "13_nesting_depth_analysis.png", out_dir
            )
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    # 14) Function efficiency analysis (time per call vs nesting level)
    if nesting_db and function_stats:
        # Get all nested functions at each level
        level_data = {}

        def categorize_by_level(func_name, level=0, visited=None):
            if visited is None:
                visited = set()
            if func_name in visited or func_name not in function_stats:
                return
            visited.add(func_name)

            if level not in level_data:
                level_data[level] = []

            func_stats = function_stats[func_name]
            time_per_call = func_stats["total_time"] / max(
                func_stats["call_count"], 1
            )
            level_data[level].append(
                {
                    "name": func_name,
                    "time_per_call": time_per_call,
                    "total_time": func_stats["total_time"],
                }
            )

            # Recurse to nested functions
            nested_funcs = nesting_db.get(func_name, {}).get(
                "nested_functions", []
            )
            for nested in nested_funcs:
                categorize_by_level(nested, level + 1, visited.copy())

        # Start from top-level functions
        all_nested = set()
        for func_data in nesting_db.values():
            all_nested.update(func_data.get("nested_functions", []))

        for func_name in function_stats.keys():
            if func_name not in all_nested and func_name in nesting_db:
                categorize_by_level(func_name)

        if level_data:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Create box plot of efficiency by level
            levels = sorted(level_data.keys())
            efficiency_data = []
            labels = []

            for level in levels:
                if level_data[level]:
                    efficiencies = [
                        item["time_per_call"] for item in level_data[level]
                    ]
                    efficiency_data.append(efficiencies)
                    labels.append(
                        f"Level {level}\\n({len(efficiencies)} funcs)"
                    )

            if efficiency_data:
                box_plot = ax.boxplot(
                    efficiency_data, labels=labels, patch_artist=True
                )

                # Color boxes
                colors = plt.cm.viridis(
                    np.linspace(0, 1, len(box_plot["boxes"]))
                )
                for patch, color in zip(box_plot["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel("Time per Call (ms)")
                ax.set_xlabel("Nesting Level")
                ax.set_title("Function Efficiency by Nesting Level")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                p = create_output_path(
                    output_path,
                    prefix,
                    "14_efficiency_by_nesting_level.png",
                    out_dir,
                )
                plt.savefig(p, dpi=200, bbox_inches="tight")
                plots_created.append(p)
                if show_plot:
                    plt.show()
                plt.close()

    # 15) Comparative analysis: Flat vs Hierarchical timing
    if nesting_db and function_stats:
        # Calculate "exclusive" time (time not spent in nested functions)
        exclusive_times = {}

        for func_name, stats in function_stats.items():
            if func_name in nesting_db:
                total_time = stats["total_time"]

                # Subtract time spent in nested functions
                nested_time = 0
                nested_funcs = nesting_db.get(func_name, {}).get(
                    "nested_functions", []
                )
                for nested_func in nested_funcs:
                    if nested_func in function_stats:
                        nested_time += function_stats[nested_func][
                            "total_time"
                        ]

                exclusive_time = max(0, total_time - nested_time)
                exclusive_times[func_name] = {
                    "total": total_time,
                    "exclusive": exclusive_time,
                    "nested": nested_time,
                }

        if exclusive_times:
            # Get top functions by total time
            top_funcs = sorted(
                exclusive_times.items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )[:15]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

            # Stacked bar chart: exclusive vs nested time
            func_names = [item[0] for item in top_funcs]
            exclusive = [item[1]["exclusive"] for item in top_funcs]
            nested = [item[1]["nested"] for item in top_funcs]

            y_pos = np.arange(len(func_names))

            ax1.barh(
                y_pos,
                exclusive,
                label="Exclusive Time",
                alpha=0.8,
                color="steelblue",
            )
            ax1.barh(
                y_pos,
                nested,
                left=exclusive,
                label="Nested Function Time",
                alpha=0.8,
                color="lightcoral",
            )

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(
                [
                    name[:25] + ("..." if len(name) > 25 else "")
                    for name in func_names
                ]
            )
            ax1.set_xlabel("Time (ms)")
            ax1.set_title("Function Time Breakdown: Exclusive vs Nested")
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis="x")

            # Efficiency comparison: exclusive time vs call count
            all_exclusive = [
                item[1]["exclusive"] for item in exclusive_times.items()
            ]
            all_total = [item[1]["total"] for item in exclusive_times.items()]
            all_calls = [
                function_stats[name]["call_count"]
                for name in exclusive_times.keys()
            ]

            # Time per call for exclusive time
            exclusive_per_call = [
                exc / max(calls, 1)
                for exc, calls in zip(all_exclusive, all_calls)
            ]
            total_per_call = [
                tot / max(calls, 1) for tot, calls in zip(all_total, all_calls)
            ]

            ax2.scatter(total_per_call, exclusive_per_call, alpha=0.6, s=60)
            ax2.set_xlabel("Total Time per Call (ms)")
            ax2.set_ylabel("Exclusive Time per Call (ms)")
            ax2.set_title(
                "Function Efficiency: Total vs Exclusive Time per Call"
            )
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            # Add diagonal line
            min_val = min(min(total_per_call), min(exclusive_per_call))
            max_val = max(max(total_per_call), max(exclusive_per_call))
            ax2.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.5,
                label="y=x",
            )
            ax2.legend()

            plt.tight_layout()
            p = create_output_path(
                output_path,
                prefix,
                "15_flat_vs_hierarchical_timing.png",
                out_dir,
            )
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    # 16) Function call hierarchy treemap visualization
    if nesting_db and function_stats:
        try:
            import squarify  # For treemap visualization

            # Build hierarchical data for treemap
            def build_treemap_data():
                treemap_data = []

                # Find top-level functions (not nested in others)
                all_nested = set()
                for func_data in nesting_db.values():
                    all_nested.update(func_data.get("nested_functions", []))

                top_level_funcs = [
                    func
                    for func in function_stats.keys()
                    if func not in all_nested and func in nesting_db
                ]

                for func_name in sorted(
                    top_level_funcs,
                    key=lambda x: function_stats[x]["total_time"],
                    reverse=True,
                )[:8]:  # Top 8 for readability
                    func_time = function_stats[func_name]["total_time"]

                    # Add main function
                    treemap_data.append(
                        {
                            "label": func_name[:20]
                            + ("..." if len(func_name) > 20 else ""),
                            "size": func_time,
                            "color": "steelblue",
                            "alpha": 0.8,
                        }
                    )

                    # Add nested functions
                    nested_funcs = nesting_db.get(func_name, {}).get(
                        "nested_functions", []
                    )
                    for nested_func in nested_funcs[
                        :3
                    ]:  # Top 3 nested for each
                        if nested_func in function_stats:
                            nested_time = function_stats[nested_func][
                                "total_time"
                            ]
                            treemap_data.append(
                                {
                                    "label": f"  {nested_func[:15]}..."
                                    if len(nested_func) > 15
                                    else f"  {nested_func}",
                                    "size": nested_time
                                    * 0.8,  # Scale down to show hierarchy
                                    "color": "lightcoral",
                                    "alpha": 0.6,
                                }
                            )

                return treemap_data

            treemap_data = build_treemap_data()

            if treemap_data:
                fig, ax = plt.subplots(figsize=(16, 10))

                sizes = [item["size"] for item in treemap_data]
                labels = [
                    f"{item['label']}\n{item['size']:.1f}ms"
                    for item in treemap_data
                ]
                colors = [item["color"] for item in treemap_data]

                # Create treemap
                squarify.plot(
                    sizes=sizes, label=labels, color=colors, alpha=0.7, ax=ax
                )

                ax.set_title(
                    "Function Call Hierarchy (Treemap View)",
                    fontsize=16,
                    pad=20,
                )
                ax.axis("off")

                plt.tight_layout()
                p = create_output_path(
                    output_path,
                    prefix,
                    "16_function_hierarchy_treemap.png",
                    out_dir,
                )
                plt.savefig(p, dpi=200, bbox_inches="tight")
                plots_created.append(p)
                if show_plot:
                    plt.show()
                plt.close()

        except ImportError:
            # squarify not available, skip this plot
            pass

    # 17) Hierarchical function timing breakdown (waterfall chart)
    if nesting_db and function_stats:
        # Find a good example function with multiple levels of nesting
        best_func = None
        max_nested_count = 0

        for func_name in function_stats.keys():
            if func_name in nesting_db:
                nested_count = len(
                    nesting_db[func_name].get("nested_functions", [])
                )
                if nested_count > max_nested_count:
                    max_nested_count = nested_count
                    best_func = func_name

        if best_func and max_nested_count > 0:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Build waterfall data
            func_time = function_stats[best_func]["total_time"]
            nested_funcs = nesting_db[best_func].get("nested_functions", [])

            # Calculate exclusive time
            nested_time = sum(
                function_stats.get(nf, {"total_time": 0})["total_time"]
                for nf in nested_funcs
                if nf in function_stats
            )
            exclusive_time = max(0, func_time - nested_time)

            # Prepare data for waterfall
            categories = [f"{best_func}\n(Total)"]
            values = [func_time]
            colors = ["steelblue"]

            categories.append(f"{best_func}\n(Exclusive)")
            values.append(-exclusive_time)
            colors.append("lightcoral")

            # Add nested functions
            for nf in sorted(
                nested_funcs,
                key=lambda x: function_stats.get(x, {"total_time": 0})[
                    "total_time"
                ],
                reverse=True,
            )[:6]:  # Top 6 nested
                if nf in function_stats:
                    categories.append(
                        f"{nf[:15]}...\n(Nested)"
                        if len(nf) > 15
                        else f"{nf}\n(Nested)"
                    )
                    values.append(-function_stats[nf]["total_time"])
                    colors.append("orange")

            # Create waterfall chart
            cumulative = []
            running_total = 0
            for i, val in enumerate(values):
                if i == 0:
                    cumulative.append(val)
                    running_total = val
                else:
                    cumulative.append(running_total)
                    running_total += val

            # Plot bars
            for i, (cat, val, color, cum) in enumerate(
                zip(categories, values, colors, cumulative)
            ):
                if i == 0:
                    ax.bar(i, val, color=color, alpha=0.8)
                else:
                    ax.bar(
                        i,
                        abs(val),
                        bottom=cum - abs(val),
                        color=color,
                        alpha=0.8,
                    )

                # Add value labels
                if i == 0:
                    ax.text(
                        i,
                        val / 2,
                        f"{val:.1f}ms",
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        i,
                        cum - abs(val) / 2,
                        f"{abs(val):.1f}ms",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.set_ylabel("Time (ms)")
            ax.set_title(
                f"Hierarchical Timing Breakdown: {best_func}",
                fontsize=14,
                pad=20,
            )
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            p = create_output_path(
                output_path,
                prefix,
                "17_hierarchical_timing_breakdown.png",
                out_dir,
            )
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    # 18) Function nesting depth analysis
    if nesting_db and function_stats:

        def calculate_nesting_depth(func_name, visited=None):
            if visited is None:
                visited = set()
            if func_name in visited:
                return 0  # Avoid cycles
            visited.add(func_name)

            if func_name not in nesting_db:
                return 0

            nested_funcs = nesting_db[func_name].get("nested_functions", [])
            if not nested_funcs:
                return 0

            max_depth = 0
            for nested_func in nested_funcs:
                depth = calculate_nesting_depth(nested_func, visited.copy())
                max_depth = max(max_depth, depth + 1)

            return max_depth

        # Calculate depths and group functions
        depth_data = {}
        for func_name in function_stats.keys():
            if func_name in nesting_db:
                depth = calculate_nesting_depth(func_name)
                if depth not in depth_data:
                    depth_data[depth] = []
                depth_data[depth].append(
                    {
                        "name": func_name,
                        "time": function_stats[func_name]["total_time"],
                        "calls": function_stats[func_name]["call_count"],
                    }
                )

        if depth_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Left plot: Total time by depth
            depths = sorted(depth_data.keys())
            depth_times = []
            depth_counts = []

            for depth in depths:
                total_time = sum(item["time"] for item in depth_data[depth])
                count = len(depth_data[depth])
                depth_times.append(total_time)
                depth_counts.append(count)

            bars1 = ax1.bar(depths, depth_times, alpha=0.8, color="steelblue")
            ax1.set_xlabel("Nesting Depth")
            ax1.set_ylabel("Total Time (ms)")
            ax1.set_title("Total Execution Time by Nesting Depth")
            ax1.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, time in zip(bars1, depth_times):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(depth_times) * 0.01,
                    f"{time:.1f}ms",
                    ha="center",
                    va="bottom",
                )

            # Right plot: Function count by depth
            bars2 = ax2.bar(
                depths, depth_counts, alpha=0.8, color="lightcoral"
            )
            ax2.set_xlabel("Nesting Depth")
            ax2.set_ylabel("Number of Functions")
            ax2.set_title("Function Count by Nesting Depth")
            ax2.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, count in zip(bars2, depth_counts):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(depth_counts) * 0.01,
                    f"{count}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "18_nesting_depth_analysis.png", out_dir
            )
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    print(f"\nCreated {len(plots_created)} plots:")
    for p in plots_created:
        print(f"  - {p}")

    # Totals
    total_function = sum(v["total_time"] for v in function_stats.values())
    total_operation = sum(v["total_time"] for v in operation_stats.values())
    total_all = total_function + total_operation

    # A) Top function timers table with operation breakdown
    headers = [
        "Function Timer",
        "Total Time (ms)",
        "% of All Functions",
        "Calls",
        "Avg/call (ms)",
        "Operations Sum (ms)",
        "% Accounted",
    ]
    rows = []
    for tid, s in sorted_functions[: min(top_n, 25)]:
        total_time = s["total_time"]
        calls = s["call_count"]
        nm = display_name(tid)
        if tid.startswith("SYNTHETIC:"):
            function_name = tid[10:]  # Remove "SYNTHETIC:" prefix
        else:
            function_name = timer_db[tid].function

        # Sum operation timers for this function
        operation_sum = 0.0
        for op_tid, op_s in operation_stats.items():
            if timer_db[op_tid].function == function_name:
                operation_sum += op_s["total_time"]

        # Calculate what percentage of function time is accounted for by
        # operations
        accounted_pct = (
            (100 * operation_sum / total_time) if total_time > 0 else 0.0
        )

        rows.append(
            [
                (nm if len(nm) <= 60 else nm[:60] + "..."),
                f"{total_time:.1f}",
                f"{(100 * total_time / total_function):.1f}%",
                f"{calls}",
                f"{(total_time / calls if calls > 0 else 0):.2f}",
                f"{operation_sum:.1f}",
                f"{accounted_pct:.1f}%",
            ]
        )
    print(
        "\n"
        + create_ascii_table(
            headers, rows, "TOP FUNCTION TIMERS (with operation breakdown)"
        )
    )

    # B) Timers grouped by function using nesting relationships
    print("\n" + "=" * 100)
    print("TIMERS BY FUNCTION (using nesting relationships)")
    print("=" * 100)

    def get_nested_timers_for_function(
        func_name,
        all_stats_dict,
        timer_db_dict,
        nesting_db_dict,
        visited=None,
    ):
        """Get function timer.

        This function recursively collects all nested timers to any depth.

        Args:
            func_name (str): The name of the function to get timers for.
            all_stats_dict (dict): Dictionary of all timer stats.
            timer_db_dict (dict): Dictionary of all timer metadata.
            nesting_db_dict (dict): Dictionary of nesting relationships.
            visited (set): Set of already visited function names to prevent
               cycles.

        Returns:
            list: List of (timer_id, stats) tuples for the function and its
                  nested timers.
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if func_name in visited:
            return []

        visited.add(func_name)
        nested_timers = []

        # Find the function timer for this function
        function_timer = None
        for tid, stats in all_stats_dict.items():
            if tid.startswith("SYNTHETIC:"):
                # Handle synthetic timer IDs
                if tid[10:] == func_name:  # SYNTHETIC:func_name
                    function_timer = (tid, stats)
                    break
            elif (
                timer_db_dict[tid].function == func_name
                and timer_db_dict[tid].timer_type == "function"
            ):
                function_timer = (tid, stats)
                break

        if function_timer:
            nested_timers.append(function_timer)

        # Find operation timers that belong to this function according to
        # nesting DB
        if func_name in nesting_db_dict:
            nested_operations = nesting_db_dict[func_name].get(
                "nested_operations", []
            )

            for tid, stats in all_stats_dict.items():
                if tid.startswith("SYNTHETIC:"):
                    continue  # Skip synthetic timers in operation lookup
                if (
                    timer_db_dict[tid].function == func_name
                    and timer_db_dict[tid].timer_type == "operation"
                ):
                    # Check if this operation timer's description matches any
                    # in the nesting DB
                    timer_desc = timer_db_dict[tid].label_text
                    for nested_op_pattern in nested_operations:
                        # Remove the format specifiers to match descriptions
                        expected_desc = (
                            nested_op_pattern.replace("%.3f %s.", "")
                            .replace("%.3f %s", "")
                            .strip()
                        )
                        if (
                            expected_desc in timer_desc
                            or timer_desc.startswith(expected_desc)
                        ):
                            nested_timers.append((tid, stats))
                            break

            # Find nested function timers that belong to this function
            # according to nesting DB
            nested_functions = nesting_db_dict[func_name].get(
                "nested_functions", []
            )

            for nested_func_name in nested_functions:
                # Recursively get all timers for the nested function
                recursive_timers = get_nested_timers_for_function(
                    nested_func_name,
                    all_stats_dict,
                    timer_db_dict,
                    nesting_db_dict,
                    visited.copy(),
                )
                nested_timers.extend(recursive_timers)

        return nested_timers

    def build_function_hierarchy(
        func_name, all_stats_dict, timer_db_dict, nesting_db_dict, visited=None
    ):
        """Build a hierarchical structure for a function."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return {"function": None, "operations": [], "nested_functions": {}}

        visited.add(func_name)

        hierarchy = {
            "function": None,
            "operations": [],
            "nested_functions": {},
        }

        # Find the function timer (including synthetic ones)
        for tid, stats in all_stats_dict.items():
            if tid.startswith("SYNTHETIC:"):
                # Handle synthetic timer IDs
                if tid[10:] == func_name:  # SYNTHETIC:func_name
                    hierarchy["function"] = (tid, stats)
                    break
            elif (
                timer_db_dict[tid].function == func_name
                and timer_db_dict[tid].timer_type == "function"
            ):
                hierarchy["function"] = (tid, stats)
                break

        # Find operation timers belonging DIRECTLY to this function
        # (not recursively)
        if func_name in nesting_db_dict:
            nested_operations = nesting_db_dict[func_name].get(
                "nested_operations", []
            )

            for tid, stats in all_stats_dict.items():
                if tid.startswith("SYNTHETIC:"):
                    continue  # Skip synthetic timers in operation lookup
                if (
                    timer_db_dict[tid].function == func_name
                    and timer_db_dict[tid].timer_type == "operation"
                ):
                    timer_desc = timer_db_dict[tid].label_text
                    for nested_op_pattern in nested_operations:
                        expected_desc = (
                            nested_op_pattern.replace("%.3f %s.", "")
                            .replace("%.3f %s", "")
                            .strip()
                        )
                        if (
                            expected_desc in timer_desc
                            or timer_desc.startswith(expected_desc)
                        ):
                            hierarchy["operations"].append((tid, stats))
                            break

            # Build nested function hierarchies (using shared visited set to
            # prevent duplication)
            nested_functions = nesting_db_dict[func_name].get(
                "nested_functions", []
            )
            for nested_func_name in nested_functions:
                nested_hierarchy = build_function_hierarchy(
                    nested_func_name,
                    all_stats_dict,
                    timer_db_dict,
                    nesting_db_dict,
                    visited,
                )
                if (
                    nested_hierarchy["function"]
                    or nested_hierarchy["operations"]
                    or nested_hierarchy["nested_functions"]
                ):
                    hierarchy["nested_functions"][nested_func_name] = (
                        nested_hierarchy
                    )

        return hierarchy

    def add_hierarchical_rows(
        rows,
        hierarchy,
        function_timer_time,
        timer_db_dict,
        indent,
        all_stats_dict,
    ):
        """Add rows to the table in hierarchical order."""
        # Combine operations and nested functions, then sort by time
        all_items = []

        # Add operations from this level
        for tid, stats in hierarchy["operations"]:
            total_time = stats["total_time"]
            calls = stats["call_count"]
            pct_of_func = (
                (100 * total_time / function_timer_time)
                if function_timer_time > 0
                else 0.0
            )

            description = timer_db_dict[tid].label_text
            if " took " in description:
                description = description.split(" took ")[0]
            description = f"{indent}└─ {description}"

            row = [
                description[:60] + ("..." if len(description) > 60 else ""),
                f"{total_time:.1f}",
                f"{pct_of_func:.1f}%",
                f"{calls}",
                f"{(total_time / calls if calls > 0 else 0):.2f}",
            ]
            all_items.append((total_time, "operation", row, None, None))

        # Add nested functions
        for nested_func_name, nested_hierarchy in hierarchy[
            "nested_functions"
        ].items():
            if nested_hierarchy["function"]:
                _, func_stats = nested_hierarchy["function"]
                total_time = func_stats["total_time"]
                calls = func_stats["call_count"]
                pct_of_func = (
                    (100 * total_time / function_timer_time)
                    if function_timer_time > 0
                    else 0.0
                )

                description = (
                    f"{indent}└─ {nested_func_name} (nested function)"
                )

                row = [
                    description[:60]
                    + ("..." if len(description) > 60 else ""),
                    f"{total_time:.1f}",
                    f"{pct_of_func:.1f}%",
                    f"{calls}",
                    f"{(total_time / calls if calls > 0 else 0):.2f}",
                ]
                all_items.append(
                    (
                        total_time,
                        "function",
                        row,
                        nested_func_name,
                        nested_hierarchy,
                    )
                )
            else:
                # If no function timer, but we have operations, create a
                # synthetic function entry
                if nested_hierarchy["operations"]:
                    total_ops_time = sum(
                        stats["total_time"]
                        for _, stats in nested_hierarchy["operations"]
                    )
                    # Use the call count from the first operation as an
                    # approximation
                    calls = (
                        nested_hierarchy["operations"][0][1]["call_count"]
                        if nested_hierarchy["operations"]
                        else 1
                    )
                    pct_of_func = (
                        (100 * total_ops_time / function_timer_time)
                        if function_timer_time > 0
                        else 0.0
                    )

                    description = (
                        f"{indent}└─ {nested_func_name} (nested function)"
                    )

                    row = [
                        description[:60]
                        + ("..." if len(description) > 60 else ""),
                        f"{total_ops_time:.1f}",
                        f"{pct_of_func:.1f}%",
                        f"{calls}",
                        f"{(total_ops_time / calls if calls > 0 else 0):.2f}",
                    ]
                    all_items.append(
                        (
                            total_ops_time,
                            "function",
                            row,
                            nested_func_name,
                            nested_hierarchy,
                        )
                    )
                else:
                    # If no function timer and no operations, skip completely
                    continue

        # Sort all items by time descending
        all_items.sort(key=lambda x: x[0], reverse=True)

        # Add items to rows in sorted order
        for (
            total_time,
            item_type,
            row,
            nested_func_name,
            nested_hierarchy,
        ) in all_items:
            if row is not None:
                rows.append(row)

            # If this is a nested function, recursively add its content
            if item_type == "function" and nested_hierarchy is not None:
                add_hierarchical_rows(
                    rows,
                    nested_hierarchy,
                    function_timer_time,
                    timer_db_dict,
                    indent + "   ",
                    all_stats_dict,
                )

    # Calculate total time per function including only nested timers
    func_totals = {}
    for func_name in nesting_db.keys():
        nested_timers = get_nested_timers_for_function(
            func_name, all_stats, timer_db, nesting_db
        )
        if nested_timers:
            func_totals[func_name] = sum(
                s["total_time"] for _, s in nested_timers
            )

    # Also include functions that have timers but no nesting info
    for tid, s in all_stats.items():
        if tid.startswith("SYNTHETIC:"):
            continue  # Skip synthetic timers in hierarchical analysis
        func_name = timer_db[tid].function
        if func_name not in func_totals:
            if func_name not in nesting_db:  # Only if not in nesting DB
                if func_name not in func_totals:
                    func_totals[func_name] = 0
                func_totals[func_name] += s["total_time"]

    sorted_funcs = sorted(
        func_totals.items(), key=lambda x: x[1], reverse=True
    )

    for func_name, func_total in sorted_funcs[
        : min(top_n, 10)
    ]:  # Show top functions
        # Build hierarchical timer structure for this function (non-recursive
        # for display)
        if func_name in nesting_db:
            hierarchy = build_function_hierarchy(
                func_name, all_stats, timer_db, nesting_db
            )
        else:
            # Fall back to all timers in this function if no nesting info
            timers = [
                (tid, s)
                for tid, s in all_stats.items()
                if timer_db[tid].function == func_name
            ]
            hierarchy = {
                "function": None,
                "operations": [],
                "nested_functions": {},
            }
            for tid, s in timers:
                if timer_db[tid].timer_type == "function":
                    hierarchy["function"] = (tid, s)
                else:
                    hierarchy["operations"].append((tid, s))

        # Skip if no timers found for this function at all
        if (
            not hierarchy["function"]
            and not hierarchy["operations"]
            and not hierarchy["nested_functions"]
        ):
            continue

        # Find the function timer (baseline for 100%)
        function_timer_time = 0.0
        if hierarchy["function"]:
            _, func_stats = hierarchy["function"]
            function_timer_time = func_stats["total_time"]
        else:
            # If no function timer, skip (we need a baseline)
            continue

        if function_timer_time == 0.0:
            continue  # Skip if no function timer found

        print(
            f"\n{func_name}: {function_timer_time:.1f} ms "
            "(function execution time)"
        )
        print("-" * (len(func_name) + 40))

        headers = [
            "Timer Description",
            "Time (ms)",
            "% of Function Time",
            "Calls",
            "Avg/call (ms)",
        ]
        rows = []

        # First add the function timer itself (always 100%)
        if hierarchy["function"]:
            tid, func_stats = hierarchy["function"]
            function_label = "Function execution time"
            if tid.startswith("SYNTHETIC:"):
                function_label = "Function execution time (sum of operations)"
            pcent = func_stats["total_time"] / func_stats["call_count"]
            rows.append(
                [
                    function_label,
                    f"{func_stats['total_time']:.1f}",
                    "100.0%",
                    f"{func_stats['call_count']}",
                    f"{pcent:.2f}",
                ]
            )

        # Add hierarchical rows with proper ordering
        add_hierarchical_rows(
            rows, hierarchy, function_timer_time, timer_db, "", all_stats
        )

        if len(rows) > 1:  # Only show if there are nested items
            print(create_ascii_table(headers, rows, ""))

    # C) Top timers by **call count** (table)
    headers = [
        "Timer (function [file:line])",
        "Call Count",
        "Total Time (ms)",
        "Type",
        "Avg/call (ms)",
    ]
    rows = []
    for tid, _ in sorted_by_calls[: min(top_n, 25)]:
        calls = all_stats.get(tid, {}).get("call_count", 0)
        total_time = all_stats.get(tid, {}).get("total_time", 0.0)
        timer_type = timer_db[tid].timer_type if tid in timer_db else "unknown"
        nm = display_name(tid)
        rows.append(
            [
                (nm if len(nm) <= 60 else nm[:60] + "..."),
                f"{calls}",
                f"{total_time:.1f}",
                timer_type,
                f"{(total_time / calls if calls > 0 else 0):.2f}",
            ]
        )
    print("\n" + create_ascii_table(headers, rows, "TOP TIMERS BY CALL COUNT"))

    # C) Task category summary (if present)
    if task_category_times:
        headers = [
            "Task Category",
            "Total Time (ms)",
            "% of Category Time",
            "Avg/Step (ms)",
            "Steps",
        ]
        rows = []
        cat_total_time = sum(
            sum(vals) for vals in task_category_times.values()
        )
        for cat, times in sorted(
            task_category_times.items(), key=lambda x: sum(x[1]), reverse=True
        ):
            if times:
                total_cat = sum(times)
                avg = total_cat / len(times)
                pct = (
                    (100 * total_cat / cat_total_time)
                    if cat_total_time > 0
                    else 0.0
                )
                rows.append(
                    [
                        (cat if len(cat) <= 30 else cat[:30] + "..."),
                        f"{total_cat:.1f}",
                        f"{pct:.1f}%",
                        f"{avg:.1f}",
                        f"{len(times)}",
                    ]
                )
        print(
            "\n" + create_ascii_table(headers, rows, "TASK CATEGORY SUMMARY")
        )

    # D) Task count summary (if present)
    if task_counts:
        headers = ["Task Type", "Total Count", "Max Count", "Avg/Step"]
        rows = []
        for tname, counts in sorted(
            task_counts.items(), key=lambda x: sum(x[1]), reverse=True
        ):
            if counts:
                tot = sum(counts)
                mx = max(counts)
                avg = tot / len(counts)
                if tot > 0:
                    rows.append(
                        [
                            (
                                tname
                                if len(tname) <= 25
                                else tname[:25] + "..."
                            ),
                            f"{tot}",
                            f"{mx}",
                            f"{avg:.1f}",
                        ]
                    )
        print("\n" + create_ascii_table(headers, rows, "TASK COUNT SUMMARY"))

    # E) Overall summary
    print("\nPERFORMANCE SUMMARY")
    print("-" * 100)
    print(f"Total function timer time:              {total_function:.1f} ms")
    print(f"Total operation timer time:             {total_operation:.1f} ms")
    print(f"Total all timer time:                   {total_all:.1f} ms")
    print(f"Function timers:                        {len(function_stats)}")
    print(f"Operation timers:                       {len(operation_stats)}")
    print(f"Total unique timers:                    {len(all_stats)}")
    print(
        "Total timer call instances:             "
        f"{sum(s['call_count'] for s in all_stats.values())}"
    )
    print(f"Steps processed (with any info):        {len(step_info)}")

    # Top-k coverage
    if len(sorted_all) >= 5:
        top_5 = sum(s["total_time"] for _, s in sorted_all[:5])
        pct = (100 * top_5 / total_all) if total_all > 0 else 0.0
        print(
            f"Top 5 timers account for:              {pct:.1f}% of total time"
        )
    if len(sorted_all) >= 10:
        top_10 = sum(s["total_time"] for _, s in sorted_all[:10])
        pct = (100 * top_10 / total_all) if total_all > 0 else 0.0
        print(
            f"Top 10 timers account for:             {pct:.1f}% of total time"
        )

    # Type breakdown
    if total_all > 0:
        print(
            f"Function timer share:                   "
            f"{100 * total_function / total_all:.1f}%"
        )
        print(
            f"Operation timer share:                  "
            f"{100 * total_operation / total_all:.1f}%"
        )
