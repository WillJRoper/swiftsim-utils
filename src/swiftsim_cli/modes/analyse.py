"""Analyse mode for analysing SWIFT runs."""

import argparse
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.src_parser import (
    compile_site_patterns,
    generate_timer_nesting_database,
    load_timer_db,
    scan_log_instances_by_step,
)
from swiftsim_cli.utilities import create_ascii_table, create_output_path


def classify_timers_by_max_time(
    instances_by_step: Dict, timer_db: Dict, nesting_db: Dict
) -> set:
    """Classify timers using max time per function.

    For each function, the timer with the most total time is treated as the
    function timer. All others are operation timers. No synthetic timers.

    Args:
        instances_by_step: Dictionary mapping step numbers to timer instances
        timer_db: Dictionary of timer definitions by timer ID
        nesting_db: Dictionary of nesting relationships by function name

    Returns:
        Set of timer IDs that are function timers
    """
    function_timer_ids = set()

    # Calculate total time per timer
    timer_totals = defaultdict(float)
    for inst_list in instances_by_step.values():
        for inst in inst_list:
            timer_totals[inst.timer_id] += inst.time_ms

    # Group timers by function
    timers_by_function = defaultdict(list)
    for tid, total_time in timer_totals.items():
        if tid in timer_db:
            func_name = timer_db[tid].function
            timers_by_function[func_name].append((tid, total_time))

    # For each function, select the timer with max time as function timer
    for func_name, timer_list in timers_by_function.items():
        if timer_list:
            # Sort by total time and pick the highest
            timer_list.sort(key=lambda x: x[1], reverse=True)
            max_timer_id = timer_list[0][0]
            function_timer_ids.add(max_timer_id)

    # Update timer type classification in timer_db for analysis
    for tid, timer_def in timer_db.items():
        if tid in function_timer_ids:
            timer_def.timer_type = "function"
        else:
            timer_def.timer_type = "operation"

    return function_timer_ids


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
        default=None,
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
        "--hierarchy-functions",
        nargs="*",
        help="Functions to show hierarchical timing for. If not specified, "
        "shows default important functions: engine_rebuild, space_rebuild, "
        "and engine_maketasks. Use 'all' to show all functions "
        "with hierarchy data, or specify individual function names.",
        default=None,
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
        hierarchy_functions=args.hierarchy_functions,
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
    hierarchy_functions: list[str] | None = None,
) -> None:
    """Analyse SWIFT timing logs and emit comprehensive timing reports.

    This function provides advanced timing analysis of SWIFT simulation logs,
    featuring hierarchical timing breakdowns that avoid double-counting and
    provide clear visibility into function performance.

    Key Features:
      * Automatically generates timer nesting database from SWIFT source code
      * Parses log files to extract timer instances and step information
      * Creates hierarchical timing tables showing function call trees
      * Implements function timer = max(explicit_timer, sum_of_nested) rule
      * Shows "UNACCOUNTED TIME" for missing instrumentation
      * Produces standard plots: top timers, distributions, task counts, etc.
      * Estimates per-step untimed overhead when step totals are available

    Hierarchical Analysis:
      * Functions are analyzed using nesting relationships from source code
      * Direct operations and nested function calls are properly accounted
      * Percentages are calculated against corrected function timer totals
      * Tables show proper indentation and avoid double-counting issues

    Notes:
      * Nesting database is regenerated automatically from SWIFT source code
      * Display names use `function [file:line]` format for traceability
      * Function timers use the maximum of explicit timers and nested sums

    Args:
      log_file: Path to the SWIFT log file to analyse.
      output_path: Directory where figures are saved. If None, saves to CWD.
      prefix: Optional filename prefix and output subdirectory prefix.
      show_plot: Whether to display plots interactively.
      top_n: Number of top timers shown in multi-item plots/tables.
      hierarchy_functions: Functions to show hierarchical timing for. If None,
        shows default important functions. Use ["all"] to show all functions.

    Returns:
      None. Figures are written to disk; a textual summary is printed to
            stdout.
    """

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
    nesting_db = load_timer_nesting(auto_generate=True, force_regenerate=True)
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

    # Classify timers: timer with max time per function becomes function timer
    print(
        "Classifying timers: max time per function becomes function timer..."
    )
    function_timer_ids = classify_timers_by_max_time(
        instances_by_step, timer_db, nesting_db
    )
    print(f"Identified {len(function_timer_ids)} function timers")

    # Update timer instances with the new classification
    for inst_list in instances_by_step.values():
        for inst in inst_list:
            if inst.timer_id in function_timer_ids:
                inst.timer_type = "function"
            else:
                inst.timer_type = "operation"

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

    # Note: No synthetic timers - only use actual function timers from the log
    operation_stats = {
        tid: build_stats(vals) for tid, vals in operation_times_by_tid.items()
    }

    # Sort by total time for each timer type
    sorted_functions = sorted(
        function_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
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

    # Helper function for clean labels
    def clean_label(s: str, max_len: int = 30) -> str:
        """Clean and truncate labels for better readability."""
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    # Helper function to avoid overlapping labels
    def avoid_label_overlap(ax, labels, values, orientation="horizontal"):
        """Adjust label positions to avoid overlap."""
        if orientation == "horizontal":
            # For horizontal bar charts, adjust y-tick label font size
            ax.tick_params(axis="y", labelsize=8)
        else:
            # For vertical plots, rotate labels
            ax.tick_params(axis="x", rotation=45, labelsize=8)

    # Helper: light label truncation (legacy)
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

    # 1) Clean function timing bar chart (single panel)
    if sorted_functions:
        print("Creating function timing chart...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Take top 15 functions for better readability
        top_funcs = sorted_functions[:15]
        names = []
        times = []

        for tid, stats in top_funcs:
            # Clean up function names for better display
            # Handle both regular and synthetic timer IDs
            if tid in timer_db:
                func_name = timer_db[tid].function
            elif tid.startswith("SYNTHETIC:"):
                func_name = tid.replace("SYNTHETIC:", "")
            else:
                func_name = "Unknown"
            # Remove SYNTHETIC prefix and cleanup
            clean_name = (
                func_name.replace("SYNTHETIC:", "").replace("_", " ").title()
            )
            names.append(clean_name)
            times.append(stats["total_time"])

        # Use a professional color gradient
        colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(names)))

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(names)),
            times,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Clean formatting
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel("Execution Time (ms)", fontsize=12)
        ax.set_title(
            "Top Function Execution Times",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            # Format time nicely
            if time >= 1000:
                label = f"{time / 1000:.1f}s"
            else:
                label = f"{time:.0f}ms"

            ax.text(
                bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, axis="x", linestyle="--")

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "01_function_timing.png", out_dir
        )
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 2) Improved pie chart (single panel)
    if sorted_functions:
        print("Creating function distribution chart...")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Take top 6 + others for better readability
        top_items = sorted_functions[:6]
        other_time = sum(st["total_time"] for _, st in sorted_functions[6:])

        names = []
        times = []

        for tid, stats in top_items:
            # Handle both regular and synthetic timer IDs
            if tid in timer_db:
                func_name = timer_db[tid].function
            elif tid.startswith("SYNTHETIC:"):
                func_name = tid.replace("SYNTHETIC:", "")
            else:
                func_name = "Unknown"
            clean_name = (
                func_name.replace("SYNTHETIC:", "").replace("_", " ").title()
            )
            names.append(clean_name[:20])  # Limit length
            times.append(stats["total_time"])

        if other_time > 0:
            names.append("Other Functions")
            times.append(other_time)

        # Use professional colors
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

        # Create pie chart with clear separation
        wedges, texts, autotexts = ax.pie(
            times,
            labels=names,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 11, "fontweight": "bold"},
            pctdistance=0.85,
            labeldistance=1.1,
        )

        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        ax.set_title(
            "Function Execution Time Distribution",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "02_function_distribution.png", out_dir
        )
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 3) Function efficiency chart (single panel)
    if sorted_functions:
        print("Creating function efficiency chart...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate efficiency (time per call)
        efficiency_data = []
        for tid, stats in sorted_functions[:20]:
            if stats["call_count"] > 0:
                # Handle both regular and synthetic timer IDs
                if tid in timer_db:
                    func_name = timer_db[tid].function
                elif tid.startswith("SYNTHETIC:"):
                    func_name = tid.replace("SYNTHETIC:", "")
                else:
                    func_name = "Unknown"
                clean_name = (
                    func_name.replace("SYNTHETIC:", "")
                    .replace("_", " ")
                    .title()
                )
                efficiency = stats["total_time"] / stats["call_count"]
                efficiency_data.append(
                    (clean_name, efficiency, stats["call_count"])
                )

        # Sort by efficiency
        efficiency_data.sort(key=lambda x: x[1], reverse=True)
        efficiency_data = efficiency_data[:15]  # Top 15

        names = [item[0] for item in efficiency_data]
        efficiencies = [item[1] for item in efficiency_data]
        call_counts = [item[2] for item in efficiency_data]

        # Color by call count
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))

        bars = ax.barh(
            range(len(names)),
            efficiencies,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel("Average Time per Call (ms)", fontsize=12)
        ax.set_title(
            "Function Efficiency Analysis",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add efficiency labels
        for i, (bar, eff, calls) in enumerate(
            zip(bars, efficiencies, call_counts)
        ):
            if eff >= 1:
                label = f"{eff:.1f}ms ({calls} calls)"
            else:
                label = f"{eff:.2f}ms ({calls} calls)"
            ax.text(
                bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, axis="x", linestyle="--")

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "03_function_efficiency.png", out_dir
        )
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 4) Timer distribution histogram (single panel)
    print("Creating timer distribution analysis...")

    all_times = []
    for stats in all_stats.values():
        if stats["total_time"] > 0:
            all_times.append(stats["total_time"])

    if all_times:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create histogram with better binning
        bins = np.logspace(
            np.log10(min(all_times)), np.log10(max(all_times)), 30
        )
        counts, bins, patches = ax.hist(
            all_times,
            bins=bins,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )

        # Color gradient for histogram
        colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        ax.set_xlabel("Timer Duration (ms)", fontsize=12)
        ax.set_ylabel("Number of Timers", fontsize=12)
        ax.set_title(
            "Distribution of Timer Execution Times",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xscale("log")

        # Add statistics text
        mean_time = np.mean(all_times)
        median_time = np.median(all_times)
        stats_text = (
            f"Mean: {mean_time:.1f}ms\n"
            f"Median: {median_time:.1f}ms\n"
            f"Total Timers: {len(all_times)}"
        )
        ax.text(
            0.7,
            0.8,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "04_timer_distribution.png", out_dir
        )
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 5) Task category overview (single panel)
    if task_category_times:
        print("Creating task category overview...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate totals and get top categories
        cat_totals = {
            cat: sum(times)
            for cat, times in task_category_times.items()
            if times
        }

        if cat_totals:
            # Sort and take meaningful categories
            items = sorted(
                cat_totals.items(), key=lambda x: x[1], reverse=True
            )
            # Filter out very small categories
            total_time = sum(cat_totals.values())
            items = [
                (cat, time) for cat, time in items if time > total_time * 0.001
            ][:8]

            categories = [item[0].replace("_", " ").title() for item in items]
            times = [item[1] for item in items]

            # Professional color scheme
            colors = plt.cm.Dark2(np.linspace(0, 1, len(categories)))

            bars = ax.bar(
                range(len(categories)),
                times,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
            )

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(
                categories, rotation=45, ha="right", fontsize=11
            )
            ax.set_ylabel("Total Time (ms)", fontsize=12)
            ax.set_title(
                "Task Category Performance Overview",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                if time >= 1000:
                    label = f"{time / 1000:.1f}s"
                else:
                    label = f"{time:.0f}ms"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + height * 0.01,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")

            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "05_task_categories.png", out_dir
            )
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    # 6) Time series plots showing function evolution over steps (single panel)
    print("Creating time series analysis...")

    # Collect time series data for top functions
    steps_with_data = sorted(
        [k for k in instances_by_step.keys() if k is not None]
    )
    if len(steps_with_data) > 1 and function_stats:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get top 8 functions by total time for time series
        top_functions = sorted(
            function_stats.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True,
        )[:8]

        colors = plt.cm.Set1(np.linspace(0, 1, len(top_functions)))

        for i, (tid, stats) in enumerate(top_functions):
            # Collect execution times per step for this function
            step_times = []
            step_numbers = []

            for step in steps_with_data:
                # Sum all instances of this timer in this step
                step_total = sum(
                    inst.time_ms
                    for inst in instances_by_step[step]
                    if inst.timer_id == tid
                )
                step_times.append(step_total)
                step_numbers.append(step)

            # Get function name for display
            if tid in timer_db:
                func_name = timer_db[tid].function
                clean_name = func_name.replace("_", " ").title()
            else:
                clean_name = "Unknown Function"

            # Plot the time series
            ax.plot(
                step_numbers,
                step_times,
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=6,
                alpha=0.8,
                label=clean_name,
            )

        ax.set_xlabel("Simulation Step", fontsize=12)
        ax.set_ylabel("Execution Time (ms)", fontsize=12)
        ax.set_title(
            "Function Execution Times Over Simulation Steps",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale(
            "log"
        )  # Log scale for better visibility of different magnitudes

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        p = create_output_path(
            output_path, prefix, "07_time_series.png", out_dir
        )
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plots_created.append(p)
        if show_plot:
            plt.show()
        plt.close()

    # 7) Function timing variability analysis (single panel)
    print("Creating timing variability analysis...")

    if len(steps_with_data) > 2 and function_stats:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate coefficient of variation for each function
        variability_data = []
        for tid, stats in function_stats.items():
            if (
                stats["call_count"] > 2
            ):  # Need multiple calls to calculate variability
                # Get all execution times for this function
                all_times = []
                for step in steps_with_data:
                    step_times = [
                        inst.time_ms
                        for inst in instances_by_step[step]
                        if inst.timer_id == tid
                    ]
                    all_times.extend(step_times)

                if len(all_times) > 2 and np.mean(all_times) > 0:
                    cv = (
                        np.std(all_times) / np.mean(all_times) * 100
                    )  # Coefficient of variation as %
                    func_name = (
                        timer_db[tid].function
                        if tid in timer_db
                        else "Unknown"
                    )
                    clean_name = func_name.replace("_", " ").title()
                    variability_data.append(
                        (clean_name, cv, stats["total_time"])
                    )

        if variability_data:
            # Sort by coefficient of variation
            variability_data.sort(key=lambda x: x[1], reverse=True)
            variability_data = variability_data[:15]  # Top 15 most variable

            names = [item[0] for item in variability_data]
            cvs = [item[1] for item in variability_data]
            total_times = [item[2] for item in variability_data]

            # Color by total execution time
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(names)))

            bars = ax.barh(
                range(len(names)),
                cvs,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=11)
            ax.set_xlabel("Coefficient of Variation (%)", fontsize=12)
            ax.set_title(
                "Function Timing Variability (Higher = More Variable)",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            # Add CV% labels on bars
            for i, (bar, cv, total_time) in enumerate(
                zip(bars, cvs, total_times)
            ):
                label = f"{cv:.1f}%"
                ax.text(
                    bar.get_width() * 1.02,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    fontsize=10,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")

            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "08_timing_variability.png", out_dir
            )
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plots_created.append(p)
            if show_plot:
                plt.show()
            plt.close()

    # 8) Performance summary scatter plot (single panel)
    if function_stats:
        print("Creating performance summary...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for scatter plot
        scatter_data = []
        for tid, stats in function_stats.items():
            if (
                stats["call_count"] > 0 and stats["total_time"] > 1
            ):  # Filter very small timers
                # Handle both regular and synthetic timer IDs
                if tid in timer_db:
                    func_name = timer_db[tid].function
                elif tid.startswith("SYNTHETIC:"):
                    func_name = tid.replace("SYNTHETIC:", "")
                else:
                    func_name = "Unknown"
                avg_duration = stats["total_time"] / stats["call_count"]
                scatter_data.append(
                    {
                        "name": func_name.replace("SYNTHETIC:", "")
                        .replace("_", " ")
                        .title(),
                        "calls": stats["call_count"],
                        "avg_duration": avg_duration,
                        "total_time": stats["total_time"],
                    }
                )

        if scatter_data:
            calls = [item["calls"] for item in scatter_data]
            avg_durations = [item["avg_duration"] for item in scatter_data]
            total_times = [item["total_time"] for item in scatter_data]

            # Create scatter plot with size proportional to total time
            sizes = np.array(total_times)
            sizes = (sizes / sizes.max()) * 300 + 20  # Scale for visibility

            scatter = ax.scatter(
                calls,
                avg_durations,
                s=sizes,
                alpha=0.6,
                c=total_times,
                cmap="viridis",
                edgecolors="white",
                linewidth=0.5,
            )

            ax.set_xlabel("Number of Calls", fontsize=12)
            ax.set_ylabel("Average Duration per Call (ms)", fontsize=12)
            ax.set_title(
                "Function Call Patterns (Size = Total Time)",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(
                "Total Execution Time (ms)",
                rotation=270,
                labelpad=20,
                fontsize=11,
            )

            # Add annotations for outliers
            for item in scatter_data[:5]:  # Top 5 by total time
                ax.annotate(
                    item["name"][:15],
                    (item["calls"], item["avg_duration"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle="--")

            plt.tight_layout()
            p = create_output_path(
                output_path, prefix, "06_performance_summary.png", out_dir
            )
            plt.savefig(p, dpi=300, bbox_inches="tight")
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

        # Calculate unaccounted time (only for top level, when indent is "")
        if indent == "":
            # Sum up only direct children (level 1) - operations from this
            # function plus nested functions
            # This avoids double-counting since nested functions already
            # include their own operations
            accounted_time = 0.0

            # Add direct operations from this function level
            for tid, stats in hierarchy["operations"]:
                accounted_time += stats["total_time"]

            # Add direct nested functions (they already include their own
            # nested content)
            for nested_func_name, nested_hierarchy in hierarchy[
                "nested_functions"
            ].items():
                if nested_hierarchy["function"]:
                    _, func_stats = nested_hierarchy["function"]
                    accounted_time += func_stats["total_time"]

            # Calculate unaccounted time as the difference between total
            # function time and the sum of direct children
            unaccounted_time = function_timer_time - accounted_time
            if unaccounted_time > 0.1:  # Only show if significant (> 0.1 ms)
                pct_unaccounted = (
                    (100 * unaccounted_time / function_timer_time)
                    if function_timer_time > 0
                    else 0.0
                )
                unaccounted_row = [
                    f"{indent}└─ UNACCOUNTED TIME",
                    f"{unaccounted_time:.1f}",
                    f"{pct_unaccounted:.1f}%",
                    "-",
                    "-",
                ]
                # Add unaccounted time to the items list so it gets sorted
                # properly by time
                all_items.append(
                    (
                        unaccounted_time,
                        "unaccounted",
                        unaccounted_row,
                        None,
                        None,
                    )
                )

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

    # Determine which functions to show hierarchically
    if hierarchy_functions is None:
        # Default to showing a few key functions that are typically important
        default_functions = [
            "engine_rebuild",
            "space_rebuild",
            "space_split",
            "engine_maketasks",
        ]
        functions_to_show = [
            (func_name, func_totals.get(func_name, 0))
            for func_name in default_functions
            if func_name in func_totals
        ]
        # Sort by total time
        functions_to_show.sort(key=lambda x: x[1], reverse=True)
    elif len(hierarchy_functions) == 1 and hierarchy_functions[0] == "all":
        # Show all functions with hierarchy data (up to top_n limit)
        functions_to_show = sorted_funcs[: min(top_n, 10)]
    else:
        # Show only the specified functions
        functions_to_show = [
            (func_name, func_totals.get(func_name, 0))
            for func_name in hierarchy_functions
            if func_name in func_totals
        ]
        # Sort by total time
        functions_to_show.sort(key=lambda x: x[1], reverse=True)

    for func_name, func_total in functions_to_show:
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

        # Calculate function timer time using the rule:
        # function_timer = max(explicit_function_timer, sum_of_nested_timers)
        explicit_function_timer_time = 0.0
        if hierarchy["function"]:
            _, func_stats = hierarchy["function"]
            explicit_function_timer_time = func_stats["total_time"]

        # Calculate sum of all nested timers (operations + nested functions)
        nested_sum = 0.0
        for _, stats in hierarchy["operations"]:
            nested_sum += stats["total_time"]

        for nested_func_hierarchy in hierarchy["nested_functions"].values():
            if nested_func_hierarchy["function"]:
                _, nested_func_stats = nested_func_hierarchy["function"]
                nested_sum += nested_func_stats["total_time"]

        # Apply the rule: function timer = max(explicit, sum_of_nested)
        function_timer_time = max(explicit_function_timer_time, nested_sum)

        if function_timer_time == 0.0:
            continue  # Skip if no timer data found

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
            # Use function_timer_time (the corrected max value) instead of
            # func_stats total_time
            pcent = function_timer_time / func_stats["call_count"]
            rows.append(
                [
                    function_label,
                    f"{function_timer_time:.1f}",
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
