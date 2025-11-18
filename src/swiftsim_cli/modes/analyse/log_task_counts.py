"""Task-count analysis module for SWIFT simulations.

This module analyses the `engine_print_task_counts` output in SWIFT logs.

CLI integration:
  * add_task_counts_arguments(subparsers)
  * run_swift_task_counts(args)

Core functionality:
  * analyse_swift_task_counts(log_file, output_path, prefix, show_plot)

The analysis is intentionally light-weight:
  * Uses scan_task_counts_by_step() from swiftsim_cli.src_parser
  * Builds a time series of per-step total task counts (preferring rank 0)
  * Produces:
      - Scatter plot: total tasks vs simulation time
      - Cumulative plot: cumulative tasks vs simulation time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from swiftsim_cli.src_parser import (
    TaskCountSnapshot,
    scan_task_counts_by_step,
)
from swiftsim_cli.utilities import create_output_path


class LogData(TypedDict):
    """Type definition for log file data."""

    log_file: str
    times: NDArray[np.float64]
    totals: NDArray[np.float64]
    cumulative: NDArray[np.float64]
    marker: str


__all__ = [
    "add_task_counts_arguments",
    "run_swift_task_counts",
    "analyse_swift_task_counts",
]


# ============================================================================
# CLI ARGUMENT SETUP
# ============================================================================


def add_task_counts_arguments(subparsers) -> None:
    """Add CLI arguments for engine_print_task_counts analysis.

    Subcommand name: 'task-counts'
    """
    task_parser = subparsers.add_parser(
        "task-counts",
        help=(
            "Analyse engine_print_task_counts output from SWIFT log files. "
            "Produces a per-step task count time series and cumulative plot."
        ),
    )

    task_parser.add_argument(
        "log_files",
        nargs="+",
        help="SWIFT log file(s) to analyse. Multiple files will be "
        "plotted together with different markers.",
        type=Path,
    )

    task_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    task_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="Prefix to add to analysis files and output "
        "directory (default: '').",
        default=None,
    )

    task_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively.",
        default=False,
    )

    task_parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        nargs="+",
        help="List of specific task names to include in the analysis. "
        "Task names must match the task strings in the log "
        "(e.g., 'sort', 'self', 'pair'). If not specified, all tasks "
        "are included.",
        default=None,
    )


def run_swift_task_counts(args: argparse.Namespace) -> None:
    """Entry point for the 'task-counts' CLI subcommand.

    This mirrors run_swift_log_timing() and simply forwards args.
    """
    analyse_swift_task_counts(
        log_files=[str(f) for f in args.log_files],
        output_path=str(args.output_path) if args.output_path else None,
        prefix=args.prefix,
        show_plot=args.show,
        task_filter=args.tasks,
    )


# ============================================================================
# CORE ANALYSIS
# ============================================================================


def analyse_swift_task_counts(
    log_files: list[str],
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    task_filter: list[str] | None = None,
) -> None:
    """Analyse engine_print_task_counts blocks in SWIFT log files.

    This function:
      * Accepts multiple log files for comparison
      * Uses scan_task_counts_by_step() to extract per-step task-count
        snapshots keyed by step number.
      * Collapses snapshots per step to a single series, preferring rank 0.
      * Optionally filters to include only specific task types.
      * Builds:
          - A scatter plot of total tasks vs simulation time.
          - A cumulative total tasks vs simulation time plot.
      * Each log file gets a different marker on the plots.

    Args:
        log_files:
            List of paths to SWIFT log files to analyse.
        output_path:
            Directory where figures are saved. If None, saves to CWD.
        prefix:
            Optional filename and output-subdirectory prefix.
        show_plot:
            Whether to display plots interactively.
        task_filter:
            Optional list of task names to include. If None, all tasks
            are included.
    """
    print(
        f"Analyzing engine_print_task_counts in {len(log_files)} log file(s)"
    )

    if task_filter:
        print(f"Filtering for specific tasks: {', '.join(task_filter)}")

    # Consistent with your other analysis: prefix determines output directory.
    out_dir = (
        "task_counts_analysis"
        if prefix is None
        else f"{prefix}_task_counts_analysis"
    )

    # Define markers for different log files
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # ------------------------------------------------------------------
    # Process each log file
    # ------------------------------------------------------------------
    all_data: list[LogData] = []

    for log_idx, log_file in enumerate(log_files):
        print(f"\nLog {log_idx + 1}/{len(log_files)}: {log_file}")

        # Parse the log
        snapshots_by_step, step_lines = scan_task_counts_by_step(log_file)

        total_snapshots = sum(len(v) for v in snapshots_by_step.values())
        print(
            f"  Found {total_snapshots} snapshots across "
            f"{len(snapshots_by_step)} steps"
        )

        # Build time series
        steps: List[int] = []
        sim_times: List[float] = []
        totals: List[int] = []

        for step in sorted(
            k for k in snapshots_by_step.keys() if k is not None
        ):
            snaps: list[TaskCountSnapshot] = snapshots_by_step[step]
            if not snaps:
                continue

            snap = next((s for s in snaps if s.rank == 0), snaps[0])

            steps.append(step)
            sim_times.append(snap.sim_time)

            # Calculate total tasks
            if task_filter:
                total = sum(
                    snap.counts.get(task_name, 0) for task_name in task_filter
                )
                totals.append(total)
            else:
                if snap.total_tasks is not None:
                    totals.append(int(snap.total_tasks))
                elif snap.system_total is not None:
                    totals.append(int(snap.system_total))
                else:
                    totals.append(0)

        if not steps:
            print("  WARNING: No usable data found in this log")
            continue

        steps_arr = np.asarray(steps, dtype=int)
        times_arr = np.asarray(sim_times, dtype=float)
        totals_arr = np.asarray(totals, dtype=float)
        cumulative_arr = np.cumsum(totals_arr)

        print(
            f"  Prepared time series: {len(steps_arr)} steps "
            f"(step {steps_arr.min()} - {steps_arr.max()})"
        )

        all_data.append(
            {
                "log_file": log_file,
                "times": times_arr,
                "totals": totals_arr,
                "cumulative": cumulative_arr,
                "marker": markers[log_idx % len(markers)],
            }
        )

    if not all_data:
        print("\nNo usable data found in any log files!")
        return

    # ------------------------------------------------------------------
    # Plot 1: Scatter of total tasks vs simulation time
    # ------------------------------------------------------------------
    print("Creating per-step task-count scatter plot...")

    # Build plot titles based on whether filtering is active
    if task_filter:
        task_list = ", ".join(task_filter)
        title_suffix = f" (filtered: {task_list})"
        ylabel_suffix = " - filtered tasks"
    else:
        title_suffix = ""
        ylabel_suffix = ""

    fig, ax = plt.subplots(figsize=(10, 6))
    for data in all_data:
        label = Path(data["log_file"]).name
        ax.scatter(
            data["times"],
            data["totals"],
            marker=data["marker"],
            alpha=0.7,
            label=label,
        )
    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Simulation time")
    ax.set_ylabel(f"Total tasks per step (rank 0 preferred){ylabel_suffix}")
    ax.set_title(f"engine_print_task_counts: per-step totals{title_suffix}")
    ax.grid(True, alpha=0.3, linestyle="--")

    p1 = create_output_path(
        output_path, prefix, "task_counts_per_step.png", out_dir
    )
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Cumulative tasks vs simulation time
    # ------------------------------------------------------------------
    print("Creating cumulative task-count plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    for data in all_data:
        label = Path(data["log_file"]).name
        ax.plot(
            data["times"],
            data["cumulative"],
            marker=data["marker"],
            label=label,
        )
    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Simulation time")
    ax.set_ylabel(f"Cumulative tasks{ylabel_suffix}")
    ax.set_title(
        f"engine_print_task_counts: cumulative total tasks{title_suffix}"
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    p2 = create_output_path(
        output_path, prefix, "task_counts_cumulative.png", out_dir
    )
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    print("\nCreated task-count plots:")
    print(f"  - {p1}")
    print(f"  - {p2}")
