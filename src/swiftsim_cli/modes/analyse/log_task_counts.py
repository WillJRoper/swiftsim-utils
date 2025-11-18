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
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.src_parser import (
    TaskCountSnapshot,
    scan_task_counts_by_step,
)
from swiftsim_cli.utilities import create_output_path

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
        "log_file",
        help="SWIFT log file to analyse.",
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
        log_file=str(args.log_file),
        output_path=str(args.output_path) if args.output_path else None,
        prefix=args.prefix,
        show_plot=args.show,
        task_filter=args.tasks,
    )


# ============================================================================
# CORE ANALYSIS
# ============================================================================


def analyse_swift_task_counts(
    log_file: str,
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    task_filter: list[str] | None = None,
) -> None:
    """Analyse engine_print_task_counts blocks in a SWIFT log.

    This function:
      * Uses scan_task_counts_by_step() to extract per-step task-count
        snapshots keyed by step number.
      * Collapses snapshots per step to a single series, preferring rank 0.
      * Optionally filters to include only specific task types.
      * Builds:
          - A scatter plot of total tasks vs simulation time.
          - A cumulative total tasks vs simulation time plot.

    Args:
        log_file:
            Path to the SWIFT log file to analyse.
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
    print(f"Analyzing engine_print_task_counts in log:  {log_file}")

    if task_filter:
        print(f"Filtering for specific tasks: {', '.join(task_filter)}")

    # Consistent with your other analysis: prefix determines output directory.
    out_dir = (
        "task_counts_analysis"
        if prefix is None
        else f"{prefix}_task_counts_analysis"
    )

    # ------------------------------------------------------------------
    # Parse the log with the fast streaming parser from src_parser
    # ------------------------------------------------------------------
    snapshots_by_step, step_lines = scan_task_counts_by_step(log_file)

    total_snapshots = sum(len(v) for v in snapshots_by_step.values())
    print(
        f"Found {total_snapshots} engine_print_task_counts snapshots "
        f"across {len(snapshots_by_step)} steps (step-lines: "
        f"{len(step_lines)})."
    )

    # ------------------------------------------------------------------
    # Build a time series: simulation time vs total tasks (prefer rank 0)
    # ------------------------------------------------------------------
    steps: List[int] = []
    sim_times: List[float] = []
    totals: List[int] = []

    # Only consider entries with a valid step number
    for step in sorted(k for k in snapshots_by_step.keys() if k is not None):
        snaps: list[TaskCountSnapshot] = snapshots_by_step[step]
        if not snaps:
            continue

        # Prefer rank 0 if present, otherwise fall back to the first snapshot
        snap = next((s for s in snaps if s.rank == 0), snaps[0])

        steps.append(step)
        sim_times.append(snap.sim_time)

        # Calculate total tasks for this step
        if task_filter:
            # Sum only the specified tasks
            total = sum(
                snap.counts.get(task_name, 0) for task_name in task_filter
            )
            totals.append(total)
        else:
            # Use total from snapshot if available
            if snap.total_tasks is not None:
                totals.append(int(snap.total_tasks))
            elif snap.system_total is not None:
                totals.append(int(snap.system_total))
            else:
                totals.append(0)

    if not steps:
        print(
            "No usable engine_print_task_counts blocks with step "
            "numbers found."
        )
        return

    steps_arr = np.asarray(steps, dtype=int)
    times_arr = np.asarray(sim_times, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    cumulative_arr = np.cumsum(totals_arr)

    print(
        f"Prepared time series for {len(steps_arr)} steps "
        f"(min step={steps_arr.min()}, max step={steps_arr.max()})."
    )

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
    ax.scatter(times_arr, totals_arr, alpha=0.7)
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
    ax.plot(times_arr, cumulative_arr, marker="o")
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
