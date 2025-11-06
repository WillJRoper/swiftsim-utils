"""Timestep analysis for SWIFT simulations.

This module analyzes timestep evolution files from SWIFT simulations,
providing insights into simulation dynamics and timestep behavior over time.

Key functions:
- analyse_timestep_files: Main analysis and plotting routine
- run_timestep: CLI entry point for timestep analysis
- add_timestep_arguments: CLI argument setup
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Ensure matplotlib has the tab10 colormap
try:
    tab10 = cm.get_cmap("tab10")
except (AttributeError, ValueError):
    # Fallback if tab10 is not available
    tab10 = cm.get_cmap("Set1")

from swiftsim_cli.utilities import create_output_path


def add_timestep_arguments(subparsers) -> None:
    """Add timestep analysis arguments to the subparser."""
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
        help="Labels for the files (same order as files).",
        type=str,
        default=None,
    )

    timestep_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path for the plot.",
        type=Path,
        default=Path("./timestep_analysis.png"),
    )

    timestep_parser.add_argument(
        "--time-bins",
        "-t",
        help="Number of time bins to use for the timestep histogram.",
        type=int,
        default=100,
    )

    timestep_parser.add_argument(
        "--dt-bins",
        help="Number of timestep bins to use for the timestep histogram.",
        type=int,
        default=100,
    )

    timestep_parser.add_argument(
        "--prefix",
        "-p",
        help="Prefix for the output files.",
        type=str,
        default="",
    )


def run_timestep(args: argparse.Namespace) -> None:
    """Run timestep analysis."""
    analyse_timestep_files(
        args.files,
        args.labels,
        args.output_path,
        args.time_bins,
        args.dt_bins,
        args.prefix,
    )


def analyse_timestep_files(
    files: List[Path],
    labels: List[str] | None = None,
    output_path: Path = Path("./timestep_analysis.png"),
    time_bins: int = 100,
    dt_bins: int = 100,
    prefix: str = "",
) -> None:
    """Analyse a list of timestep files and create comparison plots.

    Creates comprehensive timestep analysis plots including:
    - Timestep evolution over time
    - Timestep distribution histograms
    - Cumulative timestep statistics

    Args:
        files: List of timestep file paths to analyze
        labels: Optional labels for each file (defaults to filenames)
        output_path: Output directory for generated plots
        time_bins: Number of bins for time-based histograms
        dt_bins: Number of bins for timestep histograms
        prefix: Prefix for output filenames

    Raises:
        FileNotFoundError: If any of the input files don't exist
        ValueError: If files and labels lists have different lengths
    """
    if labels is None:
        labels = [file.name for file in files]

    if len(files) != len(labels):
        raise ValueError(
            f"Number of files ({len(files)}) and labels ({len(labels)}) "
            f"must match"
        )

    print(f"Analyzing {len(files)} timestep files...")

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Timestep Analysis", fontsize=16, fontweight="bold")

    colors = tab10(np.linspace(0, 1, len(files)))

    for i, (file, label, color) in enumerate(zip(files, labels, colors)):
        if not file.exists():
            raise FileNotFoundError(f"Timestep file not found: {file}")

        print(f"Processing {file.name}...")

        # Read the timestep file with robust error handling
        times = []
        timesteps = []
        with open(file, "r") as f:
            for line in f:
                # Skip comment lines
                if line.startswith("#"):
                    continue

                # Skip non-data lines (data lines start with whitespace)
                if not line[0].isspace():
                    continue

                # Split the line into parts
                parts = line.split()

                # Try to parse as floats, silently skip unparseable lines
                try:
                    # Convert all parts to floats to validate
                    float_parts = [float(part) for part in parts]

                    # Extract time and timestep (columns 1 and 2)
                    if len(float_parts) >= 3:
                        times.append(float_parts[1])
                        timesteps.append(float_parts[2])
                except (ValueError, IndexError):
                    # Silently skip lines that can't be parsed
                    continue

        # Convert to numpy arrays
        times = np.array(times)
        timesteps = np.array(timesteps)

        # Skip if no valid data found
        if len(times) == 0:
            continue

        # Plot 1: Timestep evolution over time
        axes[0, 0].plot(times, timesteps, label=label, color=color, alpha=0.8)

        # Plot 2: Timestep distribution histogram
        axes[0, 1].hist(
            timesteps,
            bins=dt_bins,
            alpha=0.6,
            label=label,
            color=color,
            density=True,
        )

        # Plot 3: Time distribution
        axes[1, 0].hist(
            times,
            bins=time_bins,
            alpha=0.6,
            label=label,
            color=color,
            density=True,
        )

        # Plot 4: Cumulative timestep distribution
        sorted_timesteps = np.sort(timesteps)
        cumulative = np.arange(1, len(sorted_timesteps) + 1) / len(
            sorted_timesteps
        )
        axes[1, 1].plot(sorted_timesteps, cumulative, label=label, color=color)

    # Format plots
    axes[0, 0].set_xlabel("Simulation Time")
    axes[0, 0].set_ylabel("Timestep")
    axes[0, 0].set_title("Timestep Evolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Timestep Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Simulation Time")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Time Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Cumulative Fraction")
    axes[1, 1].set_title("Cumulative Timestep Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = create_output_path(
        str(output_path.parent),
        prefix,
        "timestep_analysis.png",
        str(output_path.parent),
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Timestep analysis plot saved to: {output_file}")
    plt.close()
