"""Argument collector for SWFTISim-CLI."""

import argparse
from pathlib import Path
from typing import Literal, Sequence

from swiftsim_utils.config import load_swift_config

# Declare the modes available for swift-utils.
Mode = Literal[
    "init",
    "config",
    "new",
    "output-times",
]


def _add_common_arguments(common: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    # Common arguments for all modes (all are optional)

    # Get the config
    config = load_swift_config()
    swift_dir = config.swiftsim_dir if config else None
    data_dir = config.data_dir if config else None

    # Overrides to the config file
    common.add_argument(
        "--swift-dir",
        type=Path,
        help=(
            "Path to the SWIFT directory (overrides "
            f"config file: {swift_dir})."
        ),
        default=None,
    )
    common.add_argument(
        "--data-dir",
        type=Path,
        help=(
            f"Path to the data directory (overrides config file: {data_dir})."
        ),
        default=None,
    )


def _init_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for the 'init' mode."""
    p_init = subparser.add_parser(
        "init",
        help="Configure SWIFTsim-CLI settings.",
    )
    _add_common_arguments(p_init)
    # No other arguments, this will just walk the user through the config.


def _config_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for configuring SWIFT itself."""
    p_config = subparser.add_parser(
        "config",
        help="Configure SWIFT configuration in the SWIFT directory.",
    )
    _add_common_arguments(p_config)

    # Here we need to ingest an arbitrary length string containing all the
    # config options.
    p_config.add_argument(
        "--options",
        "-o",
        type=str,
        help=(
            "Configuration options to set in the SWIFT config file as they"
            " would be passed to the ./configure command."
        ),
        default="",
    )

    # Show the configuration options (equivalent to running
    # `./configure --help`).
    p_config.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show the available configuration options.",
    )


def _new_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for the 'new' mode."""
    p_new = subparser.add_parser(
        "new",
        help="Create a new SWIFT run directory.",
    )
    _add_common_arguments(p_new)
    p_new.add_argument(
        "--path",
        required=True,
        help="Path to the new SWIFT run.",
    )
    p_new.add_argument(
        "--inic",
        required=True,
        help="Path to the initial conditions HDF5 file.",
    )


def _output_times_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for the 'output-times' mode."""
    p_times = subparser.add_parser(
        "output-times",
        help=(
            "Generate an output list file containing times for each"
            " snap/snipshot."
        ),
    )
    _add_common_arguments(p_times)

    # We always need the output file
    p_times.add_argument(
        "--out",
        "-o",
        default="output_list.txt",
        help="Output file for the list of times.",
    )

    # We can optionally provide a parameter file, this will be used to
    # get cosmology parameters and other settings.
    p_times.add_argument(
        "-p",
        "--params",
        type=Path,
        help="Path to a parameter file.",
        default=None,
    )

    # We will always need one definition of the first snapshot but this can be
    # defined in terms of redshift, time, or scale factor.
    p_times.add_argument(
        "--first-snap-z",
        "-fz",
        type=float,
        default=None,
        help="Redshift of the first snapshot to include.",
    )
    p_times.add_argument(
        "--first-snap-time",
        "-ft",
        type=float,
        default=None,
        help="Time of the first snapshot in internal units.",
    )
    p_times.add_argument(
        "--first-snap-scale-factor",
        "-fa",
        type=float,
        default=None,
        help="Scale factor of the first snapshot to include.",
    )

    # Similarly with the delta between snapshots, we can define this in terms
    # of redshift, time, scale factor, or logarithmic scale factor.
    p_times.add_argument(
        "--delta-z",
        "-dz",
        type=float,
        default=None,
        help="Redshift interval between snapshots.",
    )
    p_times.add_argument(
        "--delta-time",
        "-dt",
        type=float,
        default=None,
        help="Time interval between snapshots in internal units.",
    )
    p_times.add_argument(
        "--delta-scale-factor",
        "-da",
        type=float,
        default=None,
        help="Scale factor interval between snapshots.",
    )
    p_times.add_argument(
        "--delta-log-scale-factor",
        "-dla",
        type=float,
        default=None,
        help="Logarithmic scale factor interval between snapshots.",
    )

    # If we want snipshots, in between we just define a smaller delta for these
    # (again, in terms of redshift, time, scale factor, or
    # logarithmic scale factor).
    p_times.add_argument(
        "--snipshot-delta-z",
        "-sdz",
        type=float,
        default=None,
        help="Redshift interval between snipshots.",
    )
    p_times.add_argument(
        "--snipshot-delta-time",
        "-sdt",
        type=float,
        default=None,
        help="Time interval between snipshots in internal units.",
    )
    p_times.add_argument(
        "--snipshot-delta-scale-factor",
        "-sda",
        type=float,
        default=None,
        help="Scale factor interval between snipshots.",
    )
    p_times.add_argument(
        "--snipshot-delta-log-scale-factor",
        "-sdla",
        type=float,
        default=None,
        help="Logarithmic scale factor interval between snipshots.",
    )

    # We will also need to the final snapshot, for redshift and scale factor
    # this has a well defined default, for time it does not.
    p_times.add_argument(
        "--final-snap-z",
        "-fzf",
        type=float,
        default=0.0,
        help="Redshift of the final snapshot to include (default: 0.0).",
    )
    p_times.add_argument(
        "--final-snap-time",
        "-ftf",
        type=float,
        default=None,
        help="Time of the final snapshot in internal units (default: None).",
    )
    p_times.add_argument(
        "--final-snap-scale-factor",
        "-faf",
        type=float,
        default=1.0,
        help="Scale factor of the final snapshot to include (default: 1.0).",
    )


class SwiftUtilsArgs:
    """Builds and parses command-line arguments for swift-utils.

    Example:
        opts = SwiftUtilsArgs()   # parses sys.argv by default
    """

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        """Initialize the argument parser and parse the arguments.

        Args:
            argv: The command-line arguments to parse. If None, uses sys.argv.
        """
        self._parser = self._build_parser()
        self.args = self._parser.parse_args(argv)

        self.mode: Mode = self.args.mode

    def __getattr__(self, name: str) -> object:
        """Return the parsed arguments for the specified mode."""
        if not hasattr(self.args, name):
            raise AttributeError(
                f"Unknown attribute: {name} in {self.mode} mode"
            )
        return getattr(self.args, name)

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build the argument parser for swift-utils.

        Returns:
            An argparse.ArgumentParser instance configured with subcommands.
        """
        parser = argparse.ArgumentParser(
            prog="swift-utils",
            description="Utilities for Swift development workflows.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output.",
        )

        subparsers = parser.add_subparsers(
            dest="mode",
            metavar="<mode>",
            required=True,
        )

        # initialise
        _init_mode_setup(subparsers)

        # config
        _config_mode_setup(subparsers)

        # new
        _new_mode_setup(subparsers)

        # output-times
        _output_times_mode_setup(subparsers)

        return parser
