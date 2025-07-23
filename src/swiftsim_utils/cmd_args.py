"""Argument collector for swift-utils."""

import argparse
from typing import Literal, Sequence

Mode = Literal[
    "config",
    "new",
    "output-times",
]


def _config_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for the 'init' mode."""
    p_init = subparser.add_parser(
        "config",
        help="Configure SWIFT-utils settings.",
    )
    # No other arguments, this will just walk the user through the config.


def _new_mode_setup(subparser: argparse._SubParsersAction) -> None:
    """Add arguments for the 'new' mode."""
    p_new = subparser.add_parser(
        "new",
        help="Create a new SWIFT run directory.",
    )
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
        help="Generate an output list file containing times for each"
        " snap/snipshot.",
    )

    # We always need the output file
    p_times.add_argument(
        "--out",
        "-o",
        default="output_list.txt",
        help="Output file for the list of times.",
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
        help="Time of the first snapshot in Gyrs.",
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
        help="Time interval between snapshots in Gyrs.",
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
        help="Time interval between snipshots in Gyrs.",
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

    # ---------------------- internals ----------------------

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

        # config
        _config_mode_setup(subparsers)

        # new
        _new_mode_setup(subparsers)

        # output-times
        _output_times_mode_setup(subparsers)

        return parser
