"""The main module containing the swiftsim-utils CLI tool."""

from swiftsim_utils.cmd_args import SWIFTSimCLIArgs
from swiftsim_utils.config import config_swift_utils, load_swift_config
from swiftsim_utils.output_list import generate_output_list
from swiftsim_utils.params import load_parameters
from swiftsim_utils.swiftsim_dir import (
    compile_swift,
    config_swiftsim,
    show_config_options,
    switch_swift_branch,
    update_swift,
)


def main(argv: list[str] | None = None) -> None:
    """Run the SWIFT-utils command-line interface.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    # Load the config first, we may need it later and this will cache it for
    # future use.
    _ = load_swift_config()

    # Parse the command-line arguments
    args = SWIFTSimCLIArgs(argv).args

    # Load the parameters if they exist
    _ = load_parameters(getattr(args, "params", None))

    # Run the appropriate command based on the mode
    if args.mode == "init":
        config_swift_utils()
    elif args.mode == "output-times":
        generate_output_list(vars(args))
    elif args.mode == "config":
        # Are we just showing the config options?
        if args.show:
            show_config_options()
        else:
            config_swiftsim(
                opts=args.options,
                swift_dir=args.swift_dir,
            )
    elif args.mode == "update":
        update_swift(args.swift_dir)
    elif args.mode == "switch-branch":
        switch_swift_branch(
            branch=args.branch,
            swift_dir=args.swift_dir,
        )
    elif args.mode == "compile":
        compile_swift(
            swift_dir=args.swift_dir,
            nr_threads=args.nr_threads,
        )
