"""Mode definitions and constants for swift-cli."""

from typing import Literal

# Available modes for swift-utils CLI
AVAILABLE_MODES = [
    "init",
    "config", 
    "output-times",
    "update",
    "switch",
    "make",
    "new",
    "analyse",
]

# Type hint for mode names
Mode = Literal[
    "init",
    "config",
    "output-times", 
    "update",
    "switch",
    "make",
    "new",
    "analyse",
]

# Mode module imports
from . import init, config, output_times, update, switch, make, new, analyse

# Mapping of mode names to their modules
MODE_MODULES = {
    "init": init,
    "config": config,
    "output-times": output_times,
    "update": update,
    "switch": switch,
    "make": make,
    "new": new,
    "analyse": analyse,
}