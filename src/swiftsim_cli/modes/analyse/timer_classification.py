"""Timer classification utilities for SWIFT log analysis.

This module provides functions to classify timers in SWIFT logs into function
and operation timers. The classification is based on execution time analysis
rather than synthetic timer generation.

Key functions:
- classify_timers_by_max_time: Main classification logic using max time per
  function
"""

from collections import defaultdict
from typing import Dict


def classify_timers_by_max_time(
    instances_by_step: Dict, timer_db: Dict, nesting_db: Dict
) -> tuple[set, dict]:
    """Classify timers using max time per function.

    For each function, the timer with the most total time is treated as the
    function timer. All others are operation timers. No synthetic timers.

    This approach eliminates the need for complex synthetic timer generation
    and instead uses the actual timing data from the SWIFT logs to determine
    which timer represents the overall function execution.

    Args:
        instances_by_step: Dictionary mapping step numbers to timer instances
        timer_db: Dictionary of timer definitions by timer ID
        nesting_db: Dictionary of nesting relationships by function name

    Returns:
        Tuple of (function_timer_ids, synthetic_timers):
        - function_timer_ids: Set of timer IDs that are function timers
        - synthetic_timers: Dictionary of synthetic timers (empty in this
            implementation)

    Example:
        >>> function_timer_ids, synthetic_timers = classify_timers_by_max_time(
        ...     instances_by_step, timer_db, nesting_db
        ... )
        >>> print(f"Found {len(function_timer_ids)} function timers")
    """
    function_timer_ids = set()

    # Calculate total time per timer across all steps
    timer_totals: dict[int, float] = defaultdict(float)
    for inst_list in instances_by_step.values():
        for inst in inst_list:
            timer_totals[inst.timer_id] += inst.time_ms

    # Group timers by function name
    timers_by_function = defaultdict(list)
    for tid, total_time in timer_totals.items():
        if tid in timer_db:
            func_name = timer_db[tid].function
            timers_by_function[func_name].append((tid, total_time))

    # For each function, select the timer with max time as function timer
    # or create synthetic timer if appropriate
    synthetic_timers: dict[str, float] = {}

    for func_name, timer_list in timers_by_function.items():
        if len(timer_list) == 1:
            # Single timer - check if it's a generic function timer
            timer_id = timer_list[0][0]
            timer_def = timer_db[timer_id]

            # If it's a generic "took" timer pattern, treat as function timer
            if timer_def.label_text in ["took %.3f %s.", "took %.3f %s"]:
                function_timer_ids.add(timer_id)
            else:
                # Single operation timer - create synthetic timer
                total_time = sum(time for _, time in timer_list)
                synthetic_timers[func_name] = total_time
        elif len(timer_list) > 1:
            # Check if any timer is a generic function timer
            generic_timer = None
            for timer_id, time in timer_list:
                timer_def = timer_db[timer_id]
                if timer_def.label_text in ["took %.3f %s.", "took %.3f %s"]:
                    generic_timer = timer_id
                    break

            if generic_timer:
                # Found generic function timer - use it
                function_timer_ids.add(generic_timer)
            else:
                # No generic timer - check nesting database guidance
                if func_name in nesting_db:
                    # Nesting database available - create synthetic timer
                    total_time = sum(time for _, time in timer_list)
                    synthetic_timers[func_name] = total_time
                else:
                    # No nesting database - use heuristic based on timer ratio
                    timer_list.sort(key=lambda x: x[1], reverse=True)
                    largest_time = timer_list[0][1]
                    second_largest_time = timer_list[1][1]

                    # If largest timer is more than 2x the second largest,
                    # treat as function timer
                    if largest_time > 2 * second_largest_time:
                        max_timer_id = timer_list[0][0]
                        function_timer_ids.add(max_timer_id)
                    else:
                        # No clear function timer - create synthetic timer
                        total_time = sum(time for _, time in timer_list)
                        synthetic_timers[func_name] = total_time

    # Update timer type classification in timer_db for analysis
    for tid, timer_def in timer_db.items():
        if tid in function_timer_ids:
            timer_def.timer_type = "function"
        else:
            timer_def.timer_type = "operation"
    return function_timer_ids, synthetic_timers
