"""A module for generating output lists for defining SWIFT snapshot times."""

import numpy as np


def generate_output_list(args: list[str]) -> None:
    """Generate an output list file containing times for each snapshot.

    Args:
        args: Command-line arguments containing the configuration for the
              output list generation.
    """
    # Unpack the arguments
    out_file = args.get("out", "output_list.txt")
    first_snap_z = args.get("first_snap_z", None)
    first_snap_time = args.get("first_snap_time", None)
    first_snap_scale_factor = args.get("first_snap_scale_factor", None)
    delta_z = args.get("delta_z", None)
    delta_time = args.get("delta_time", None)
    delta_scale_factor = args.get("delta_scale_factor", None)
    delta_log_scale_factor = args.get("delta_log_scale_factor", None)
    snip_delta_z = args.get("snipshot_delta_z", None)
    snip_delta_time = args.get("snipshot_delta_time", None)
    snip_delta_scale_factor = args.get("snipshot_delta_scale_factor", None)
    snip_delta_log_scale_factor = args.get(
        "snipshot_delta_log_scale_factor", None
    )
    final_snap_z = args.get("final_snap_z", None)
    final_snap_time = args.get("final_snap_time", None)
    final_snap_scale_factor = args.get("final_snap_scale_factor", None)

    # Are we doing redshift, time, or scale factor?
    doing_z = first_snap_z is not None
    doing_time = first_snap_time is not None
    doing_scale_factor = first_snap_scale_factor is not None

    # Do we have everything we need?
    if doing_z and not delta_z:
        raise ValueError(
            "If you specify the first snapshot redshift, you must also "
            "specify the delta in terms of redshift."
        )
    if doing_time and not delta_time:
        raise ValueError(
            "If you specify the first snapshot time, you must also "
            "specify the delta in terms of time."
        )
    if (
        doing_scale_factor
        and delta_scale_factor is None
        and delta_log_scale_factor is None
    ):
        raise ValueError(
            "If you specify the first snapshot scale factor, you must also "
            "specify the delta in terms of scale factor or logarithmic"
            " scale factor."
        )

    # Collapse the variables into a simpler set now we know we have what
    # we need.
    first_snap = first_snap_z or first_snap_time or first_snap_scale_factor
    delta = (
        delta_z or delta_time or delta_scale_factor or delta_log_scale_factor
    )
    snip_delta = (
        snip_delta_z
        or snip_delta_time
        or snip_delta_scale_factor
        or snip_delta_log_scale_factor
    )

    # Make sure we use the right final snapshot time, this has defaults so we
    # need to be careful
    final_snap = None
    if doing_z:
        final_snap = final_snap_z
    elif doing_time:
        final_snap = final_snap_time
    elif doing_scale_factor:
        final_snap = final_snap_scale_factor

    # Ensure we have a meaningful final snapshot
    if final_snap is None:
        raise ValueError(
            "You must specify the final snapshot time, redshift, or scale "
            "factor."
        )

    # Is our delta logarithmic? This will require special handling.
    doing_log_scale_factor = delta_log_scale_factor is not None
    doing_snip_log_scale_factor = snip_delta_log_scale_factor is not None

    # Do we have snipshots?
    has_snipshots = snip_delta is not None

    # Make sure our snipshot delta is the same type as our delta if we have it.
    if has_snipshots:
        if doing_z and snip_delta_z is None:
            raise ValueError(
                "Snipshot delta must be given in terms of redshift."
            )
        if doing_time and snip_delta_time is None:
            raise ValueError("Snipshot delta must be given in terms of time.")
        if (
            doing_scale_factor
            and snip_delta_scale_factor is None
            and snip_delta_log_scale_factor is None
        ):
            raise ValueError(
                "Snipshot delta must be given in terms of scale factor or "
                "logarithmic scale factor."
            )

    # OK, we seem to have what we need. Lets tell the user what we have
    print("Generating output list with the following parameters:")
    print(f"  First snapshot: {first_snap} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor:
        print(" (Scale factor)")
    print(f"  Delta: {delta} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor and not doing_log_scale_factor:
        print(" (Scale factor)")
    elif doing_scale_factor and doing_log_scale_factor:
        print(" (Logarithmic scale factor)")
    if has_snipshots:
        print(f"  Snipshot delta: {snip_delta} ", end="")
        if doing_z:
            print(" (Redshift)")
        elif doing_time:
            print(" (Time in internal units)")
        elif doing_scale_factor and not doing_snip_log_scale_factor:
            print(" (Scale factor)")
        elif doing_scale_factor and doing_snip_log_scale_factor:
            print(" (Logarithmic scale factor)")
    print(f"  Final snapshot: {final_snap} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor:
        print(" (Scale factor)")

    # Ensure the delta has the right sign
    if doing_z and delta > 0:
        delta = -delta
    elif doing_time and delta < 0:
        delta = -delta
    elif doing_scale_factor and delta < 0 and not doing_log_scale_factor:
        delta = -delta
    elif doing_scale_factor and delta < 0 and doing_log_scale_factor:
        delta = -delta

    # Set up lists to hold the output times
    snapshot_times = (
        np.arange(first_snap, final_snap + delta, delta)
        if not doing_log_scale_factor
        else 10
        ** np.arange(np.log10(first_snap), np.log10(final_snap) + delta, delta)
    )
    snipshot_times = (
        np.arange(first_snap, final_snap + snip_delta, snip_delta)
        if not doing_snip_log_scale_factor
        else 10
        ** np.arange(
            np.log10(first_snap), np.log10(final_snap) + snip_delta, snip_delta
        )
    )

    # No point in doubling up on snapshots and snipshots, remove any snipshots
    # that are already in the snapshot list.
    snipshot_times = snipshot_times[~np.isin(snipshot_times, snapshot_times)]

    # Define the first line of the output text file (with two columns)
    output_lines = []
    if doing_z:
        output_lines.append("# Redshift, Select Output")
    elif doing_time:
        output_lines.append("# Time, Select Output")
    elif doing_scale_factor:
        output_lines.append("# Scale Factor, Select Output")

    # Create the select output column
    select_output_snap = ["Snapshot"] * len(snapshot_times)
    select_output_snip = ["Snipshot"] * len(snipshot_times)

    # Combine the times and select output into lines
    times = np.concatenate((snapshot_times, snipshot_times))
    select_output = np.concatenate((select_output_snap, select_output_snip))

    # Sort the times and select output together (here we need to treat redshift
    # as descending and time/scale factor as ascending)
    if doing_z:
        sorted_indices = np.argsort(times)[::-1]
    else:
        sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_select_output = select_output[sorted_indices]

    # Create the output lines
    for time, select in zip(sorted_times, sorted_select_output):
        output_lines.append(f"{time}, {select}")

    # Write the output to the file
    with open(out_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"Output list written to {out_file}.")
