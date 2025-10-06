"""Unit tests for timer classification in the analyse module."""

from collections import defaultdict

from swiftsim_cli.src_parser import TimerDef, TimerInstance


class TestTimerClassification:
    """Test timer classification logic."""

    def create_timer_classification_function(self):
        """Create the timer classification function for testing."""

        def classify_timers_by_max(instances_by_step, timer_db, nesting_db):
            """Classify timers dynamically."""
            # Collect all timer times by function
            timer_totals_by_function = defaultdict(lambda: defaultdict(float))

            for inst_list in instances_by_step.values():
                for inst in inst_list:
                    func_name = timer_db[inst.timer_id].function
                    timer_totals_by_function[func_name][inst.timer_id] += (
                        inst.time_ms
                    )

            # For each function, determine the function timer intelligently
            function_timer_ids = set()
            synthetic_function_timers = {}

            for func_name, timer_totals in timer_totals_by_function.items():
                if not timer_totals:  # Skip if no timers
                    continue

                # Check if nesting database has guidance for this function
                if func_name in nesting_db and nesting_db[func_name].get(
                    "nested_operations"
                ):
                    # Nesting database indicates this function should have
                    # multiple operations Look for a timer that matches the
                    # function_timer pattern
                    function_timer_pattern = nesting_db[func_name].get(
                        "function_timer", ""
                    )
                    function_timer_found = False

                    # Try to find a timer matching the function timer pattern
                    for tid in timer_totals.keys():
                        timer_label = timer_db[tid].label_text
                        # Simple pattern matching - if function timer pattern
                        # is "took %.3f %s." then look for timers with just
                        # "took" without specific operation descriptions
                        if (
                            function_timer_pattern
                            and "took" in function_timer_pattern
                            and "took" in timer_label
                        ):
                            # Check if this is a generic "took" timer (not a
                            # specific operation)
                            # Specific operations usually have descriptive
                            # text before "took"
                            words_before_took = timer_label.split("took")[
                                0
                            ].strip()
                            if (
                                not words_before_took
                                or len(words_before_took.split()) <= 2
                            ):
                                # This looks like a generic function timer
                                function_timer_ids.add(tid)
                                function_timer_found = True
                                break

                    if not function_timer_found:
                        # No function timer found, create synthetic one from
                        # sum of operations
                        total_time = sum(timer_totals.values())
                        synthetic_function_timers[func_name] = total_time
                        # All existing timers remain as operations
                else:
                    # No nesting database guidance, fall back to heuristic
                    if len(timer_totals) == 1:
                        # Only one timer - it's the function timer
                        function_timer_ids.add(list(timer_totals.keys())[0])
                    else:
                        # Multiple timers - check if max timer represents the
                        # whole function
                        sorted_timers = sorted(
                            timer_totals.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        max_timer_id, max_time = sorted_timers[0]
                        other_timers_sum = sum(
                            time for tid, time in sorted_timers[1:]
                        )

                        # Use a more sophisticated heuristic:
                        # Only treat max timer as function timer if it's
                        # significantly larger (at least 2x) than the sum of
                        # others, indicating it encompasses them
                        ratio_threshold = 2.0
                        if max_time > ratio_threshold * other_timers_sum:
                            # Max timer is significantly larger than sum of
                            # others - it's the function timer
                            function_timer_ids.add(max_timer_id)
                        else:
                            # No single dominant timer - function timer is
                            # sum of all operations. We'll create a synthetic
                            # function timer entry
                            total_time = sum(timer_totals.values())
                            synthetic_function_timers[func_name] = total_time

            # Update timer_db with dynamic classification
            for tid, timer_def in timer_db.items():
                if tid in function_timer_ids:
                    timer_def.timer_type = "function"
                else:
                    timer_def.timer_type = "operation"

            return function_timer_ids, synthetic_function_timers

        return classify_timers_by_max

    def test_single_timer_classification(
        self, sample_timer_db, sample_nesting_db
    ):
        """Test classification when function has only one timer."""
        classify_func = self.create_timer_classification_function()

        # Create instances for a function with single timer
        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="engine_launch.c:50",
                    function="engine_launch",
                    step=0,
                    time_ms=800.0,
                    line_index=50,
                    timer_type="function",
                )
            ]
        }

        # Filter timer_db to only include the single timer
        single_timer_db = {
            "engine_launch.c:50": sample_timer_db["engine_launch.c:50"]
        }

        # Nesting DB without nested operations for this function
        nesting_db = {}

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, single_timer_db, nesting_db
        )

        assert "engine_launch.c:50" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_space_split_classification_with_nesting(
        self, sample_timer_db, sample_nesting_db
    ):
        """Test classification for space_split with nesting database."""
        classify_func = self.create_timer_classification_function()

        # Create instances for space_split function
        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="space_split.c:100",
                    function="space_split",
                    step=0,
                    time_ms=609.098,
                    line_index=100,
                    timer_type="operation",
                ),
                TimerInstance(
                    timer_id="space_split.c:105",
                    function="space_split",
                    step=0,
                    time_ms=15.180,
                    line_index=101,
                    timer_type="operation",
                ),
            ]
        }

        # Filter timer_db to only include space_split timers
        space_split_timer_db = {
            "space_split.c:100": sample_timer_db["space_split.c:100"],
            "space_split.c:105": sample_timer_db["space_split.c:105"],
        }

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, space_split_timer_db, sample_nesting_db
        )

        # Should create synthetic timer since no simple "took" timer found
        assert "space_split" in synthetic_timers
        assert len(function_timer_ids) == 0
        assert (
            abs(synthetic_timers["space_split"] - (609.098 + 15.180)) < 0.001
        )

    def test_heuristic_classification_large_ratio(self, sample_timer_db):
        """Test heuristic classification when max timer is larger."""
        classify_func = self.create_timer_classification_function()

        # Create instances where one timer is much larger (> 2x) than others
        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="space_split.c:100",
                    function="space_split",
                    step=0,
                    time_ms=1000.0,  # Much larger
                    line_index=100,
                    timer_type="operation",
                ),
                TimerInstance(
                    timer_id="space_split.c:105",
                    function="space_split",
                    step=0,
                    time_ms=100.0,  # Much smaller
                    line_index=101,
                    timer_type="operation",
                ),
            ]
        }

        space_split_timer_db = {
            "space_split.c:100": sample_timer_db["space_split.c:100"],
            "space_split.c:105": sample_timer_db["space_split.c:105"],
        }

        # No nesting database guidance
        nesting_db = {}

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, space_split_timer_db, nesting_db
        )

        # Should promote the larger timer to function timer (1000 > 2 * 100)
        assert "space_split.c:100" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_heuristic_classification_small_ratio(self, sample_timer_db):
        """Test heuristic classification when max timer is not larger."""
        classify_func = self.create_timer_classification_function()

        # Create instances where timers are similar in size
        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="space_split.c:100",
                    function="space_split",
                    step=0,
                    time_ms=300.0,  # Not much larger
                    line_index=100,
                    timer_type="operation",
                ),
                TimerInstance(
                    timer_id="space_split.c:105",
                    function="space_split",
                    step=0,
                    time_ms=250.0,  # Close in size
                    line_index=101,
                    timer_type="operation",
                ),
            ]
        }

        space_split_timer_db = {
            "space_split.c:100": sample_timer_db["space_split.c:100"],
            "space_split.c:105": sample_timer_db["space_split.c:105"],
        }

        # No nesting database guidance
        nesting_db = {}

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, space_split_timer_db, nesting_db
        )

        # Should create synthetic timer (300 < 2 * 250)
        assert "space_split" in synthetic_timers
        assert len(function_timer_ids) == 0
        assert abs(synthetic_timers["space_split"] - 550.0) < 0.001

    def test_nesting_db_with_function_timer_found(
        self, sample_timer_db, sample_nesting_db
    ):
        """Test nesting database when function timer pattern is found."""
        classify_func = self.create_timer_classification_function()

        # Add a generic "took" timer to the database
        generic_timer_db = dict(sample_timer_db)
        generic_timer_db["space_split.c:110"] = TimerDef(
            timer_id="space_split.c:110",
            function="space_split",
            log_pattern=r"^.*space_split:\s+took\s+([\d.]+)\s+ms",
            start_line=108,
            end_line=110,
            label_text="took %.3f %s.",  # Generic pattern
            timer_type="operation",
        )

        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="space_split.c:100",
                    function="space_split",
                    step=0,
                    time_ms=609.098,
                    line_index=100,
                    timer_type="operation",
                ),
                TimerInstance(
                    timer_id="space_split.c:105",
                    function="space_split",
                    step=0,
                    time_ms=15.180,
                    line_index=101,
                    timer_type="operation",
                ),
                TimerInstance(
                    timer_id="space_split.c:110",
                    function="space_split",
                    step=0,
                    time_ms=650.0,
                    line_index=102,
                    timer_type="operation",
                ),
            ]
        }

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, generic_timer_db, sample_nesting_db
        )

        # Should find the generic "took" timer as function timer
        assert "space_split.c:110" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_empty_instances(self):
        """Test classification with empty instances."""
        classify_func = self.create_timer_classification_function()

        instances_by_step = {}
        timer_db = {}
        nesting_db = {}

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, timer_db, nesting_db
        )

        assert len(function_timer_ids) == 0
        assert len(synthetic_timers) == 0

    def test_nesting_db_none_function_timer_pattern(self, sample_timer_db):
        """Test nesting database when function_timer is None."""
        classify_func = self.create_timer_classification_function()

        instances_by_step = {
            0: [
                TimerInstance(
                    timer_id="space_split.c:100",
                    function="space_split",
                    step=0,
                    time_ms=609.098,
                    line_index=100,
                    timer_type="operation",
                )
            ]
        }

        # Nesting DB with None function_timer
        nesting_db = {
            "space_split": {
                "function_timer": None,  # None pattern
                "file": "space_split.c",
                "nested_operations": ["Zoom cell tree took %.3f %s."],
                "nested_functions": [],
            }
        }

        space_split_timer_db = {
            "space_split.c:100": sample_timer_db["space_split.c:100"]
        }

        function_timer_ids, synthetic_timers = classify_func(
            instances_by_step, space_split_timer_db, nesting_db
        )

        # Should create synthetic timer since pattern is None
        assert "space_split" in synthetic_timers
        assert len(function_timer_ids) == 0
