"""Unit tests for timer classification in the analyse module."""

from swiftsim_cli.modes.analyse import classify_timers_by_max
from swiftsim_cli.src_parser import TimerDef, TimerInstance


class TestTimerClassification:
    """Test timer classification logic."""

    def test_single_timer_classification(
        self, sample_timer_db, sample_nesting_db
    ):
        """Test classification when function has only one timer."""
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
            instances_by_step, single_timer_db, nesting_db
        )

        assert "engine_launch.c:50" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_space_split_classification_with_nesting(
        self, sample_timer_db, sample_nesting_db
    ):
        """Test classification for space_split with nesting database."""
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
            instances_by_step, space_split_timer_db, nesting_db
        )

        # Should promote the larger timer to function timer (1000 > 2 * 100)
        assert "space_split.c:100" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_heuristic_classification_small_ratio(self, sample_timer_db):
        """Test heuristic classification when max timer is not larger."""
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
            instances_by_step, generic_timer_db, sample_nesting_db
        )

        # Should find the generic "took" timer as function timer
        assert "space_split.c:110" in function_timer_ids
        assert len(synthetic_timers) == 0

    def test_empty_instances(self):
        """Test classification with empty instances."""
        instances_by_step = {}
        timer_db = {}
        nesting_db = {}

        function_timer_ids, synthetic_timers = classify_timers_by_max(
            instances_by_step, timer_db, nesting_db
        )

        assert len(function_timer_ids) == 0
        assert len(synthetic_timers) == 0

    def test_nesting_db_none_function_timer_pattern(self, sample_timer_db):
        """Test nesting database when function_timer is None."""
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

        function_timer_ids, synthetic_timers = classify_timers_by_max(
            instances_by_step, space_split_timer_db, nesting_db
        )

        # Should create synthetic timer since pattern is None
        assert "space_split" in synthetic_timers
        assert len(function_timer_ids) == 0
