"""Unit tests for the src_parser module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.src_parser import (
    TimerDef,
    TimerInstance,
    TimerNestingGenerator,
    TimerSite,
    _build_log_pattern,
    _classify_timer_type,
    _extract_function_name_from_header,
    _printf_to_regex,
    _scan_balanced_call,
    _unescape_minimal,
    compile_site_patterns,
    scan_log_instances_by_step,
)


class TestUtilityFunctions:
    """Test utility functions in src_parser module."""

    def test_unescape_minimal(self):
        """Test minimal string unescaping."""
        assert _unescape_minimal(r"Hello \"World\"") == 'Hello "World"'
        assert (
            _unescape_minimal(r"Path\\to\\file") == "Path\\to\\file"
        )  # \\ -> \
        assert _unescape_minimal(r"Line\nbreak") == "Line\nbreak"
        assert _unescape_minimal(r"Tab\there") == "Tab\there"

    def test_printf_to_regex(self):
        """Test printf format string to regex conversion."""
        label, regex = _printf_to_regex("took %.3f %s")
        assert label == "took %.3f %s"
        assert ".*?" in regex

        label, regex = _printf_to_regex("Operation took %f ms")
        assert label == "Operation took %f ms"
        assert ".*?" in regex

    def test_classify_timer_type(self):
        """Test timer type classification."""
        # All timers should return 'timer' for initial classification
        assert _classify_timer_type("took %.3f %s") == "timer"
        assert _classify_timer_type("Operation took %.3f %s") == "timer"
        assert _classify_timer_type("Complex operation took %f ms") == "timer"

    def test_build_log_pattern(self):
        """Test log pattern building."""
        label, pattern = _build_log_pattern("test_function", "took %.3f %s")

        assert "test_function" in pattern
        assert label == "took %.3f %s"
        # Check for the actual pattern components
        assert "took" in pattern
        assert "test_function" in pattern
        assert "([\\d.]+)" in pattern  # Captures decimal numbers

    def test_extract_function_name_from_header(self):
        """Test function name extraction from C headers."""
        # Valid function headers
        assert (
            _extract_function_name_from_header("void test_function(int x) {")
            == "test_function"
        )
        assert (
            _extract_function_name_from_header(
                "static int calculate_sum(float a, float b) {"
            )
            == "calculate_sum"
        )
        assert (
            _extract_function_name_from_header(
                "INLINE double space_split(struct cell *c) {"
            )
            == "space_split"
        )

        # Should filter out keywords
        assert _extract_function_name_from_header("if (condition) {") is None
        assert (
            _extract_function_name_from_header(
                "for (int i = 0; i < 10; i++) {"
            )
            is None
        )

        # No parentheses
        assert _extract_function_name_from_header("int variable = 5;") is None

    def test_scan_balanced_call(self):
        """Test balanced parentheses scanning."""
        lines = [
            'message("Simple message")',
            'message("Multi-line"',
            '        " message")',
            "other_code();",
        ]

        # Simple case
        result, end_line = _scan_balanced_call(lines, 0, 7)  # After "message"
        assert '"Simple message"' in result
        assert end_line == 0

        # Multi-line case
        result, end_line = _scan_balanced_call(lines, 1, 7)  # After "message"
        assert '"Multi-line"' in result
        assert '" message"' in result
        assert end_line == 2


class TestTimerDataStructures:
    """Test timer data structure classes."""

    def test_timer_site_creation(self):
        """Test TimerSite creation."""
        site = TimerSite(
            timer_id="test.c:100",
            file="test.c",
            function="test_func",
            start_line=95,
            end_line=100,
            tic_var="tic",
            label_text="Operation took time",
            log_pattern="test pattern",
            timer_type="operation",
        )

        assert site.timer_id == "test.c:100"
        assert site.function == "test_func"
        assert site.timer_type == "operation"

    def test_timer_def_creation(self):
        """Test TimerDef creation."""
        timer_def = TimerDef(
            timer_id="test.c:100",
            function="test_func",
            log_pattern="test pattern",
            start_line=95,
            end_line=100,
            label_text="Operation took time",
            timer_type="operation",
        )

        assert timer_def.timer_id == "test.c:100"
        assert timer_def.function == "test_func"
        assert timer_def.timer_type == "operation"

    def test_timer_instance_creation(self):
        """Test TimerInstance creation."""
        instance = TimerInstance(
            timer_id="test.c:100",
            function="test_func",
            step=5,
            time_ms=123.45,
            line_index=200,
            timer_type="operation",
        )

        assert instance.timer_id == "test.c:100"
        assert instance.time_ms == 123.45
        assert instance.step == 5


class TestPatternCompilation:
    """Test pattern compilation functions."""

    def test_compile_site_patterns(self, sample_timer_db):
        """Test timer pattern compilation."""
        compiled = compile_site_patterns(sample_timer_db)

        assert len(compiled) == len(sample_timer_db)

        for timer_id, pattern in compiled:
            assert timer_id in sample_timer_db
            assert hasattr(pattern, "search")  # Should be compiled regex

    def test_compile_site_patterns_bad_regex(self):
        """Test pattern compilation with bad regex."""
        bad_timer_db = {
            "bad.c:1": TimerDef(
                timer_id="bad.c:1",
                function="bad_func",
                log_pattern="[unclosed bracket",  # Invalid regex
                start_line=1,
                end_line=1,
                label_text="bad",
                timer_type="operation",
            )
        }

        # Should handle bad regex gracefully
        with patch("builtins.print") as mock_print:
            compiled = compile_site_patterns(bad_timer_db)
            assert len(compiled) == 0  # Bad regex should be filtered out
            mock_print.assert_called()  # Should print warning


class TestLogScanning:
    """Test log scanning functionality."""

    def test_scan_log_instances_by_step(
        self, temp_dir, sample_timer_db, sample_log_content
    ):
        """Test log scanning for timer instances."""
        # Create a test log file
        log_file = temp_dir / "test.log"
        log_file.write_text(sample_log_content)

        # Compile patterns
        compiled = compile_site_patterns(sample_timer_db)

        # Scan the log
        instances_by_step, step_lines = scan_log_instances_by_step(
            str(log_file), compiled, sample_timer_db
        )

        # Should find instances
        assert len(instances_by_step) > 0

        # Check that instances are properly created
        for step, instances in instances_by_step.items():
            for instance in instances:
                assert isinstance(instance, TimerInstance)
                assert instance.timer_id in sample_timer_db
                assert instance.time_ms > 0

    def test_scan_log_instances_empty_file(self, temp_dir, sample_timer_db):
        """Test log scanning with empty file."""
        log_file = temp_dir / "empty.log"
        log_file.write_text("")

        compiled = compile_site_patterns(sample_timer_db)

        instances_by_step, step_lines = scan_log_instances_by_step(
            str(log_file), compiled, sample_timer_db
        )

        assert len(instances_by_step) == 0
        assert len(step_lines) == 0

    @patch("builtins.open", side_effect=OSError("File not found"))
    def test_scan_log_instances_file_error(self, mock_open, sample_timer_db):
        """Test log scanning with file error."""
        compiled = compile_site_patterns(sample_timer_db)

        # Should handle file errors gracefully
        with pytest.raises(OSError):
            scan_log_instances_by_step(
                "nonexistent.log", compiled, sample_timer_db
            )


class TestTimerNestingGeneration:
    """Test timer nesting generation functionality."""

    def test_tree_sitter_availability(self):
        """Test tree-sitter availability check."""
        # Test that tree-sitter can be imported
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_c  # noqa: F401

            tree_sitter_available = True
        except ImportError:
            tree_sitter_available = False

        assert isinstance(tree_sitter_available, bool)

    @pytest.mark.skipif(
        not pytest.importorskip(
            "tree_sitter", reason="tree-sitter not available"
        ),
        reason="Requires tree-sitter",
    )
    def test_timer_nesting_generator_init(self):
        """Test TimerNestingGenerator initialization."""
        timer_data = [
            {
                "timer_id": "test.c:1",
                "function": "test_func",
                "timer_type": "function",
                "file": "test.c",
            }
        ]

        generator = TimerNestingGenerator("/fake/src", timer_data)
        assert generator.src_dir == Path("/fake/src")
        assert len(generator.timer_data) == 1
