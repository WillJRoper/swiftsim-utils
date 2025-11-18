"""Tests for the task counts analysis module."""

from pathlib import Path
from unittest.mock import Mock, patch

from swiftsim_cli.modes.analyse.log_task_counts import (
    add_task_counts_arguments,
    analyse_swift_task_counts,
    run_swift_task_counts,
)
from swiftsim_cli.src_parser import TaskCountSnapshot


class TestTaskCountsArguments:
    """Tests for task counts CLI argument setup."""

    def test_add_task_counts_arguments(self):
        """Test that task counts arguments are added correctly."""
        from argparse import ArgumentParser

        parent_parser = ArgumentParser()
        subparsers = parent_parser.add_subparsers()

        add_task_counts_arguments(subparsers)

        # Parse test arguments
        args = parent_parser.parse_args(
            ["task-counts", "test.log", "--prefix", "test"]
        )

        assert args.log_file == Path("test.log")
        assert args.prefix == "test"
        assert args.show is False
        assert args.tasks is None

    def test_add_task_counts_arguments_with_tasks(self):
        """Test task counts arguments with task filter."""
        from argparse import ArgumentParser

        parent_parser = ArgumentParser()
        subparsers = parent_parser.add_subparsers()

        add_task_counts_arguments(subparsers)

        # Parse test arguments with tasks
        args = parent_parser.parse_args(
            ["task-counts", "test.log", "--tasks", "sort", "self", "pair"]
        )

        assert args.log_file == Path("test.log")
        assert args.tasks == ["sort", "self", "pair"]


class TestRunSwiftTaskCounts:
    """Tests for the task counts CLI entry point."""

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts"
    )
    def test_run_swift_task_counts(self, mock_analyse):
        """Test the CLI entry point calls analyse with correct args."""
        args = Mock()
        args.log_file = Path("/path/to/test.log")
        args.output_path = Path("/output")
        args.prefix = "test"
        args.show = True
        args.tasks = ["sort", "self"]

        run_swift_task_counts(args)

        mock_analyse.assert_called_once_with(
            log_file=str(args.log_file),
            output_path=str(args.output_path),
            prefix="test",
            show_plot=True,
            task_filter=["sort", "self"],
        )

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts"
    )
    def test_run_swift_task_counts_no_filter(self, mock_analyse):
        """Test the CLI entry point without task filter."""
        args = Mock()
        args.log_file = Path("/path/to/test.log")
        args.output_path = None
        args.prefix = None
        args.show = False
        args.tasks = None

        run_swift_task_counts(args)

        mock_analyse.assert_called_once_with(
            log_file=str(args.log_file),
            output_path=None,
            prefix=None,
            show_plot=False,
            task_filter=None,
        )


class TestAnalyseSwiftTaskCounts:
    """Tests for the core task counts analysis function."""

    def create_mock_log(self, tmp_path):
        """Create a mock SWIFT log file with task counts."""
        log_file = tmp_path / "test.log"
        lines = [
            "0 [0.100] engine_print_task_counts: "
            "System total: 1000, no. cells: 10",
            "0 [0.100] engine_print_task_counts: "
            "Total = 1000 (per cell = 100.0)",
            "0 [0.100] engine_print_task_counts: "
            "task counts are [ sort=200 self=300 pair=500 ]",
            "1 [0.200] engine_print_task_counts: "
            "System total: 1200, no. cells: 10",
            "1 [0.200] engine_print_task_counts: "
            "Total = 1200 (per cell = 120.0)",
            "1 [0.200] engine_print_task_counts: "
            "task counts are [ sort=250 self=350 pair=600 ]",
        ]
        log_file.write_text("\n".join(lines))
        return str(log_file)

    @patch("swiftsim_cli.modes.analyse.log_task_counts.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_task_counts.plt")
    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.scan_task_counts_by_step"
    )
    def test_analyse_swift_task_counts_no_filter(
        self, mock_scan, mock_plt, mock_create_path, tmp_path
    ):
        """Test analysis without task filtering."""
        # Mock the output path creation
        mock_create_path.side_effect = lambda *args: tmp_path / "plot.png"

        # Mock matplotlib subplot return
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        # Create mock snapshots
        mock_snap1 = TaskCountSnapshot(
            step=0,
            rank=0,
            sim_time=0.1,
            system_total=1000,
            num_cells=10,
            total_tasks=1000,
            per_cell_avg=100.0,
            per_cell_max=None,
            counts={"sort": 200, "self": 300, "pair": 500},
            line_index=0,
        )
        mock_snap2 = TaskCountSnapshot(
            step=1,
            rank=0,
            sim_time=0.2,
            system_total=1200,
            num_cells=10,
            total_tasks=1200,
            per_cell_avg=120.0,
            per_cell_max=None,
            counts={"sort": 250, "self": 350, "pair": 600},
            line_index=3,
        )

        mock_scan.return_value = ({0: [mock_snap1], 1: [mock_snap2]}, [])

        log_file = self.create_mock_log(tmp_path)

        analyse_swift_task_counts(
            log_file=log_file,
            output_path=None,
            prefix=None,
            show_plot=False,
            task_filter=None,
        )

        # Verify scan was called
        mock_scan.assert_called_once_with(log_file)

        # Verify plots were created (2 plots)
        assert mock_plt.savefig.call_count == 2

    @patch("swiftsim_cli.modes.analyse.log_task_counts.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_task_counts.plt")
    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.scan_task_counts_by_step"
    )
    def test_analyse_swift_task_counts_with_filter(
        self, mock_scan, mock_plt, mock_create_path, tmp_path
    ):
        """Test analysis with task filtering."""
        # Mock the output path creation
        mock_create_path.side_effect = lambda *args: tmp_path / "plot.png"

        # Mock matplotlib subplot return
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        # Create mock snapshots
        mock_snap1 = TaskCountSnapshot(
            step=0,
            rank=0,
            sim_time=0.1,
            system_total=1000,
            num_cells=10,
            total_tasks=1000,
            per_cell_avg=100.0,
            per_cell_max=None,
            counts={"sort": 200, "self": 300, "pair": 500},
            line_index=0,
        )
        mock_snap2 = TaskCountSnapshot(
            step=1,
            rank=0,
            sim_time=0.2,
            system_total=1200,
            num_cells=10,
            total_tasks=1200,
            per_cell_avg=120.0,
            per_cell_max=None,
            counts={"sort": 250, "self": 350, "pair": 600},
            line_index=3,
        )

        mock_scan.return_value = ({0: [mock_snap1], 1: [mock_snap2]}, [])

        log_file = self.create_mock_log(tmp_path)

        analyse_swift_task_counts(
            log_file=log_file,
            output_path=None,
            prefix=None,
            show_plot=False,
            task_filter=["sort", "self"],
        )

        # Verify scan was called
        mock_scan.assert_called_once_with(log_file)

        # Verify plots were created (2 plots)
        assert mock_plt.savefig.call_count == 2

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.scan_task_counts_by_step"
    )
    def test_analyse_swift_task_counts_no_data(
        self, mock_scan, tmp_path, capsys
    ):
        """Test analysis with no valid data."""
        mock_scan.return_value = ({}, [])

        log_file = self.create_mock_log(tmp_path)

        analyse_swift_task_counts(
            log_file=log_file,
            output_path=None,
            prefix=None,
            show_plot=False,
            task_filter=None,
        )

        # Verify appropriate message was printed
        captured = capsys.readouterr()
        assert "No usable engine_print_task_counts blocks" in captured.out
