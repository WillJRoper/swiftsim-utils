"""Unit tests for CLI module."""

from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.cli import main


class TestCLI:
    """Test CLI functionality."""

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    @patch("importlib.import_module")
    def test_main_basic(self, mock_import, mock_parse):
        """Test basic CLI main function."""
        # Mock the argument parsing
        mock_args = Mock()
        mock_args.mode = "analyse"
        mock_parse.return_value = mock_args

        # Mock the mode module
        mock_module = Mock()
        mock_import.return_value = mock_module

        # Should not raise exception
        main(["analyse", "timesteps", "test.log"])

        mock_parse.assert_called_once()
        mock_import.assert_called_once_with("swiftsim_cli.modes.analyse")
        mock_module.run.assert_called_once_with(mock_args)

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    def test_main_no_args(self, mock_parse):
        """Test CLI main function with no arguments."""
        main()
        mock_parse.assert_called_once_with(None)

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    def test_main_with_args(self, mock_parse):
        """Test CLI main function with specific arguments."""
        test_args = ["config", "--enable-debug"]
        main(test_args)
        mock_parse.assert_called_once_with(test_args)

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    @patch("importlib.import_module")
    def test_main_multiple_modes(self, mock_import, mock_parse):
        """Test CLI with multiple modes."""
        # Mock args for multiple modes
        mock_args = Mock()
        mock_args.mode = "config"
        mock_parse.return_value = mock_args

        # Mock the mode module
        mock_module = Mock()
        mock_import.return_value = mock_module

        main(["config", "--enable-debug", "make", "-j", "4"])

        mock_parse.assert_called_once()
        mock_import.assert_called_once_with("swiftsim_cli.modes.config")
        mock_module.run.assert_called_once_with(mock_args)

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    @patch("importlib.import_module")
    def test_main_import_error(self, mock_import, mock_parse):
        """Test CLI main function with import error."""
        mock_args = Mock()
        mock_args.mode = "invalid_mode"
        mock_parse.return_value = mock_args

        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ImportError):
            main(["invalid_mode"])

    @patch("swiftsim_cli.multi_mode_args.parse_multimode_args")
    @patch("importlib.import_module")
    def test_main_mode_execution_error(self, mock_import, mock_parse):
        """Test CLI main function with mode execution error."""
        mock_args = Mock()
        mock_args.mode = "analyse"
        mock_parse.return_value = mock_args

        mock_module = Mock()
        mock_module.run.side_effect = Exception("Mode execution failed")
        mock_import.return_value = mock_module

        with pytest.raises(Exception):
            main(["analyse", "timesteps", "test.log"])
