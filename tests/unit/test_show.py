"""Tests for the show mode."""

import argparse
from unittest.mock import patch

from swiftsim_cli.modes import show


class TestShowMode:
    """Tests for the show mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds cosmo and all flags."""
        parser = argparse.ArgumentParser()
        show.add_arguments(parser)

        # Test defaults
        args = parser.parse_args([])
        assert args.cosmo is False
        assert args.all is False

        # Test --cosmo flag
        args = parser.parse_args(["--cosmo"])
        assert args.cosmo is True
        assert args.all is False

        # Test --all flag
        args = parser.parse_args(["--all"])
        assert args.cosmo is False
        assert args.all is True

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_default(self, mock_display):
        """Test the run function with default args (profile only)."""
        args = argparse.Namespace(cosmo=False, all=False)

        show.run(args)

        mock_display.assert_called_once_with(
            show_profile=True, show_cosmology=False
        )

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_cosmo_only(self, mock_display):
        """Test the run function with --cosmo flag."""
        args = argparse.Namespace(cosmo=True, all=False)

        show.run(args)

        mock_display.assert_called_once_with(
            print_header=False,
            show_profile=False,
            show_cosmology=True,
        )

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_all(self, mock_display):
        """Test the run function with --all flag."""
        args = argparse.Namespace(cosmo=False, all=True)

        show.run(args)

        mock_display.assert_called_once_with(
            show_profile=True, show_cosmology=True
        )
