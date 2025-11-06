"""Tests for the profile mode module."""

import argparse
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.profile import add_arguments, run


def test_add_arguments():
    """Test that add_arguments adds all profile arguments."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    # Test defaults
    args = parser.parse_args([])
    assert args.init is False
    assert args.clear is False
    assert args.override is False
    assert args.new is None
    assert args.switch is None
    assert args.list is False
    assert args.show is False

    # Test --init flag
    args = parser.parse_args(["--init"])
    assert args.init is True

    # Test --clear flag
    args = parser.parse_args(["--clear"])
    assert args.clear is True

    # Test --override flag
    args = parser.parse_args(["--override"])
    assert args.override is True

    # Test --new flag
    args = parser.parse_args(["--new", "myprofile"])
    assert args.new == "myprofile"

    # Test --switch flag
    args = parser.parse_args(["--switch", "otherprofile"])
    assert args.switch == "otherprofile"

    # Test --list flag
    args = parser.parse_args(["--list"])
    assert args.list is True

    # Test --show flag
    args = parser.parse_args(["--show"])
    assert args.show is True

    # Test short forms
    args = parser.parse_args(["-i"])
    assert args.init is True

    args = parser.parse_args(["-c"])
    assert args.clear is True

    args = parser.parse_args(["-o"])
    assert args.override is True

    args = parser.parse_args(["-n", "test"])
    assert args.new == "test"

    args = parser.parse_args(["-s", "test"])
    assert args.switch == "test"

    args = parser.parse_args(["-l"])
    assert args.list is True

    args = parser.parse_args(["-S"])
    assert args.show is True


@patch("swiftsim_cli.modes.profile.initial_profile_profile")
def test_run_init(mock_init):
    """Test running profile init."""
    args = Mock()
    args.init = True
    args.clear = False
    args.override = False
    args.new = None
    args.switch = None
    args.edit = None
    args.show = False
    args.list = False

    run(args)

    mock_init.assert_called_once()


@patch("swiftsim_cli.modes.profile.clear_swift_profile")
def test_run_clear(mock_clear):
    """Test running profile clear."""
    args = Mock()
    args.init = False
    args.clear = True
    args.override = False
    args.new = None
    args.switch = None
    args.edit = None
    args.show = False
    args.list = False

    run(args)

    mock_clear.assert_called_once()


@patch("swiftsim_cli.modes.profile.save_current_profile_as_default")
def test_run_override(mock_override):
    """Test running profile override."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = True
    args.new = None
    args.switch = None
    args.edit = None
    args.show = False
    args.list = False

    run(args)

    mock_override.assert_called_once()


@patch("swiftsim_cli.modes.profile.new_profile")
def test_run_new(mock_new):
    """Test running profile new."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = False
    args.new = "test_profile"
    args.switch = None
    args.edit = None
    args.show = False
    args.list = False

    run(args)

    mock_new.assert_called_once_with("test_profile")


@patch("swiftsim_cli.modes.profile.switch_profile")
def test_run_switch(mock_switch):
    """Test running profile switch."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = False
    args.new = None
    args.switch = "other_profile"
    args.edit = None
    args.show = False
    args.list = False

    run(args)

    mock_switch.assert_called_once_with("other_profile")


@patch("swiftsim_cli.modes.profile.edit_profile")
def test_run_edit(mock_edit):
    """Test running profile edit."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = False
    args.new = None
    args.switch = None
    args.edit = "profile_to_edit"
    args.show = False
    args.list = False

    run(args)

    mock_edit.assert_called_once_with("profile_to_edit")


@patch("swiftsim_cli.modes.profile.display_profile")
def test_run_show(mock_show):
    """Test running profile show."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = False
    args.new = None
    args.switch = None
    args.edit = None
    args.show = True
    args.list = False

    run(args)

    mock_show.assert_called_once()


@patch("swiftsim_cli.modes.profile.list_profiles")
def test_run_list(mock_list):
    """Test running profile list."""
    args = Mock()
    args.init = False
    args.clear = False
    args.override = False
    args.new = None
    args.switch = None
    args.edit = None
    args.show = False
    args.list = True

    run(args)

    mock_list.assert_called_once()


class TestProfileFunctions:
    """Test profile management functions."""

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    @patch("swiftsim_cli.modes.profile._save_swift_profile")
    @patch("swiftsim_cli.modes.profile.get_cli_profiles")
    def test_new_profile_success(self, mock_get_cli, mock_save, mock_load_all):
        """Test creating a new profile."""
        from swiftsim_cli.modes.profile import new_profile

        # Mock existing profiles (without the new one)
        mock_load_all.return_value = {"Default": {"swiftsim_dir": "/test"}}
        mock_get_cli.return_value = Mock()

        new_profile("test_profile")

        mock_get_cli.assert_called_once()
        mock_save.assert_called_once()

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    def test_new_profile_empty_name(self, mock_load_all):
        """Test error when creating profile with empty name."""
        from swiftsim_cli.modes.profile import new_profile

        mock_load_all.return_value = {}

        with pytest.raises(ValueError, match="Profile name cannot be empty"):
            new_profile("")

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    def test_new_profile_already_exists(self, mock_load_all):
        """Test error when profile already exists."""
        from swiftsim_cli.modes.profile import new_profile

        mock_load_all.return_value = {"existing": {}}

        with pytest.raises(
            ValueError, match="Profile 'existing' already exists"
        ):
            new_profile("existing")

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    @patch("swiftsim_cli.modes.profile._load_swift_profile")
    @patch("swiftsim_cli.modes.profile._save_swift_profile")
    @patch("swiftsim_cli.modes.profile.display_profile")
    def test_switch_profile_success(
        self, mock_display, mock_save, mock_load, mock_load_all
    ):
        """Test switching to an existing profile."""
        from swiftsim_cli.modes.profile import switch_profile

        mock_load_all.return_value = {"test": {"swiftsim_dir": "/test"}}
        mock_load.return_value = Mock()

        switch_profile("test")

        mock_load.assert_called_once_with("test")
        mock_save.assert_called_once()
        mock_display.assert_called_once()

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    def test_switch_profile_not_found(self, mock_load_all):
        """Test error when switching to non-existent profile."""
        from swiftsim_cli.modes.profile import switch_profile

        mock_load_all.return_value = {"existing": {}}

        with pytest.raises(
            ValueError, match="Profile 'nonexistent' does not exist"
        ):
            switch_profile("nonexistent")

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    def test_list_profiles_empty(self, mock_load_all, capsys):
        """Test listing profiles when none exist."""
        from swiftsim_cli.modes.profile import list_profiles

        mock_load_all.return_value = {}

        list_profiles()

        captured = capsys.readouterr()
        assert "No profiles found" in captured.out

    @patch("swiftsim_cli.modes.profile._load_all_profiles")
    def test_list_profiles_with_profiles(self, mock_load_all, capsys):
        """Test listing existing profiles."""
        from swiftsim_cli.modes.profile import list_profiles

        mock_load_all.return_value = {
            "Current": {},
            "Default": {},
            "test_profile": {},
        }

        list_profiles()

        captured = capsys.readouterr()
        assert "Default" in captured.out
        assert "test_profile" in captured.out
        # Current should be skipped
        assert "- Current" not in captured.out
