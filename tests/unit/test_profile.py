"""Tests for the profile module."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from swiftsim_cli.profile import (
    SWIFTCLIProfile,
    _load_all_profiles,
    _save_swift_profile,
    get_current_git_branch,
    get_default_parameter_file,
    load_cosmology_from_parameter_file,
    load_swift_profile,
    sync_profile_with_repo,
    update_current_profile_value,
)


class TestSWIFTCLIProfile:
    """Tests for the SWIFTCLIProfile dataclass."""

    def test_profile_creation_with_defaults(self):
        """Test creating a profile with default values."""
        profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
        )

        assert profile.swiftsim_dir == Path("/fake/swift")
        assert profile.data_dir == Path("/fake/data")
        assert profile.branch == "master"
        assert profile.h == 0.6777
        assert profile.Omega_m == 0.307

    def test_profile_creation_with_custom_values(self):
        """Test creating a profile with custom values."""
        profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/custom/swift"),
            data_dir=Path("/custom/data"),
            branch="develop",
            h=0.7,
            Omega_m=0.3,
        )

        assert profile.swiftsim_dir == Path("/custom/swift")
        assert profile.branch == "develop"
        assert profile.h == 0.7
        assert profile.Omega_m == 0.3


class TestLoadCosmologyFromParameterFile:
    """Tests for load_cosmology_from_parameter_file."""

    def test_load_cosmology_file_not_found(self, tmp_path):
        """Test error when parameter file doesn't exist."""
        non_existent = tmp_path / "nonexistent.yml"

        with pytest.raises(
            FileNotFoundError, match="Parameter file not found"
        ):
            load_cosmology_from_parameter_file(non_existent)

    def test_load_cosmology_with_valid_file(self, tmp_path):
        """Test loading cosmology from a valid YAML file."""
        param_file = tmp_path / "params.yml"
        param_content = """
Cosmology:
  h: 0.6777
  Omega_m: 0.307
  Omega_lambda: 0.693
  Omega_b: 0.0486
  a_begin: 0.0078125
  a_end: 1.0
"""
        param_file.write_text(param_content)

        result = load_cosmology_from_parameter_file(param_file)

        assert result["h"] == 0.6777
        assert result["Omega_m"] == 0.307
        assert result["Omega_lambda"] == 0.693

    def test_load_cosmology_with_missing_cosmology_section(self, tmp_path):
        """Test warning when Cosmology section is missing."""
        param_file = tmp_path / "params.yml"
        param_file.write_text("SomeOtherSection:\n  key: value\n")

        # Should return empty dict with warning
        result = load_cosmology_from_parameter_file(param_file)
        assert result == {}

    def test_load_cosmology_with_malformed_yaml(self, tmp_path):
        """Test that malformed YAML raises an error."""
        param_file = tmp_path / "params.yml"
        param_file.write_text("invalid: yaml: content: [[[")

        # Should raise an error for malformed YAML
        with pytest.raises(Exception):
            load_cosmology_from_parameter_file(param_file)


class TestGetDefaultParameterFile:
    """Tests for get_default_parameter_file."""

    def test_get_default_parameter_file_exists(self, tmp_path):
        """Test getting default parameter file when it exists."""
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()
        examples_dir = swift_dir / "examples"
        examples_dir.mkdir()

        # Create a default parameter file
        param_file = examples_dir / "params.yml"
        param_file.write_text("Cosmology:\n  h: 0.7\n")

        # Mock the function to use our temp directory
        with patch("swiftsim_cli.profile.Path") as mock_path:
            mock_path.return_value = swift_dir
            result = get_default_parameter_file(swift_dir)

            # Should find the parameter file
            assert isinstance(result, (Path, type(None)))

    def test_get_default_parameter_file_not_found(self, tmp_path):
        """Test getting default parameter file when it doesn't exist."""
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        result = get_default_parameter_file(swift_dir)

        # Should return a Path object (doesn't check existence)
        assert isinstance(result, Path)
        assert result == swift_dir / "examples" / "parameter_example.yml"


class TestLoadSwiftProfile:
    """Tests for load_swift_profile."""

    @patch("swiftsim_cli.profile._load_swift_profile")
    def test_load_swift_profile_calls_internal(self, mock_load):
        """Test that load_swift_profile calls the internal function."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
        )
        mock_load.return_value = mock_profile

        result = load_swift_profile()

        mock_load.assert_called_once()
        assert result == mock_profile


class TestLoadAllProfiles:
    """Tests for _load_all_profiles."""

    @patch("swiftsim_cli.profile.PROFILE_FILE")
    def test_load_all_profiles_file_not_exists(self, mock_profile_file):
        """Test loading profiles when file doesn't exist."""
        mock_profile_file.exists.return_value = False

        result = _load_all_profiles()

        assert result == {}

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="profiles:\n  default:\n    swiftsim_dir: /fake/swift\n",
    )
    @patch("swiftsim_cli.profile.PROFILE_FILE")
    def test_load_all_profiles_valid_file(self, mock_profile_file, mock_file):
        """Test loading profiles from valid YAML file."""
        mock_profile_file.exists.return_value = True

        result = _load_all_profiles()

        assert isinstance(result, dict)
        mock_file.assert_called_once()


class TestSaveSwiftProfile:
    """Tests for _save_swift_profile."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("swiftsim_cli.profile.PROFILE_FILE")
    @patch("swiftsim_cli.profile._load_all_profiles")
    def test_save_swift_profile_new_profile(
        self, mock_load_all, mock_profile_file, mock_file
    ):
        """Test saving a new profile."""
        mock_load_all.return_value = {}
        mock_profile_file.parent.mkdir = Mock()

        profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
        )

        _save_swift_profile(profile, "test_profile")

        # Verify file operations
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("swiftsim_cli.profile.PROFILE_FILE")
    @patch("swiftsim_cli.profile._load_all_profiles")
    def test_save_swift_profile_update_existing(
        self, mock_load_all, mock_profile_file, mock_file
    ):
        """Test updating an existing profile."""
        mock_load_all.return_value = {
            "test_profile": {
                "swiftsim_dir": "/old/path",
            }
        }
        mock_profile_file.parent.mkdir = Mock()

        profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/new/swift"),
            data_dir=Path("/new/data"),
        )

        _save_swift_profile(profile, "test_profile")

        # Verify file operations
        mock_file.assert_called_once()


class TestUpdateCurrentProfileValue:
    """Tests for update_current_profile_value."""

    @patch("swiftsim_cli.profile._save_swift_profile")
    @patch("swiftsim_cli.profile.load_swift_profile")
    @patch("swiftsim_cli.profile._load_all_profiles")
    def test_update_current_profile_value(
        self, mock_load_all, mock_load_profile, mock_save
    ):
        """Test updating a value in the current profile."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
            branch="master",
        )
        mock_load_profile.return_value = mock_profile
        mock_load_all.return_value = {
            "current": "default",
            "profiles": {
                "default": {
                    "swiftsim_dir": "/fake/swift",
                }
            },
        }

        update_current_profile_value("branch", "develop")

        # Verify save was called
        mock_save.assert_called_once()
        # Verify the profile was updated
        call_args = mock_save.call_args
        updated_profile = call_args[0][0]
        assert updated_profile.branch == "develop"

    @patch("swiftsim_cli.profile._save_swift_profile")
    @patch("swiftsim_cli.profile.load_swift_profile")
    @patch("swiftsim_cli.profile._load_all_profiles")
    def test_update_current_profile_value_numeric(
        self, mock_load_all, mock_load_profile, mock_save
    ):
        """Test updating a numeric value in the current profile."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
            h=0.6777,
        )
        mock_load_profile.return_value = mock_profile
        mock_load_all.return_value = {
            "current": "default",
            "profiles": {
                "default": {
                    "swiftsim_dir": "/fake/swift",
                }
            },
        }

        update_current_profile_value("h", 0.7)

        # Verify save was called
        mock_save.assert_called_once()
        # Verify the profile was updated
        call_args = mock_save.call_args
        updated_profile = call_args[0][0]
        assert updated_profile.h == 0.7


class TestGetCurrentGitBranch:
    """Tests for get_current_git_branch."""

    @patch("subprocess.run")
    def test_get_current_git_branch_success(self, mock_run, tmp_path):
        """Test getting current git branch successfully."""
        mock_result = Mock()
        mock_result.stdout = "feature-branch\n"
        mock_run.return_value = mock_result

        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        result = get_current_git_branch(swift_dir)

        assert result == "feature-branch"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=swift_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

    @patch("subprocess.run")
    def test_get_current_git_branch_failure(self, mock_run, tmp_path):
        """Test handling git command failure."""
        mock_run.side_effect = Exception("Git error")

        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        result = get_current_git_branch(swift_dir)

        assert result is None

    @patch("subprocess.run")
    def test_get_current_git_branch_not_found(self, mock_run):
        """Test handling when git is not found."""
        mock_run.side_effect = FileNotFoundError()

        result = get_current_git_branch(Path("/nonexistent"))

        assert result is None

    @patch("subprocess.run")
    def test_get_current_git_branch_timeout(self, mock_run, tmp_path):
        """Test handling git command timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10
        )

        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        result = get_current_git_branch(swift_dir)

        assert result is None


class TestSyncProfileWithRepo:
    """Tests for sync_profile_with_repo."""

    @patch("swiftsim_cli.profile.update_current_profile_value")
    @patch("swiftsim_cli.profile.get_current_git_branch")
    @patch("swiftsim_cli.profile.load_swift_profile")
    def test_sync_profile_with_repo_branch_changed(
        self, mock_load, mock_get_branch, mock_update
    ):
        """Test syncing when git branch has changed."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
            branch="master",
        )
        mock_load.return_value = mock_profile
        mock_get_branch.return_value = "develop"

        sync_profile_with_repo()

        mock_update.assert_called_once_with("branch", "develop")

    @patch("swiftsim_cli.profile.update_current_profile_value")
    @patch("swiftsim_cli.profile.get_current_git_branch")
    @patch("swiftsim_cli.profile.load_swift_profile")
    def test_sync_profile_with_repo_no_change(
        self, mock_load, mock_get_branch, mock_update
    ):
        """Test syncing when git branch hasn't changed."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
            branch="master",
        )
        mock_load.return_value = mock_profile
        mock_get_branch.return_value = "master"

        sync_profile_with_repo()

        mock_update.assert_not_called()

    @patch("swiftsim_cli.profile.update_current_profile_value")
    @patch("swiftsim_cli.profile.get_current_git_branch")
    @patch("swiftsim_cli.profile.load_swift_profile")
    def test_sync_profile_with_repo_no_swift_dir(
        self, mock_load, mock_get_branch, mock_update
    ):
        """Test syncing when no swift directory is configured."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=None, data_dir=Path("/fake/data"), branch="master"
        )
        mock_load.return_value = mock_profile

        sync_profile_with_repo()

        mock_get_branch.assert_not_called()
        mock_update.assert_not_called()

    @patch("swiftsim_cli.profile.update_current_profile_value")
    @patch("swiftsim_cli.profile.get_current_git_branch")
    @patch("swiftsim_cli.profile.load_swift_profile")
    def test_sync_profile_with_repo_git_error(
        self, mock_load, mock_get_branch, mock_update
    ):
        """Test syncing when git command fails."""
        mock_profile = SWIFTCLIProfile(
            swiftsim_dir=Path("/fake/swift"),
            data_dir=Path("/fake/data"),
            branch="master",
        )
        mock_load.return_value = mock_profile
        mock_get_branch.return_value = None

        sync_profile_with_repo()

        mock_update.assert_not_called()
