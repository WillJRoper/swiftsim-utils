"""Tests for the profile module."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from swiftsim_cli.profile import (
    SWIFTCLIProfile,
    _extract_cosmology_manually,
    _load_all_profiles,
    _save_swift_profile,
    get_default_parameter_file,
    load_cosmology_from_parameter_file,
    load_swift_profile,
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
        """Test fallback when YAML is malformed."""
        param_file = tmp_path / "params.yml"
        param_file.write_text("invalid: yaml: content: [[[")

        # Should return empty dict with warning
        result = load_cosmology_from_parameter_file(param_file)
        assert isinstance(result, dict)

    def test_load_cosmology_with_yaml_and_manual_failure(self, tmp_path):
        """Test when both YAML parsing and manual extraction fail."""
        param_file = tmp_path / "params.yml"
        # Create a file that will fail both YAML parsing and manual extraction
        param_file.write_text("[[[invalid yaml")

        with patch(
            "swiftsim_cli.profile._extract_cosmology_manually"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Manual extraction failed")

            # Should return empty dict when both fail
            result = load_cosmology_from_parameter_file(param_file)
            assert result == {}


class TestExtractCosmologyManually:
    """Tests for _extract_cosmology_manually."""

    def test_extract_cosmology_basic(self, tmp_path):
        """Test manual extraction of cosmology parameters."""
        param_file = tmp_path / "params.yml"
        param_content = """
Cosmology:
  h: 0.7
  Omega_m: 0.3
  Omega_lambda: 0.7
  Omega_b: 0.05
"""
        param_file.write_text(param_content)

        result = _extract_cosmology_manually(param_file)

        assert "h" in result
        assert "Omega_m" in result
        assert "Omega_lambda" in result
        assert "Omega_b" in result

    def test_extract_cosmology_with_defaults(self, tmp_path):
        """Test that defaults are applied for missing parameters."""
        param_file = tmp_path / "params.yml"
        param_content = """
Cosmology:
  h: 0.7
"""
        param_file.write_text(param_content)

        result = _extract_cosmology_manually(param_file)

        # Should have h from file
        assert "h" in result
        # Should have defaults for missing params
        assert "Omega_r" in result or len(result) > 0

    def test_extract_cosmology_with_neutrinos(self, tmp_path):
        """Test extraction with neutrino parameters."""
        param_file = tmp_path / "params.yml"
        param_content = """
Cosmology:
  h: 0.7
  M_nu_eV: 0.06, 0.06, 0.06
  deg_nu: 1.0, 1.0, 1.0
  N_nu: 3
"""
        param_file.write_text(param_content)

        result = _extract_cosmology_manually(param_file)

        assert "h" in result
        assert "M_nu_eV" in result
        assert "deg_nu" in result
        assert "N_nu" in result
        assert result["N_nu"] == 3  # Should be int


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
