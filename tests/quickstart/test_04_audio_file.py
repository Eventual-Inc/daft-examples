"""Tests for quickstart/04_audio_file.py"""
import pytest
import subprocess
import sys


@pytest.mark.integration
class TestAudioFile:
    """Integration tests for audio file example"""

    def test_example_runs_successfully(self, quickstart_dir):
        """Test that the example runs without errors"""
        example_path = quickstart_dir / "04_audio_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"Example failed: {result.stderr}"

    def test_output_shows_audio_metadata(self, quickstart_dir):
        """Test that output displays audio file metadata"""
        example_path = quickstart_dir / "04_audio_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        # Check that audio metadata is shown
        assert "metadata" in output.lower()
        assert "sample_rate" in output.lower() or "resampled" in output.lower()

        # Check table formatting
        assert "│" in output or "|" in output

        # Check that files are shown
        assert ".mp3" in output or "audio" in output.lower()

    def test_processes_audio_files(self, quickstart_dir):
        """Test that multiple audio files are processed"""
        example_path = quickstart_dir / "04_audio_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        # Should process multiple audio files (shows "first 3")
        assert "3" in output or "first" in output.lower()


@pytest.mark.unit
class TestAudioFileComponents:
    """Unit tests for audio file handling components"""

    def test_glob_audio_files(self):
        """Test that we can discover audio files"""
        import daft

        df = daft.from_glob_path(
            "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
        )

        assert df is not None
        assert "path" in df.column_names

    def test_audio_file_function_exists(self):
        """Test that audio_file function is available"""
        from daft.functions import audio_file

        assert callable(audio_file)

    def test_audio_metadata_function_exists(self):
        """Test that audio_metadata function is available"""
        from daft.functions import audio_metadata

        assert callable(audio_metadata)

    def test_resample_function_exists(self):
        """Test that resample function is available"""
        from daft.functions import resample

        assert callable(resample)
