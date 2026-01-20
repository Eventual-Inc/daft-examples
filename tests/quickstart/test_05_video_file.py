"""Tests for quickstart/05_video_file.py"""
import pytest
import subprocess
import sys


@pytest.mark.integration
class TestVideoFile:
    """Integration tests for video file example"""

    def test_example_runs_successfully(self, quickstart_dir):
        """Test that the example runs without errors"""
        example_path = quickstart_dir / "05_video_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"Example failed: {result.stderr}"

    def test_output_shows_video_metadata(self, quickstart_dir):
        """Test that output displays video file metadata"""
        example_path = quickstart_dir / "05_video_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        # Check that video metadata is shown
        assert "metadata" in output.lower()

        # Check for video-specific metadata fields
        assert any(field in output.lower() for field in ["width", "height", "fps", "frame_count", "duration"])

        # Check table formatting
        assert "│" in output or "|" in output

        # Check that files are shown
        assert ".mp4" in output or "video" in output.lower()

    def test_processes_video_files(self, quickstart_dir):
        """Test that multiple video files are processed"""
        example_path = quickstart_dir / "05_video_file.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        # Should process multiple video files (shows "first 3" or similar)
        assert "3" in output or "first" in output.lower()


@pytest.mark.unit
class TestVideoFileComponents:
    """Unit tests for video file handling components"""

    def test_glob_video_files(self):
        """Test that we can discover video files"""
        import daft

        df = daft.from_glob_path(
            "hf://datasets/Eventual-Inc/sample-files/videos/*.mp4"
        )

        assert df is not None
        assert "path" in df.column_names

    def test_video_file_function_exists(self):
        """Test that video_file function is available"""
        from daft.functions import video_file

        assert callable(video_file)

    def test_video_metadata_function_exists(self):
        """Test that video_metadata function is available"""
        from daft.functions import video_metadata

        assert callable(video_metadata)

    def test_video_keyframes_function_exists(self):
        """Test that video_keyframes function is available"""
        from daft.functions import video_keyframes

        assert callable(video_keyframes)
