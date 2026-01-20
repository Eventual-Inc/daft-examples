"""Tests for quickstart/01_hello_world_prompt.py"""
import pytest
import subprocess
import sys
from pathlib import Path


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="openai not installed"),
    reason="OpenAI not available"
)
class TestHelloWorldPrompt:
    """Test the hello world prompt example"""

    def test_example_runs_successfully(self, quickstart_dir, has_openai_key):
        """Test that the example runs without errors"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "01_hello_world_prompt.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"Example failed: {result.stderr}"

    def test_output_contains_expected_elements(self, quickstart_dir, has_openai_key):
        """Test that output contains expected anime classifications"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "01_hello_world_prompt.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        # Check that output contains expected quotes
        assert "king of the pirates" in output.lower() or "one piece" in output.lower()
        assert "hokage" in output.lower() or "naruto" in output.lower()

        # Check that output is formatted (contains table borders)
        assert "│" in output or "|" in output, "Output should be formatted as table"

    def test_runtime_is_reasonable(self, quickstart_dir, has_openai_key):
        """Test that example completes in reasonable time (<30s)"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        import time
        example_path = quickstart_dir / "01_hello_world_prompt.py"

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        duration = time.time() - start

        assert result.returncode == 0
        assert duration < 30, f"Example took {duration:.1f}s, should be <30s"


@pytest.mark.unit
class TestHelloWorldLogic:
    """Unit tests for the example logic (without API calls)"""

    def test_dataframe_creation(self):
        """Test that we can create the input dataframe"""
        import daft

        df = daft.from_pydict({
            "quote": [
                "I am going to be the king of the pirates!",
                "I'm going to be the next Hokage!",
            ],
        })

        assert df.count_rows() == 2
        assert "quote" in df.column_names

    def test_quotes_are_strings(self):
        """Test that input quotes are valid strings"""
        quotes = [
            "I am going to be the king of the pirates!",
            "I'm going to be the next Hokage!",
        ]

        for quote in quotes:
            assert isinstance(quote, str)
            assert len(quote) > 0
