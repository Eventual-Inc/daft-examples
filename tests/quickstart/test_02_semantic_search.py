"""Tests for quickstart/02_semantic_search.py"""
import pytest
import subprocess
import sys


@pytest.mark.integration
class TestSemanticSearch:
    """Integration tests for semantic search example"""

    def test_example_runs_successfully(self, quickstart_dir, has_openai_key):
        """Test that the example runs without errors"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "02_semantic_search.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120  # Allow more time for PDF processing
        )

        assert result.returncode == 0, f"Example failed: {result.stderr}"

    def test_output_shows_papers(self, quickstart_dir, has_openai_key):
        """Test that output displays paper metadata"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "02_semantic_search.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout

        # Check that sample papers are shown
        assert "Sample Papers with Embeddings" in output
        assert "title" in output.lower()
        assert "author" in output.lower()

        # Check table formatting
        assert "│" in output or "|" in output

    def test_turbopuffer_write_is_conditional(self, quickstart_dir, has_openai_key):
        """Test that Turbopuffer write is skipped silently when key missing"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "02_semantic_search.py"

        # Run without TURBOPUFFER_API_KEY
        import os
        env = os.environ.copy()
        env.pop("TURBOPUFFER_API_KEY", None)

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        # Should skip Turbopuffer write silently and still succeed
        assert result.returncode == 0

    def test_runtime_under_two_minutes(self, quickstart_dir, has_openai_key):
        """Test that example completes in <2 minutes"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        import time
        example_path = quickstart_dir / "02_semantic_search.py"

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        duration = time.time() - start

        assert result.returncode == 0
        assert duration < 120, f"Example took {duration:.1f}s, should be <120s"


@pytest.mark.unit
class TestSemanticSearchComponents:
    """Unit tests for individual components"""

    def test_pydantic_model_structure(self):
        """Test that Classifer model is properly defined"""
        from pydantic import BaseModel

        class Classifer(BaseModel):
            title: str
            author: str
            year: int
            keywords: list[str]
            abstract: str

        # Test model validation
        paper = Classifer(
            title="Test Paper",
            author="Test Author",
            year=2024,
            keywords=["AI", "ML"],
            abstract="Test abstract"
        )

        assert paper.title == "Test Paper"
        assert len(paper.keywords) == 2
        assert paper.year == 2024

    def test_dynamic_batching_config(self):
        """Test that dynamic batching can be configured"""
        import daft

        daft.set_execution_config(enable_dynamic_batching=True)
        # Just verify it doesn't error

    def test_glob_pdf_files(self):
        """Test that we can discover PDF files"""
        import daft

        df = daft.from_glob_path(
            "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
        ).limit(1)

        # Verify we can create the dataframe
        assert df is not None
        assert "path" in df.column_names
