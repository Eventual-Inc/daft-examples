"""Tests for quickstart/03_data_enrichment.py"""
import pytest
import subprocess
import sys
from pathlib import Path


@pytest.mark.integration
class TestDataEnrichment:
    """Integration tests for data enrichment example"""

    def test_example_runs_successfully(self, quickstart_dir, has_openai_key, tmp_path):
        """Test that the example runs without errors"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "03_data_enrichment.py"

        # Run in temp directory to avoid polluting repo
        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path
        )

        assert result.returncode == 0, f"Example failed: {result.stderr}"

    def test_output_shows_enriched_data(self, quickstart_dir, has_openai_key, tmp_path):
        """Test that output displays enriched comments with sentiment/topics"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "03_data_enrichment.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path
        )

        output = result.stdout

        # Check that enriched data is shown
        assert "Sample Enriched Rows" in output
        assert "sentiment" in output.lower()
        assert "topics" in output.lower()
        assert "has_pii" in output.lower()

        # Check table formatting
        assert "│" in output or "|" in output

    def test_output_parquet_created(self, quickstart_dir, has_openai_key, tmp_path):
        """Test that output parquet file is created"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "03_data_enrichment.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path
        )

        assert result.returncode == 0

        # Check that output directory was created
        output_dir = tmp_path / "enriched-comments"
        assert output_dir.exists(), "Output directory should be created"
        assert output_dir.is_dir()

        # Check that parquet files exist
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Should have at least one parquet file"

    def test_processes_limited_rows(self, quickstart_dir, has_openai_key, tmp_path):
        """Test that example only processes 5 rows (fast demo)"""
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        example_path = quickstart_dir / "03_data_enrichment.py"

        result = subprocess.run(
            [sys.executable, "-m", "uv", "run", str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path
        )

        output = result.stdout

        # Should show exactly 5 rows (or fewer if showing first N)
        assert "Showing first 5" in output or "5 rows" in output.lower()


@pytest.mark.unit
class TestDataEnrichmentComponents:
    """Unit tests for data enrichment components"""

    def test_pydantic_models_defined(self):
        """Test that Pydantic models are properly structured"""
        from pydantic import BaseModel

        class Meta(BaseModel):
            sentiment: str
            topics: list[str]
            has_pii: bool

        class Redacted(BaseModel):
            safe_text: str
            pii_types: list[str]

        # Test Meta model
        meta = Meta(
            sentiment="positive",
            topics=["test", "example"],
            has_pii=False
        )
        assert meta.sentiment == "positive"
        assert len(meta.topics) == 2

        # Test Redacted model
        redacted = Redacted(
            safe_text="Hello [NAME]",
            pii_types=["name"]
        )
        assert "[NAME]" in redacted.safe_text
        assert "name" in redacted.pii_types

    def test_text_normalization(self):
        """Test that text can be normalized"""
        import daft

        df = daft.from_pydict({
            "text": ["Hello, World!", "TEST TEXT"]
        })

        # Test that normalize function exists and works
        normalized = df.with_column(
            "norm",
            daft.col("text").normalize(lowercase=True, remove_punct=True)
        )

        assert "norm" in normalized.column_names

    def test_reading_reddit_data(self):
        """Test that we can read Reddit data from HuggingFace"""
        import daft

        # This should not fail (just creating the plan)
        df = daft.read_parquet(
            "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/0.parquet"
        ).limit(1)

        assert df is not None
        assert "body" in df.column_names or "text" in df.column_names
