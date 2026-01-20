"""Shared pytest fixtures for daft-examples tests"""
import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def quickstart_dir(project_root):
    """Return the quickstart examples directory"""
    return project_root / "quickstart"


@pytest.fixture(scope="session")
def has_openai_key():
    """Check if OpenAI API key is available"""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def has_turbopuffer_key():
    """Check if Turbopuffer API key is available"""
    return bool(os.environ.get("TURBOPUFFER_API_KEY"))


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def load_env():
    """Automatically load .env file for all tests"""
    from dotenv import load_dotenv
    load_dotenv()


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing without API calls"""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "model": "gpt-5-mini",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Test response"
            },
            "finish_reason": "stop"
        }]
    }
