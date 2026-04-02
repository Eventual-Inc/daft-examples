"""Shared pytest configuration for daft-examples."""

from dotenv import load_dotenv

# Load .env once at collection time so env-based skips work.
load_dotenv()
