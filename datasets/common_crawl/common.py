# /// script
# description = "Shared helpers for Common Crawl example scripts"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6", "python-dotenv"]
# ///

"""Shared helpers for Common Crawl example scripts."""

from __future__ import annotations

from daft.io import IOConfig, S3Config


def get_common_crawl_io() -> tuple[bool, IOConfig]:
    """Return (in_aws, io_config) for Common Crawl access.

    Common Crawl data on s3://commoncrawl/ is publicly accessible.
    Always uses anonymous S3 access to avoid issues with invalid local credentials.
    """
    return False, IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))
