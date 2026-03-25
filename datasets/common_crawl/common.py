"""Shared helpers for Common Crawl example scripts."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from daft.io import IOConfig, S3Config


def get_common_crawl_io() -> tuple[bool, IOConfig | None]:
    """Return (in_aws, io_config) for Common Crawl requester-pays access."""
    load_dotenv()

    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if key_id and access_key:
        return (
            True,
            IOConfig(
                s3=S3Config(
                    region_name="us-east-1",
                    requester_pays=True,
                    key_id=key_id,
                    access_key=access_key,
                    anonymous=False,
                )
            ),
        )

    return False, None
