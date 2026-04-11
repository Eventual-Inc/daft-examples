# /// script
# description = "Synthetic Q&A generation pipeline with question generation, answering, and verification"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[unity, deltalake, openai]>=0.7.8", "pydantic", "numpy", "pillow", "python-dotenv", "tenacity"]
# ///
import os

from dotenv import load_dotenv

import daft
from daft.catalog import Table
from daft.unity_catalog import UnityCatalog

if __name__ == "__main__":
    load_dotenv()

    # Define Configuration
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_ENDPOINT = os.getenv("DATABRICKS_ENDPOINT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    UC_TABLE_NAME = os.getenv("UC_TABLE_NAME")

    # Configure UnityCatalog
    unity = UnityCatalog(
        endpoint=DATABRICKS_ENDPOINT,
        token=DATABRICKS_TOKEN,
    )

    # Configure OpenAI Provider
    daft.set_provider("openai", api_key=OPENAI_API_KEY)

    coco_table = unity.load_table(UC_TABLE_NAME)

    tbl = Table.from_unity(coco_table)
    df = daft.read_table(tbl)
