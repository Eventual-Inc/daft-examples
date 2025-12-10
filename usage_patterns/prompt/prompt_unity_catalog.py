# /// script
# description = "Synthetic Q&A generation pipeline with question generation, answering, and verification"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[unity, deltalake]", "pydantic", "openai", "numpy", "pillow", "python-dotenv"]
# ///
import os
import daft 
from daft.catalog import Table
from daft.unity_catalog import UnityCatalog
from daft.functions import prompt
from dotenv import load_dotenv

load_dotenv()

# Define Configuration 
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_ENDPOINT = os.getenv("DATABRICKS_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TABLE_NAME = os.getenv("TABLE_NAME")

# Configure UnityCatalog
unity = UnityCatalog(
    endpoint=DATABRICKS_ENDPOINT,
    token=DATABRICKS_TOKEN,
)

# Configure OpenAI Provider
daft.set_provider("openai", api_key=OPENAI_API_KEY)

coco_table = unity.load_table(TABLE_NAME)

tbl = Table.from_unity(coco_table)
df = daft.read_table(tbl)

