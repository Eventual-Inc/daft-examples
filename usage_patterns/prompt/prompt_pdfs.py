# /// script
# description = "Prompt with PDF files"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13","openai", "numpy", "python-dotenv"]
# ///
import daft 
from daft import lit, col
from daft.functions import prompt, file
from dotenv import load_dotenv

load_dotenv()

# Discover Markdown Files in your Documents Folder
df = daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")

df = (
    df
    # Create a daft.File column from the path
    .with_column("file", file(col("path")))
    # Prompt GPT-5-nano with PDF files as context
    .with_column(
        "response", 
        prompt(
            messages=[lit("Immediately detail the core points of the attached paper."), col("file")], 
            model="gpt-5-nano", 
            provider="openai",
            reasoning={"effort": "high"},
        )
    )
)
df.select("path", "size", "response").show(3, format="fancy", max_width=80)




