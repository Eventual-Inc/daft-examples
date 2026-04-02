# /// script
# description = "Prompt with PDF files"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.6", "numpy", "python-dotenv"]
# ///
from dotenv import load_dotenv

import daft
from daft import col, lit
from daft.functions import file, prompt

if __name__ == "__main__":
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
                messages=[
                    lit("Immediately detail the core points of the attached paper."),
                    col("file"),
                ],
                model="gpt-5-nano",
                provider="openai",
                reasoning={"effort": "high"},
            ),
        )
    )
    df.select("path", "size", "response").show(3, format="fancy", max_width=80)
