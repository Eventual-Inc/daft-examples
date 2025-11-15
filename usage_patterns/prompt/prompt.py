# /// script
# description = "Prompt a model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.12","openai","pydantic","python-dotenv", "numpy"]
# ///
from dotenv import load_dotenv
import daft
from daft.functions import prompt

load_dotenv()

df = daft.from_pydict({
    "quote": [
        "I am going to be the king of the pirates!",
        "I'm going to be the next Hokage!",
    ],
})

df = (
    df
    .with_column(
        "response",
        prompt(
            daft.col("quote"),
            system_message="Classify the anime from the quote and return the show, character name, and explanation.",
            provider="openai",
            model="gpt-5-nano"
        )
    )
)

df.show(format="fancy", max_width=120)
