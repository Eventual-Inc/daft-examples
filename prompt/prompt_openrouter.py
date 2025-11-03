# /// script
# description = "Prompt an OpenRouter model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv"]
# ///
import os
from dotenv import load_dotenv
import daft
from daft.ai.openai.provider import OpenAIProvider
from daft.functions.ai import prompt
from daft.session import Session
from pydantic import BaseModel, Field


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Create an OpenRouter provider
    openrouter_provider = OpenAIProvider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Create a dataframe with the quotes
    df = daft.from_pydict(
        {
            "quote": [
                "I am going to be the king of the pirates!",
                "I'm going to be the next Hokage!",
            ],
        }
    )

    # Use the prompt function to classify the quotes
    df = df.with_column(
        "response",
        prompt(
            daft.col("quote"),
            system_message="You are an anime expert. Classify the anime based on the text and returns the name, character, and quote.",
            provider=openrouter_provider,
            model="nvidia/nemotron-nano-9b-v2:free",
        ),
    )

    df.show(format="fancy", max_width=120)
