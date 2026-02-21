# /// script
# description = "Embed Video Frames from a Youtube Video"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv", "numpy"]
# ///
import os
import daft
from daft.functions import unnest, prompt
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

daft.set_provider(
    "openai",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


class Anime(BaseModel):
    show: str = Field(description="The name of the anime show")
    character: str = Field(description="The name of the character who says the quote")
    explanation: str = Field(description="Why the character says the quote")


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
        system_message="You are an anime expert. Classify the anime based on the text and return the name, character, and quote.",
        return_format=Anime,
        model="nvidia/nemotron-nano-9b-v2:free",
        use_chat_completions=True,
    ),
).select("quote", unnest(daft.col("response")))

df.show(format="fancy", max_width=80)
