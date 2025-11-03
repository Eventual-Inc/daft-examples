# /// script
# description = "Prompt an OpenAI model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv"]
# ///
from dotenv import load_dotenv
import daft
from daft.functions.ai import prompt

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

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
            provider="openai",  # Automatically creates an OpenAI provider
            model="gpt-5-nano",
        ),
    )

    df.show(format="fancy", max_width=120)
