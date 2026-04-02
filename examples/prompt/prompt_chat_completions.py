# /// script
# description = "Prompt an OpenAI model"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.6","pydantic","python-dotenv", "numpy"]
# ///
import os

from dotenv import load_dotenv

import daft
from daft.functions import format, prompt

if __name__ == "__main__":
    load_dotenv()

    daft.set_provider(
        "openai",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # Create a dataframe with the quotes
    df = (
        daft.read_huggingface("nvidia/Nemotron-Personas-USA")
        # Craft Prompt Template
        .with_column(
            "prompt",
            format(
                "Asumming the persona of <persona>{}</persona> answer the following question: <question>{}</question> ",
                daft.col("professional_persona"),
                daft.lit("Who will win the race to AGI?"),
            ),
        )
        # Prompt with Chat Completions
        .with_column(
            "response",
            prompt(
                messages=daft.col("prompt"),
                system_message="Impersonating the persona provided, authentically represent your perspective to the prompt posed.",
                model="google/gemini-2.5-flash",
                use_chat_completions=True,
                max_tokens=100,
            ),
        )
    )

    # Show the results
    df = df.select("professional_persona", "response").limit(100)
    df.write_json("../../.data/prompt/prompt_chat_completions.json")
    df.show(format="fancy", max_width=60)
