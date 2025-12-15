# This is a simple hello world example showcasing Daft Cloud
#
# To try this example locally, run this as you would a Python script. This script is
# configured to run a very small subset of 3 rows of fake data:
#
# `uv run hello_world.py`
#
# Now to run on Daft Cloud, navigate to the UI at `cloud.daft.ai`. Your results will
# be stored as JSON-lines files and be retrievable afterwards.

import os

import daft
from daft.functions import prompt, embed_text


# This function can be run on Daft Cloud
# Simply reference it like so: `hello_world.py:example`
def example():
    data = daft.from_pydict({"content": [
        "Einstein was a brilliant scientist."
        "Shakespeare was a brilliant writer",
        "Mozart was a brilliant pianist.",
    ]})
    data = data.with_column(
        "summary",
        prompt(
            system_message="You are an expert web scraper summarizer.",
            messages=data["content"],
        )
    )
    data = data.with_column(
        "embedding",
        embed_text(text=data["content"])
    )
    return data

# Run this as a script on your local laptop:
# Set your provider to hit Daft's managed inference platform
if __name__ == "__main__":
    daft.set_provider(
        "openai",
        base_url="https://inference.daft.ai",
        api_key=os.getenv("DAFT_API_KEY"),
    )
    result = example()
    result.show()
