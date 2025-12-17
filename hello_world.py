# This is a simple hello world example showcasing Daft Cloud
#
# To run on Daft Cloud, navigate to the UI at `cloud.daft.ai` and launch a
# run from this script/function
# 
# Your results will be stored as JSON-lines files in Daft Cloud.

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
            model="gpt-4.1-mini",
        )
    )
    data = data.with_column(
        "embedding",
        embed_text(text=data["content"], model="Qwen/Qwen3-Embedding-0.6B")
    )
    return data
