# This is a simple hello world example showcasing Daft Cloud
#
# In this example, we will work with the Commoncrawl dataset (web scraping data)
# and run some simple LLM-powered summarization/classifcation
#
# To try this example locally, run this as you would a Python script. This script is
# configured to run a very small subset of 3 rows of fake data:
#
# `uv run hello_world.py`
#
# Now to run the full job on the actualy CommonCrawl dataset, use the Daft CLI and point
# your function at our publicly hosted CommonCrawl dataset:
#
# `daft run hello_world.py:classify_commoncrawl_taxonomy --data daft-public://commoncrawl`
#
# This now runs your workload as a managed job in Daft Cloud on 1M rows instead. The
# result is computed and stored by default as JSON files which you can retrieve when your run
# completes


import daft


def commoncrawl_example(data: daft.DataFrame, num_rows: int = 1_000_000):
    data = data.limit(num_rows)
    data = data.with_column(
        "summary",
        daft.prompt(
            system_message="You are an expert web scraper summarizer.",
            content=data["content"],
        )
    )
    data = data.with_column(
        "taxonomy",
        daft.classify(
            system_message="You are an expert web scraper classifier.",
            content=data["content"],
            labels=["science", "literature", "music"],
        )
    )
    data = data.with_column(
        "embedding",
        daft.embed_text(content=data["content"], model="qwen/0.8b")
    )
    return data


if __name__ == "__main__":
    data = daft.from_pydict({"content": [
        "Einstein was a brilliant scientist."
        "Shakespeare was a brilliant writer",
        "Mozart was a brilliant pianist.",
    ]})
    result = commoncrawl_example(data=data)
    result.show()
