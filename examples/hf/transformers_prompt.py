# /// script
# description = "Run a local Hugging Face instruction model through Daft prompt"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[transformers]==0.7.19"]
# ///

import daft
from daft.functions import prompt

requests = daft.from_pydict(
    {"request": ["Write one sentence explaining why lazy DataFrames help with large datasets."]}
)

(
    requests.with_column(
        "response",
        prompt(
            daft.col("request"),
            provider="transformers",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=80,
            do_sample=False,
        ),
    ).show()
)
