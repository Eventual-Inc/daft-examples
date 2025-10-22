# /// script
# description = "Classify text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "transformers"]
# ///

import daft
from daft.functions import classify_text

df = daft.from_pydict({"text": ["Daft is wicked fast!"]})

df = df.with_column(
    "label", 
    classify_text(
        daft.col("text"), 
        labels=["Positive", "Negative"],
        provider="transformers",
        model="tabularisai/multilingual-sentiment-analysis"
    )
)

df.show()