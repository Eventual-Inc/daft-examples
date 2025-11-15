# /// script
# description = "Classify text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "transformers", "torch"]
# ///
import daft
from daft.functions import classify_text

df = daft.from_pydict({
    "text": [
        "One day I will see the world",
        "I've always enjoyed preparing dinner for my family",
    ],
})

df = df.with_column(
    "label",
    classify_text(
        daft.col("text"),
        labels=['travel', 'cooking', 'dancing'],
        provider="transformers",
        model="facebook/bart-large-mnli",
        multi_label=True,
    ),
)

df.show()
