# /// script
# description = "Classify text"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10", "transformers", "torch"]
# ///
import daft
from daft.functions import classify_text

if __name__ == "__main__":
    df = daft.from_pydict(
        {
            "text": [
                "One day I will see the world",
                "I've always enjoyed preparing dinner for my family",
            ],
        }
    )

    df = df.with_column(
        "label",
        classify_text(
            daft.col("text"),
            labels=["travel", "cooking", "dancing"],
            provider="transformers",
            model="facebook/bart-large-mnli",
            multi_label=True,
        ),
    )

    df.show()
