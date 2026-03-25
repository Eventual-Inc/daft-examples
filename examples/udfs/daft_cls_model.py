# /// script
# description = "Simple UDF example to clip values"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[transformers]>=0.7.4"]
# ///

import daft
from transformers import pipeline

@daft.cls
class SentimentClassifier:
    def __init__(self):
        # Runs ONCE per worker -- download weights, allocate memory
        self.pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def __call__(self, text: str) -> str:
        # Runs on EVERY ROW -- reuses the loaded model
        return self.pipe(text)[0]["label"]

classifier = SentimentClassifier()

df = daft.from_pydict({"review": [
    "This product is amazing",
    "Worst purchase I've ever made",
    "It's okay, nothing special",
]})
df = df.select(classifier(df["review"]).alias("sentiment"))
df.show()