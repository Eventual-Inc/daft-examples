# /// script
# description = "Prompt a model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai,transformers]","python-dotenv"]
# ///
from dotenv import load_dotenv
import daft
from daft.functions import embed_text

load_dotenv()

import daft
from daft.functions import embed_image, download

df = (
    daft.from_glob_path("/Users/everettkleven/Downloads/*.png")
    .with_column("image", download(daft.col("path")).decode_image())
    .with_column(
        "image_embeddings",
        embed_image(
            daft.col("image"),
            provider="transformers",
            model="apple/aimv2-large-patch14-224-lit",
        ),
    )
)
df.show(3)
