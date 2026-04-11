# /// script
# description = "Prompt a model"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8","python-dotenv","transformers<5","torch","pillow","torchvision"]
# ///
from dotenv import load_dotenv

import daft
from daft.functions import download, embed_image

if __name__ == "__main__":
    load_dotenv()

    df = (
        daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")
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
