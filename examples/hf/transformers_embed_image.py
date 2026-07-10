# /// script
# description = "Generate local image embeddings from a Hugging Face dataset"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[transformers]==0.7.19"]
# ///

import daft
from daft.functions import decode_image, download, embed_image

images = daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images/*").limit(2)

(
    images.with_column("image_bytes", download(daft.col("path")))
    .with_column("image", decode_image(daft.col("image_bytes")))
    .with_column("image_rgb", daft.col("image").convert_image("RGB").resize(224, 224))
    .with_column(
        "embedding",
        embed_image(
            daft.col("image_rgb"),
            provider="transformers",
            model="openai/clip-vit-base-patch32",
        ),
    )
    .select("path", "embedding")
    .show()
)
