# /// script
# description = "Embed images from a parquet file"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8", "transformers<5", "torch", "pillow", "torchvision"]
# ///

import daft
from daft.functions import decode_image, download, embed_image

if __name__ == "__main__":
    # Embed Text with Defaults
    df = (
        # Discover a few images from HuggingFace
        daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")
        # Read the 4 PNG, JPEG, TIFF, WEBP Images
        .with_column("image_bytes", download(daft.col("path")))
        # Decode the image bytes into a daft Image DataType
        .with_column("image_type", decode_image(daft.col("image_bytes")))
        # Convert Image to RGB and resize the image to 288x288
        .with_column("image_resized", daft.col("image_type").convert_image("RGB").resize(288, 288))
        # Embed the image
        .with_column(
            "image_embeddings",
            embed_image(
                daft.col("image_resized"),
                provider="transformers",
                model="apple/aimv2-large-patch14-224-lit",
            ),
        )
    )

    # Show the dataframe
    df.show()
