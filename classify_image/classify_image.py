# /// script
# description = "Classify image"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "transformers","torch","torchvision"]
# ///

import daft
from daft.functions import decode_image
from daft.functions.ai import classify_image

df = (
    # Discover a few images from HuggingFace
    daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")
    # Read the 4 PNG, JPEG, TIFF, WEBP Images
    .with_column("image_bytes", daft.col("path").url.download())
    # Decode the image bytes into a daft Image DataType
    .with_column("image_type", decode_image(daft.col("image_bytes")))
    # Convert Image to RGB and resize the image to 288x288
    .with_column(
        "image_resized", daft.col("image_type").convert_image("RGB").resize(288, 288)
    )
    # Classify the image
    .with_column(
        "image_label",
        classify_image(
            daft.col("image_resized"),
            labels=[],
            provider="transformers",
            model="google/vit-base-patch16-224",
        ),
    )
)

df.show()
