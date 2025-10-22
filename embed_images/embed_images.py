# /// script
# description = "Embed images from a parquet file"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "transformers", "torch", "pillow", "torchvision"]
# ///

import daft
from daft.functions import embed_image, decode_image

# Embed Text with Defaults
df = (
    # Discover a few images from HuggingFace
    daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")

    # Read the 4 PNG, JPEG, TIFF, WEBP Images
    .with_column("image_bytes", daft.col("path").url.download())

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
            model="apple/aimv2-large-patch14-224-lit"
        )
    )
)

# Show the dataframe
df.show()