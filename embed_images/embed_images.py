# /// script
# dependencies = ["daft", "transformers", "torch", "pillow", "torchvision"]
# ///

import daft
from daft import col
from daft.functions import embed_image

if __name__ == "__main__":
    SOURCE_URI = "hf://datasets/HuggingFaceM4/the_cauldron/ai2d/train-00000-of-00001-2ce340398c113b79.parquet"
    MODEL_ID = "google/siglip2-base-patch16-224"
    NUM_IMAGES = 10

    # Read in the parquet file
    df = (
        daft.read_parquet()
        .limit(NUM_IMAGES)
        .explode(col("images"))
        .with_column("image", col("images").struct.get("bytes").image.decode())
        .with_column(
            "image_embeddings",
            embed_image(col("image"), model_name=MODEL_ID, provider="transformers"),
        )
    )

    # Write to turbopuffer
    df.write_parquet(".data/embed_images")

    # Show the dataframe
    df.show()
