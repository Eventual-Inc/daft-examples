# /// script
# description = "Embed images from a parquet file"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.14", "openai", "numpy", "python-dotenv", "torch","torchvision", "pillow", "transformers"]
# ///
import os
import daft
from daft.functions import embed_image

SUBSET = os.environ.get("SUBSET", "ai2d")
LIMIT = os.environ.get("LIMIT", 100)
DEST_URI = os.environ.get("DEST_URI", None)

def run_image_embedding(df: daft.DataFrame): 
    df = (
        df
        .with_column("images", daft.col("images")["bytes"].decode_image())
        .with_column(
            "image_embeddings",
            embed_image(
                daft.col("images"),
                provider="transformers",
                model="openai/clip-vit-base-patch32",
            ),
        )
        .limit(LIMIT)
    )
    return df

def main():
    df = daft.read_parquet(f"hf://datasets/HuggingFaceM4/the_cauldron/{SUBSET}/*.parquet")
    df = df.explode("images").with_column("image", daft.col("images")["bytes"].decode_image())
    
    df = run_image_embedding(df)
    
    if DEST_URI is not None:
        df.write_parquet(DEST_URI)
    else:
        df.show()
    

if __name__ == "__main__":
    main()
