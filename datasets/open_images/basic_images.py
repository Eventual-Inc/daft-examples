# /// script
# description = "Load and inspect Google Open Images validation dataset"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6"]
# ///

import daft
from daft import col
from daft.functions import decode_image, image_height, image_width

if __name__ == "__main__":
    # Load image paths
    df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")

    print("\n=== Schema ===")
    df.show(1)

    print("\n=== Dataset Statistics ===")
    print(f"Total images: {df.count_rows()}")

    print("\n=== File Size Distribution ===")
    df.select("size").describe()

    print("\n=== Sample Image Paths ===")
    df.select("path").show(5)

    print("\n=== Load and Decode Images ===")
    df_images = df.limit(5).with_column("image", decode_image(col("path").download()))
    df_images.show(5)

    print("\n=== Image Dimensions ===")
    df_dims = (
        df.limit(10)
        .with_column("image", decode_image(col("path").download()))
        .with_column("width", image_width(col("image")))
        .with_column("height", image_height(col("image")))
    )
    df_dims.select("path", "width", "height", "size").show(10)
