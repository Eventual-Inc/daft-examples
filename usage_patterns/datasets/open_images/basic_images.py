# /// script
# description = "Load and inspect Google Open Images validation dataset"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col
from daft.functions import file, image_metadata

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
df_images = df.limit(5).with_column("image", file(col("path")))
df_images.show(5)

print("\n=== Image Metadata ===")
df_meta = df.limit(10).with_column("image", file(col("path")))
df_meta = df_meta.with_column("metadata", image_metadata(col("image")))
df_meta = df_meta.with_column("width", col("metadata")["width"])
df_meta = df_meta.with_column("height", col("metadata")["height"])
df_meta.select("path", "width", "height", "size").show(10)
