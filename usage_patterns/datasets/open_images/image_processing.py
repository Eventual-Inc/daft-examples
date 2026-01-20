# /// script
# description = "Image preprocessing: resize, crop, and transform"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col
from daft.functions import file, image_resize, image_metadata

# Load images
df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg").limit(5)
df = df.with_column("image", file(col("path")))

print("\n=== Original Image Dimensions ===")
df_meta = df.with_column("metadata", image_metadata(col("image")))
df_meta = df_meta.with_column("width", col("metadata")["width"])
df_meta = df_meta.with_column("height", col("metadata")["height"])
df_meta.select("path", "width", "height").show()

print("\n=== Resize to 224x224 (Standard CNN Input) ===")
df_resized = df.with_column("resized", image_resize(col("image"), width=224, height=224))
df_resized = df_resized.with_column("metadata", image_metadata(col("resized")))
df_resized = df_resized.with_column("new_width", col("metadata")["width"])
df_resized = df_resized.with_column("new_height", col("metadata")["height"])
df_resized.select("path", "new_width", "new_height").show()

print("\n=== Resize to 512x512 (High Resolution) ===")
df_high_res = df.with_column("high_res", image_resize(col("image"), width=512, height=512))
df_high_res = df_high_res.with_column("metadata", image_metadata(col("high_res")))
df_high_res = df_high_res.with_column("width", col("metadata")["width"])
df_high_res.select("path", "width").show()
