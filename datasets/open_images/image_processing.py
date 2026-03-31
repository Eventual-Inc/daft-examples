# /// script
# description = "Image preprocessing: resize, crop, and transform"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col
from daft.functions import decode_image, resize, image_width, image_height

# Load images
df = (
    daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")
    .limit(5)
    .with_column("image", decode_image(col("path").download()))
)

print("\n=== Original Image Dimensions ===")
df_meta = (
    df.with_column("width", image_width(col("image")))
    .with_column("height", image_height(col("image")))
)
df_meta.select("path", "width", "height").show()

print("\n=== Resize to 224x224 (Standard CNN Input) ===")
df_resized = (
    df.with_column("resized", resize(col("image"), w=224, h=224))
    .with_column("new_width", image_width(col("resized")))
    .with_column("new_height", image_height(col("resized")))
)
df_resized.select("path", "new_width", "new_height").show()

print("\n=== Resize to 512x512 (High Resolution) ===")
df_high_res = (
    df.with_column("high_res", resize(col("image"), w=512, h=512))
    .with_column("width", image_width(col("high_res")))
)
df_high_res.select("path", "width").show()
