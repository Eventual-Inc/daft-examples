# /// script
# description = "Embed Video Frames from a Youtube Video"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pandas","numpy", "transformers", "torchvision","av","yt-dlp"]
# ///
import daft
from daft.functions import embed_image

VIDEOS = [  
    "https://www.youtube.com/watch?v=y5hs7q_LaLM&pp=ygULZGFmdCBlbmdpbmU",
]
DEST_URI = ".data/embed_video_frames"
ROW_LIMIT = 10

# Read Video Frames from MP4 Files
df_frames = daft.read_video_frames(
    VIDEOS,
    image_height=288,
    image_width=288,
).limit(ROW_LIMIT)

# Embed Image Frames
df_emb = df_frames.with_column(
    "img_embeddings",
    embed_image(
        daft.col("data"),
        model_name="apple/aimv2-large-patch14-224-lit",
        provider="transformers",
    ),
).collect()

df_emb.show()
