# /// script
# description = "Embed Video Frames from a Youtube Video"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pandas","numpy", "transformers","torchvision","matplotlib","av","yt-dlp"]
# ///
import daft
from daft.functions import embed_image



if __name__ == "__main__":
    VIDEOS = [  # Data Systems for Multimodal Madness
        "https://www.youtube.com/watch?v=y5hs7q_LaLM&pp=ygULZGFmdCBlbmdpbmU",
    ]
    DEST_URI = ".data/embed_video_frames"
    MODEL_NAME = "apple/aimv2-large-patch14-224-lit"
    B, T, H, W, C = (
        2,
        16,
        288,
        288,
        3,
    )  # Batch Size, Clip Size (# frames), Height, Width, RGB
    ROW_LIMIT = 50

    # Read Video Frames from MP4 Files
    df_frames = daft.read_video_frames(
        VIDEOS,
        image_height=H,
        image_width=W,
    ).limit(ROW_LIMIT)

    # Embed Image Frames
    df_emb = df_frames.with_column(
        "img_emb_siglip2",
        embed_image(
            df_frames["data"],
            model_name=MODEL_NAME,
            provider="transformers",
        ),
    )

    df = df_emb.collect()
    df.show()
