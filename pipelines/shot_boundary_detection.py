# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8", "pandas","numpy", "transformers<5","torchvision","matplotlib","av","yt-dlp"]
# ///
import os
import shutil

import numpy as np

import daft
from daft import DataType as dt
from daft import Window, col
from daft.file import File
from daft.functions import embed_image


@daft.func(return_dtype=dt.float32())
def l2_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    ax = np.asarray(a)
    bx = np.asarray(b)
    return float(np.linalg.norm(ax - bx))


if __name__ == "__main__":
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/videos/*.mp4"
    LOCAL_DIR = ".data/videos"
    MODEL_NAME = "apple/aimv2-large-patch14-224-lit"
    B, T, H, W, C = (
        2,
        16,
        288,
        288,
        3,
    )  # Batch Size, Clip Size (# frames), Height, Width, RGB
    ROW_LIMIT = 50
    CUT_THRESHOLD = 0.1
    DISSOLVE_THRESHOLD = 1.0

    # Download videos locally (read_video_frames uses PyArrow FS which doesn't support hf://)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    paths = daft.from_glob_path(SOURCE_URI).select("path").collect().to_pydict()["path"]
    for p in paths:
        dest = os.path.join(LOCAL_DIR, os.path.basename(p))
        if not os.path.exists(dest):
            with File(p).to_tempfile() as tmp:
                shutil.copy(tmp.name, dest)

    # Read Video Frames from MP4 Files
    df_frames = daft.read_video_frames(
        f"{LOCAL_DIR}/*.mp4",
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

    # Define Windows
    w = Window().partition_by("path").order_by("frame_time")
    w_cut = w.range_between(-0.1, Window.current_row)
    w_dissolve = w.range_between(-1.0, Window.current_row)

    # Run Window Functions
    df_shots = (
        df_emb.with_column(
            "l2_dist",
            l2_distance(col("img_emb_siglip2"), col("img_emb_siglip2").lag(1).over(w)),
        )
        .with_column("l2_dist_cut", col("l2_dist").mean().over(w_cut))
        .with_column("l2_dist_dissolve", col("l2_dist").mean().over(w_dissolve))
        .with_column("is_cut_boundary", col("l2_dist") >= CUT_THRESHOLD)
        .with_column("is_dissolve_boundary", col("l2_dist") >= DISSOLVE_THRESHOLD)
    )
    df_shots.show()
