# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pandas","numpy", "transformers","torchvision","matplotlib","av","yt-dlp"]
# ///
import daft
from daft.functions import embed_image
from daft import col, Window, DataType as dt

import numpy as np
import matplotlib.pyplot as plt


@daft.func(return_dtype=dt.float32())
def l2_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    ax = np.asarray(a)
    bx = np.asarray(b)
    return float(np.linalg.norm(ax - bx))


if __name__ == "__main__":
    VIDEOS = [  # Data Systems for Multimodal Madness
        "/Users/everettkleven/Desktop/data/videos/Build_Scalable_Batch_Inference_Pipelines_in_3_Lines_Daft_GPT_vLLM.mp4",
        "/Users/everettkleven/Desktop/data/videos/Daft_Team_Takes_On_Climbing_Dogpatch_Boulders_SF.mp4",
        "/Users/everettkleven/Desktop/data/videos/Dynamic_Execution_for_Multimodal_Data_Processing_Daft_Launch_Week_Day_2.mp4",
        "/Users/everettkleven/Desktop/data/videos/Near_100_GPU_Utilization_Embedding_Millions_of_Text_Documents_With_Qwen3.mp4",
    ]
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
