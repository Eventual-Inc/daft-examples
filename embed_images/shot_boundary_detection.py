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
        "https://www.youtube.com/watch?v=y5hs7q_LaLM&pp=ygULZGFmdCBlbmdpbmU",
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

    # Select the relevant columns and convert to pandas DataFrame for plotting
    df_plot_multiple = df_shots.select(
        "frame_time", "l2_dist", "l2_dist_cut", "l2_dist_dissolve"
    ).to_pandas()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_plot_multiple["frame_time"],
        df_plot_multiple["l2_dist"],
        label="L2 Distance",
        alpha=0.7,
    )
    plt.plot(
        df_plot_multiple["frame_time"],
        df_plot_multiple["l2_dist_cut"],
        label="L2 Distance (Cut Window)",
        linewidth=2,
    )
    plt.plot(
        df_plot_multiple["frame_time"],
        df_plot_multiple["l2_dist_dissolve"],
        label="L2 Distance (Dissolve Window)",
        linewidth=2,
    )
    plt.xlabel("Frame Time (seconds)")
    plt.ylabel("L2 Distance")
    plt.title("L2 Distance and Smoothed L2 Distances over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    df_shots.where(df_shots["is_cut_boundary"]).select("data").show()
