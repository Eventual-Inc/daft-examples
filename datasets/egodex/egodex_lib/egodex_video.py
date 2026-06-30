"""Remux EgoDex per-episode mp4s into a LeRobot v3 dataset's video shards.

The dataset's tabular data + meta are already written (use_videos=False). This
adds the `observation.image` video feature WITHOUT re-encoding: it stream-copies
the source mp4s (concatenating episodes into ~200MB shards, the LeRobot default)
and patches info.json + meta/episodes with the feature and per-episode locators.

`episode_mp4s` MUST be in episode_index order (i.e. the same file order fed to
the writer), so locator[i] describes episode i.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
from typing import Any

CHUNK_SIZE = 1000  # files per chunk (LeRobot DEFAULT_CHUNK_SIZE)
VIDEO_SIZE_CAP_MB = 200  # LeRobot DEFAULT_VIDEO_FILE_SIZE_IN_MB


def _probe(path: str) -> dict[str, Any]:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        dur = float(s.duration * s.time_base) if s.duration else (s.frames / float(s.average_rate))
        return {
            "codec": cc.name,
            "pix_fmt": cc.pix_fmt,
            "width": cc.width,
            "height": cc.height,
            "fps": round(float(s.average_rate)),
            "frames": s.frames,
            "duration": dur,
            "has_audio": len(c.streams.audio) > 0,
            "size": os.path.getsize(path),
        }


# Per-codec ffmpeg output args. "copy" stream-copies the source (fast, lossless,
# but keeps the source codec); "h264"/"av1" RE-ENCODE to a browser-playable codec.
# EgoDex source is mpeg4 (MPEG-4 Part 2), which browsers/<video> cannot play, so
# the dataset viewer shows blank video unless we transcode to h264 or av1.
_VENC = {
    "copy": ["-c", "copy"],
    "h264": ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"],
    "av1": ["-c:v", "libsvtav1", "-crf", "30",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"],
}


def _concat(ffmpeg: str, mp4s: list[str], out_path: pathlib.Path, codec: str = "h264") -> None:
    """Concat mp4s into one shard. codec='copy' stream-copies; 'h264'/'av1' re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lf:
        for m in mp4s:
            lf.write(f"file '{os.path.abspath(m)}'\n")
        listfile = lf.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             *_VENC[codec], "-loglevel", "error", str(out_path)],
            check=True,
        )
    finally:
        os.unlink(listfile)


def _pack(probes: list[dict], cap_bytes: int, chunk_size: int):
    """Greedily pack episodes into size-capped shards.

    Returns (locators, shards): locators[i] = (chunk, file, from_ts, to_ts);
    shards = list of (chunk, file, [episode indices]).
    """
    locators: list[tuple[int, int, float, float]] = [None] * len(probes)  # type: ignore
    shards: list[tuple[int, int, list[int]]] = []
    chunk_idx = file_idx = 0
    cur: list[int] = []
    cur_size = 0
    cur_offset = 0.0

    def close():
        nonlocal cur, cur_size, cur_offset, file_idx, chunk_idx
        if not cur:
            return
        shards.append((chunk_idx, file_idx, list(cur)))
        cur, cur_size, cur_offset = [], 0, 0.0
        file_idx += 1
        if file_idx >= chunk_size:
            file_idx = 0
            chunk_idx += 1

    for ei, pr in enumerate(probes):
        if cur and cur_size + pr["size"] > cap_bytes:
            close()
        from_ts = cur_offset
        to_ts = cur_offset + pr["duration"]
        locators[ei] = (chunk_idx, file_idx, from_ts, to_ts)
        cur.append(ei)
        cur_size += pr["size"]
        cur_offset = to_ts
    close()
    return locators, shards


def _patch_info(root: pathlib.Path, video_key: str, feat: dict, codec: str) -> None:
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [feat["height"], feat["width"], 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": feat["height"],
            "video.width": feat["width"],
            "video.codec": codec,
            "video.pix_fmt": feat["pix_fmt"],
            "video.is_depth_map": False,
            "video.fps": feat["fps"],
            "video.channels": 3,
            "has_audio": feat["has_audio"],
        },
    }
    p.write_text(json.dumps(info, indent=4))


def _patch_episodes(root: pathlib.Path, video_key: str, locators: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for fp in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(fp)
        eidx = t.column("episode_index").to_pylist()
        cols = {
            f"videos/{video_key}/chunk_index": pa.array([locators[e][0] for e in eidx], pa.int64()),
            f"videos/{video_key}/file_index": pa.array([locators[e][1] for e in eidx], pa.int64()),
            f"videos/{video_key}/from_timestamp": pa.array([locators[e][2] for e in eidx], pa.float32()),
            f"videos/{video_key}/to_timestamp": pa.array([locators[e][3] for e in eidx], pa.float32()),
        }
        # Idempotent: drop any pre-existing locator columns so re-running doesn't
        # append duplicates (e.g. when re-encoding an already-patched dataset).
        t = t.select([c for c in t.column_names if c not in cols])
        for name, arr in cols.items():
            t = t.append_column(name, arr)
        pq.write_table(t, fp)


def remux_concat_videos(
    dataset_dir: str,
    episode_mp4s: list[str],
    video_key: str = "observation.image",
    size_cap_mb: int = VIDEO_SIZE_CAP_MB,
    chunk_size: int = CHUNK_SIZE,
    codec: str = "h264",
) -> dict:
    """Add `video_key` to an existing tabular LeRobot dataset from episode mp4s.

    codec="h264" (default) RE-ENCODES to a browser-playable codec so the dataset
    viewer renders video; "av1" likewise (smaller, slower); "copy" stream-copies
    the source (fast/lossless but keeps EgoDex's mpeg4, which browsers can't play).
    Shard packing is by *source* size, so transcoded shards may come out under the
    cap. Safe to re-run (meta patch is idempotent). Returns a summary dict.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    root = pathlib.Path(dataset_dir)
    probes = [_probe(m) for m in episode_mp4s]
    locators, shards = _pack(probes, size_cap_mb * 1024 * 1024, chunk_size)

    for chunk_idx, file_idx, eps in shards:
        out = root / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        _concat(ffmpeg, [episode_mp4s[e] for e in eps], out, codec=codec)

    out_codec = probes[0]["codec"] if codec == "copy" else codec
    _patch_info(root, video_key, probes[0], out_codec)
    _patch_episodes(root, video_key, locators)
    return {"episodes": len(episode_mp4s), "shards": len(shards), "video_key": video_key, "codec": out_codec}
