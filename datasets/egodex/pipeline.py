# /// script
# description = "Raw EgoDex HDF5 -> queryable hand-pose features and embeddings."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[transformers,hdf5,video]>=0.7.17",
#     "sentencepiece",
#     "pillow",
# ]
# ///
"""EgoDex pipeline stages for raw HDF5, pose features, embeddings, and queries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import daft
from daft.datatype import DataType
from daft.expressions import col, lit
from daft.functions import (
    file_exists,
    hdf5_file,
    regexp_replace,
    to_struct,
    video_file,
    video_frames,
    when,
)

from .schemas import (
    FEATURE_TRAJECTORY_FIELDS,
    JOINTS,
    LIST_METADATA_FIELDS,
    METADATA_DTYPE,
    METADATA_FIELDS,
    TRAJECTORY_DTYPES,
    TRAJECTORY_FIELDS,
    TRANSFORM_JOINTS,
)

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.file.hdf5 import Hdf5File
    from daft.io import IOConfig


def _attr_value(value: object) -> object:
    """Coerce h5py attribute values (NumPy scalars/arrays, bytes) to plain Python."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "ndim"):
        if value.ndim == 0:
            return _attr_value(value.item())  # ty:ignore[unresolved-attribute]
        return [_attr_value(item) for item in value.tolist()]  # ty:ignore[unresolved-attribute]
    if hasattr(value, "item"):
        return _attr_value(value.item())  # ty:ignore[call-non-callable]
    return value


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, int):
        return [value]
    return list(value)


@dataclass(frozen=True)
class EgoDexPipeline:
    """Pipeline for processing a local EgoDex dataset copy."""

    uri: str
    features_dir: str = "features/"
    embeddings_dir: str = "embeddings/"
    overlay_path: str = "overlay.png"
    io_config: IOConfig | None = None
    sample_interval_seconds: float = 1.0

    def raw(
        self,
        *,
        tasks: str | Sequence[str] | None = None,
        episode_ids: int | Sequence[int] | None = None,
    ) -> DataFrame:
        """Load raw EgoDex episodes with lazy HDF5 and video file references.

        ``tasks`` and ``episode_ids`` are applied from the path-derived columns
        before metadata or trajectory reads, which keeps notebook previews cheap.
        """
        hdf5_glob = f"{self.uri.rstrip('/')}/**/*.hdf5"
        task_values = _as_list(tasks)
        episode_values = _as_list(episode_ids)

        @daft.func(return_dtype=METADATA_DTYPE, use_process=False)
        def read_egodex_metadata(file: Hdf5File) -> dict[str, object]:
            attrs = file.attrs()
            values = {name: _attr_value(attrs.get(name)) for name in METADATA_FIELDS}
            # Daft cannot yet build a struct series when a list-typed field is null on
            # every row of a batch, so missing list attributes become empty lists.
            for name in LIST_METADATA_FIELDS:
                if values[name] is None:
                    values[name] = []
            return values

        episode_paths = daft.from_glob_path(hdf5_glob, io_config=self.io_config).select(
            "path",
            col("path").split("/")[-2].alias("task"),
            col("path").split("/")[-1].split(".")[0].cast(DataType.int64()).alias("episode_id"),
        )
        if task_values:
            episode_paths = episode_paths.where(col("task").is_in(task_values))
        if episode_values:
            episode_paths = episode_paths.where(col("episode_id").is_in([int(value) for value in episode_values]))

        episodes = (
            episode_paths.select(
                "task",
                "episode_id",
                hdf5_file(col("path"), io_config=self.io_config).alias("trajectory"),
                video_file(
                    regexp_replace(col("path"), r"\.hdf5$", ".mp4"),
                    io_config=self.io_config,
                ).alias("video"),
            )
            .with_column(
                "video",
                when(file_exists(col("video")), col("video")).otherwise(lit(None)),
            )
            .with_column("metadata", read_egodex_metadata(col("trajectory")))
        )
        return episodes.select("task", "episode_id", "metadata", "trajectory", "video")

    def trajectory(
        self,
        episodes: DataFrame,
        fields: Sequence[str] = FEATURE_TRAJECTORY_FIELDS,
    ) -> DataFrame:
        """Read selected pose tensors from episode-level EgoDex HDF5 files."""
        import h5py

        fields = tuple(fields)
        if "trajectory" not in episodes.schema().column_names():
            raise ValueError("Expected an episode DataFrame with a `trajectory` column.")
        if len(fields) == 0:
            raise ValueError("fields must contain at least one HDF5 dataset path")

        unknown = [field for field in fields if field not in TRAJECTORY_DTYPES]
        if unknown:
            raise ValueError(f"Unknown trajectory field(s): {unknown}")

        @daft.func(
            return_dtype=DataType.struct({field: TRAJECTORY_DTYPES[field] for field in fields}),
            use_process=False,
            unnest=True,
        )
        def read_egodex_trajectory(file: Hdf5File) -> dict[str, object]:
            with file.to_tempfile() as tmp, h5py.File(tmp.name, "r") as h5:
                return {field: h5[field][()] for field in fields}

        return episodes.where(col("trajectory").not_null()).select(
            "task",
            "episode_id",
            "metadata",
            read_egodex_trajectory(col("trajectory")),
            "video",
        )

    def camera_frames(
        self,
        episodes: DataFrame,
        *,
        start_time: float = 0,
        end_time: float | None = None,
        width: int | None = None,
        height: int | None = None,
        is_key_frame: bool | None = None,
        sample_interval_seconds: float | None = None,
    ) -> DataFrame:
        """Decode EgoDex egocentric videos into a per-episode ``video_frames`` column."""
        if "video" not in episodes.schema().column_names():
            raise ValueError("Expected an episode DataFrame with an EgoDex `video` column.")

        return episodes.with_column(
            "video_frames",
            video_frames(
                col("video"),
                start_time=start_time,
                end_time=end_time,
                width=width,
                height=height,
                is_key_frame=is_key_frame,
                sample_interval_seconds=sample_interval_seconds,
            ),
        )

    def frame_features(self, trajectories: DataFrame, *, fps: float | None = None) -> DataFrame:
        """Explode trajectory tensors into one row of spatial pose geometry per frame.

        A single episode-level UDF turns the whole-episode transform tensors into a
        ``frames`` list of per-frame structs (see ``features.SpatialFeatureComputer``),
        which then explodes and unnests into per-frame rows keyed by
        ``(task, episode_id, frame_index)``. No temporal math happens here — the
        action rates are window expressions added by :meth:`temporal_features`.
        """
        column_names = set(trajectories.schema().column_names())
        required_columns = ("task", "episode_id")
        missing_columns = [name for name in required_columns if name not in column_names]
        missing_fields = [field for field in FEATURE_TRAJECTORY_FIELDS if field not in column_names]
        if missing_columns or missing_fields:
            problems = []
            if missing_columns:
                problems.append(f"missing columns: {missing_columns}")
            if missing_fields:
                problems.append(f"missing trajectory fields: {missing_fields}")
            raise ValueError("Expected a trajectory DataFrame from `trajectory(...)`; " + "; ".join(problems))

        from .features import FPS, FRAME_FEATURES_DTYPE, SpatialFeatureComputer

        @daft.func(return_dtype=FRAME_FEATURES_DTYPE, use_process=False)
        def spatial_frame_features(transforms: dict[str, object]) -> list[dict[str, object]]:
            return SpatialFeatureComputer().compute(transforms)

        transform_struct = to_struct(**{field: col(field) for field in FEATURE_TRAJECTORY_FIELDS})
        frame_rate = FPS if fps is None else fps
        return (
            trajectories.select("task", "episode_id", spatial_frame_features(transform_struct).alias("frames"))
            .explode("frames")
            .select("task", "episode_id", col("frames").unnest())
            .with_column("timestamp", col("frame_index").cast(DataType.float64()) / frame_rate)
        )

    def temporal_features(self, frames: DataFrame, *, fps: float | None = None) -> DataFrame:
        """Add in-DAG action rates to per-frame rows and select the query schema.

        Every rate is a window expression over
        ``Window().partition_by("task", "episode_id").order_by("frame_index")`` —
        see :mod:`temporal`. Returns the per-frame columns the scenario queries
        consume (``query.TRACKS`` per hand) plus the row keys.
        """
        from .features import FPS, HANDS
        from .query import TRACKS
        from .temporal import add_temporal_features

        column_names = set(frames.schema().column_names())
        required_columns = ("task", "episode_id", "frame_index", "curl_L")
        if any(name not in column_names for name in required_columns):
            raise ValueError("Expected a per-frame DataFrame from `frame_features(...)`.")

        rows = add_temporal_features(frames, fps=FPS if fps is None else fps)
        keep = ["task", "episode_id", "frame_index"]
        if "timestamp" in column_names:
            keep.append("timestamp")
        keep += [f"{name}_{tag}" for tag, _ in HANDS for name in TRACKS]
        return rows.select(*keep)

    def calculate_features(self, trajectories: DataFrame, *, fps: float | None = None) -> DataFrame:
        """Per-frame queryable pose features: spatial geometry + windowed action rates."""
        return self.temporal_features(self.frame_features(trajectories, fps=fps), fps=fps)

    def embed_frames(self, frames: DataFrame, *, keep_images: bool = False) -> DataFrame:
        """Embed decoded frame rows into the SigLIP image/text space."""
        from .embeddings import embed_image_normalized

        if "video_frames" not in frames.schema().column_names():
            raise ValueError("Expected a frame DataFrame from `camera_frames(...)` with `video_frames`.")

        rows = frames.select("task", "episode_id", "video_frames").explode("video_frames")
        rows = rows.select(
            "task",
            "episode_id",
            col("video_frames")["frame_index"].alias("frame_index"),
            col("video_frames")["frame_time"].alias("timestamp"),
            col("video_frames")["data"].alias("image"),
        )
        # episodes without a sibling video explode to a null image row; the SigLIP
        # processor cannot take None, so drop them before the embedder sees a batch
        rows = rows.where(col("image").not_null())
        rows = rows.with_column("clip_emb", embed_image_normalized(col("image")))
        return rows if keep_images else rows.exclude("image")

    def run_all(self) -> tuple[DataFrame, DataFrame]:
        """Build and write feature and embedding tables, returning both lazy plans."""
        episodes = self.raw()
        trajectories = self.trajectory(episodes)
        features = self.calculate_features(trajectories)
        frames = self.camera_frames(episodes, sample_interval_seconds=self.sample_interval_seconds)
        embeddings = self.embed_frames(frames)

        features.write_parquet(self.features_dir)
        embeddings.write_parquet(self.embeddings_dir)
        return features, embeddings

    def query_features(
        self, features: DataFrame, embeddings: DataFrame
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        """Run the example pose, text, and combined queries."""
        from .query import calibrate, query

        thresholds = calibrate(features)
        pose_hits = query(features, pose="writing_grip", k=5, thresholds=thresholds)
        text_hits = query(features, text="chopsticks", clip=embeddings, k=5)
        combined_hits = query(
            features,
            pose="hammer_grip",
            text="stapler",
            clip=embeddings,
            k=5,
            thresholds=thresholds,
        )
        return pose_hits, text_hits, combined_hits

    def visualize_hit(self, hits: list[dict[str, object]]) -> None:
        """Save an overlay for the first hit, when one exists."""
        if not hits:
            return
        from .viz import overlay

        top = hits[0]
        image = overlay(
            self.uri,
            str(top["task"]),
            int(top["episode_id"]),
            top["segments"][0][0],  # ty:ignore[not-subscriptable]
        )
        image.save(self.overlay_path)
        print(f"\nSaved skeleton overlay of {top['task']}/{top['episode_id']} to {self.overlay_path}")


def main(args: argparse.Namespace) -> None:
    pipeline = EgoDexPipeline(
        uri=args.data,
        features_dir=args.features_dir,
        embeddings_dir=args.embeddings_dir,
        overlay_path=args.overlay_path,
    )
    features, embeddings = pipeline.run_all()
    pose_hits, text_hits, combined_hits = pipeline.query_features(features, embeddings)

    for title, hits in [
        ("writing_grip", pose_hits),
        ("text 'chopsticks'", text_hits),
        ("hammer_grip + 'stapler'", combined_hits),
    ]:
        print(f"\n## {title}")
        for hit in hits:
            print(
                f"  {hit['task']}/{hit['episode_id']}: score {hit['score']:.3f}, "
                f"{hit['n_frames']} frames, segments {hit['segments'][:3]}"  # ty:ignore[not-subscriptable]
            )

    pipeline.visualize_hit(pose_hits)


__all__ = [
    "EgoDexPipeline",
    "FEATURE_TRAJECTORY_FIELDS",
    "JOINTS",
    "TRAJECTORY_FIELDS",
    "TRANSFORM_JOINTS",
]


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build queryable hand-pose features from a local EgoDex download.")
    parser.add_argument(
        "--data",
        default=os.environ.get("EGODEX_DATA", ".data"),
        help="Path to the extracted EgoDex root. Defaults to EGODEX_DATA or .data.",
    )
    parser.add_argument(
        "--features-dir",
        default="features/",
        help="Output parquet directory for pose-feature tracks.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="embeddings/",
        help="Output parquet directory for SigLIP frame embeddings.",
    )
    parser.add_argument(
        "--overlay-path",
        default="overlay.png",
        help="Output path for the best pose hit skeleton overlay.",
    )
    main(parser.parse_args())
