# /// script
# description = "Process a mixed-content knowledge base in one bucket with daft.File"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[huggingface]>=0.7.6"]
# ///
"""
One glob, every file type. Treat a mixed-content bucket as a single DataFrame.

Pairs naturally with the lakehouse pipeline's S3 backend: store structured
Iceberg tables and unstructured files (docs, code, media) in the same S3
bucket. With AWS S3 Files, mount it locally for POSIX writes while Daft
reads everything via s3:// or the Glue catalog.

The glob here points at a public HuggingFace dataset, but the same code
runs unchanged against s3:// or /mnt/s3files/... paths.
"""

from pathlib import PurePosixPath

import daft
from daft import DataType, col
from daft.functions import file as daft_file


@daft.func(return_dtype=DataType.string())
def extension(path: str) -> str:
    return PurePosixPath(path).suffix.lstrip(".").lower()


@daft.func(return_dtype=DataType.string())
def kind_of(ext: str) -> str:
    """Bucket file extensions into broad content categories."""
    if ext in {"py", "js", "ts", "rs", "go", "java"}:
        return "code"
    if ext in {"md", "txt", "rst"}:
        return "doc"
    if ext in {"pdf"}:
        return "pdf"
    if ext in {"jpg", "jpeg", "png", "gif", "webp"}:
        return "image"
    if ext in {"mp3", "wav", "flac", "ogg"}:
        return "audio"
    if ext in {"mp4", "mov", "avi", "mkv"}:
        return "video"
    if ext in {"parquet", "csv", "json", "jsonl"}:
        return "data"
    return "other"


@daft.func(return_dtype=DataType.string(), on_error="log")
def preview(f: daft.File, n_bytes: int = 200) -> str:
    """Decode the first n_bytes of a file as UTF-8 text."""
    with f.open() as fh:
        return fh.read(n_bytes).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # One glob, every file type. Swap for s3://your-bucket/** in production.
    KNOWLEDGE_BASE_URI = "hf://datasets/Eventual-Inc/sample-files/**"

    df = (
        daft.from_glob_path(KNOWLEDGE_BASE_URI)
        .with_column("file", daft_file(col("path")))
        .with_column("ext", extension(col("path")))
        .with_column("kind", kind_of(col("ext")))
    )

    print("\n=== Knowledge base catalog ===")
    df.select("kind", "ext", "size", "path").show(15)

    print("\n=== Files per content kind ===")
    df.groupby("kind").agg(col("size").count().alias("n_files"), col("size").sum().alias("total_bytes")).show()

    print("\n=== Text preview of docs and code ===")
    docs_and_code = df.where(col("kind").is_in(["doc", "code"])).with_column("preview", preview(col("file")))
    docs_and_code.select("path", "kind", "preview").show(5)
