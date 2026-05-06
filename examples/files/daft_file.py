# /// script
# description = "Basic usage patterns for daft.File"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10"]
# ///
import daft


@daft.func
def read_header(f: daft.File) -> bytes:
    with f.open() as fh:
        return fh.read(16)


if __name__ == "__main__":
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/**")
        .with_column("file", daft.functions.file(daft.col("path")))
        .with_column("header", read_header(daft.col("file")))
        .select("path", "size", "file", "header")
    )

    df.show(5)
