# /// script
# description = "Basic usage patterns for daft.File"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.2"]
# ///

import daft
import mimetypes


@daft.func(
    return_dtype=daft.DataType.struct(
        {"name": daft.DataType.string(), "mime_guess": daft.DataType.string()}
    )
)
def read_a_file(file: daft.File):
    return {"name": file.name, "mime_guess": mimetypes.guess_type(file.path)[0]}


df = (
    daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
    .with_column("file", daft.functions.file(daft.col("path")))
    .with_column("metadata", read_a_file(daft.col("file")))
    .select("path", "size", "file", daft.functions.unnest(daft.col("metadata")))
)

df.show(3)
