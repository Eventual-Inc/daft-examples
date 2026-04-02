# /// script
# description = "Simple UDF example to extract file names from File objects"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.6"]
# ///

import daft
from daft.functions import file


@daft.func
def get_name(x: daft.File) -> str:
    return str(x)


if __name__ == "__main__":
    df = daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
    df = df.with_column("file", file(daft.col("path")))
    df = df.with_column("name", get_name(daft.col("file")))
    df.show()
