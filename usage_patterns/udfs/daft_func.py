# /// script
# description = "Simple UDF example to extract file names from File objects"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft"]
# ///

import daft
from daft.functions import file

df = daft.from_glob_path("/Users/everettkleven/Downloads/*.pdf")
df = df.with_column("file", file(daft.col("path")))


@daft.func
def get_name(x: daft.File) -> str:
    return str(x)


df = df.with_column("name", get_name(daft.col("file")))
df.show()
