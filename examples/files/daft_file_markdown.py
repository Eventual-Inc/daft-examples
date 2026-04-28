# /// script
# description = "Extract markdown headings from files using daft.File"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8"]
# ///

from collections.abc import Iterator
from typing import TypedDict

import daft
from daft import col
from daft.functions import unnest


class Heading(TypedDict):
    level: int
    text: str


@daft.func
def extract_headings(file: daft.File) -> Iterator[Heading]:
    with file.open() as f:
        content = f.read().decode("utf-8")
    for line in content.splitlines():
        if line.startswith("#"):
            yield Heading(
                level=len(line) - len(line.lstrip("#")),
                text=line.lstrip("# ").strip(),
            )


if __name__ == "__main__":
    df = (
        daft.from_files("**/*.md")
        .with_column("heading", extract_headings(col("file")))
        .select(col("file"), unnest(col("heading")))
    )

    df.show(10)
