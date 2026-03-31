# /// script
# description = "Read pdf files into a dataframe"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.6"]
# ///

import daft
from daft import col

if __name__ == "__main__":
    # Config
    uri = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
    MAX_DOCS = 10

    df = (
        daft.from_glob_path(uri)  # Discover pdfs
        .with_column("document", col("path").download())  # Download documents
        .limit(MAX_DOCS)  # Limit the number of documents
    )

    # Display the first 8 results
    df.show()
