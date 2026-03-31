# /// script
# description = "Template for new Daft examples"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.6"]
# ///

import daft

if __name__ == "__main__":
    df = daft.from_pydict({"foo": [1, 2, 3]})
    df.show()
