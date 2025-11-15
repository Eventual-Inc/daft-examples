# /// script
# dependencies = ["daft"]
# ///

import daft

if __name__ == "__main__":
    df = daft.from_pydict({"foo": [1, 2, 3]})
    df.show()
