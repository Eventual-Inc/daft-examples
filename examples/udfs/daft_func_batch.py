# /// script
# description = "Simple UDF example to extract file names from File objects"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10", "pyarrow"]
# ///

import daft
from daft import DataType, Series


@daft.func.batch(return_dtype=DataType.float64())
def z_score(values: Series) -> Series:
    import pyarrow.compute as pc

    arr = values.to_arrow()
    mean = pc.mean(arr).as_py()
    stddev = pc.stddev(arr).as_py()
    if stddev == 0:
        return Series.from_arrow(pc.subtract(arr, mean))
    centered = pc.subtract(arr, mean)
    return Series.from_arrow(pc.divide(centered, stddev))


if __name__ == "__main__":
    df = daft.from_pydict({"measurement": [10.0, 20.0, 30.0, 40.0, 50.0]})
    df = df.select(z_score(df["measurement"]).alias("z"))
    df.show()
