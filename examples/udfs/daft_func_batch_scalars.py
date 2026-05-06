# /// script
# description = "Simple UDF example to clip values"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10", "pyarrow"]
# ///

import daft
from daft import DataType, Series


@daft.func.batch(return_dtype=DataType.float64())
def clip_values(values: Series, lower: float, upper: float) -> Series:
    import pyarrow.compute as pc

    arr = values.to_arrow()
    clipped = pc.max_element_wise(arr, lower)
    clipped = pc.min_element_wise(clipped, upper)
    return Series.from_arrow(clipped)


if __name__ == "__main__":
    df = daft.from_pydict({"signal": [0.1, 5.5, -3.2, 12.0, 0.8]})
    df = df.select(clip_values(df["signal"], 0.0, 10.0).alias("clipped"))
    df.show()
