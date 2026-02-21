# /// script
# description = "Class-based UDFs with TypedDict, Pydantic, batch processing, and async functions"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pydantic"]
# ///

import daft
import typing
import pydantic


class DoSomethingResultPydantic(pydantic.BaseModel):
    x3: int
    y3: int
    some_arg: str


class DoSomethingResultTypedDict(typing.TypedDict):
    x3: int
    y3: int
    some_arg: str


DoSomethingDaftDataType = daft.DataType.struct(
    {
        "x3": daft.DataType.int64(),
        "y3": daft.DataType.int64(),
        "some_arg": daft.DataType.string(),
    }
)


@daft.cls()
class NewUDFUsageExample:
    def __init__(self, x1: int, y1: int):
        self.x1 = x1
        self.y1 = y1

    # With Python Type Hint Resolution
    @daft.method()
    def do_something_typeddict(
        self, x2: int, y2: int, some_arg: str
    ) -> DoSomethingResultTypedDict:
        x3 = self.x1 + x2
        y3 = self.y1 - y2
        return {"x3": x3, "y3": y3, "some_arg": some_arg}

    @daft.method()
    def do_something_pydantic(
        self, x2: int, y2: int, some_arg: str
    ) -> DoSomethingResultPydantic:
        x3 = self.x1 + x2
        y3 = self.y1 - y2
        return DoSomethingResultPydantic(x3=x3, y3=y3, some_arg=some_arg)

    # With Return Dtype and no lint errors
    @daft.method(return_dtype=DoSomethingDaftDataType)
    def do_something_daft(self, x2: int, y2: int, some_arg: str):
        x3 = self.x1 * x2
        y3 = self.y1 // y2
        return {"x3": x3, "y3": y3, "some_arg": some_arg}


if __name__ == "__main__":
    # Instantiate the UDF
    my_udf = NewUDFUsageExample(x1=1, y1=2)

    # Create a dataframe
    df = daft.from_pydict({"x1": [1, 2, 3], "y1": [4, 5, 6]})

    # Use the UDF
    df = df.with_column(
        "something_typeddict",
        my_udf.do_something_typeddict(
            daft.col("x1"), daft.col("y1"), daft.lit("some_arg")
        ),
    )
    df = df.with_column(
        "something_pydantic",
        my_udf.do_something_pydantic(
            daft.col("x1"), daft.col("y1"), daft.lit("some_arg")
        ),
    )
    df = df.with_column(
        "something_daft",
        my_udf.do_something_daft(daft.col("x1"), daft.col("y1"), daft.lit("some_arg")),
    )
    df.show()
