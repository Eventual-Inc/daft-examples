# /// script
# description = "Simple UDF example to extract file names from File objects"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.5", "aiohttp"]
# ///

import daft
import asyncio

@daft.func
async def fetch_status(url: str) -> int:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response.status


if __name__ == "__main__":

    df = daft.from_pydict({"endpoint": [
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/500",
    ]})
    df = df.select(
        df["endpoint"],
        fetch_status(df["endpoint"]).alias("status_code"),
    )
    df.show()