# /// script
# description = "Async class UDF example with aiohttp"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.5", "aiohttp"]
# ///


import daft
import aiohttp

@daft.cls
class APIClient:
    def __init__(self):
        pass

    async def fetch(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()


if __name__ == "__main__":

    client = APIClient()
    df = daft.from_pydict({"endpoint": [
        "https://httpbin.org/get",
        "https://httpbin.org/ip",
    ]})
    df = df.select(client.fetch(df["endpoint"]).alias("response"))
    df.show()