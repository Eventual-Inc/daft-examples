# /// script
# description = "Multimodal prompting with images and PDFs using Gemini 2.5 Flash via OpenRouter"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "openai", "numpy", "python-dotenv", "pillow"]
# ///
import json
from dotenv import load_dotenv
import daft
from daft.functions import prompt, download, decode_image, file

load_dotenv()

df = daft.from_pydict({
    "prompt": ["Whats in this image and file?"],
    "my_image": ["/Users/everettkleven/Desktop/Screenshot 2025-11-12 at 9.33.45 AM.png"],
    "my_file": ["/Users/everettkleven/Downloads/ML Inference UDFs_ Online vs Offline.pdf"],
})

# Decode the image and file paths
df = df.with_column(
    "my_image",
    decode_image(download(daft.col("my_image")))
)

df = df.with_column(
    "my_file",
    file(daft.col("my_file"))
)


# Prompt Usage for GPT-5 Responses
df = df.with_column(
    "result",
    prompt(
        messages=[daft.col("prompt"), daft.col("my_image"), daft.col("my_file")],
        system_message="You are a helpful assistant.", 
        model="gpt-5-mini",
        provider="openai",
    )
)

df.select("prompt", "result").show(format="fancy", max_width=60)