# /// script
# description = "Multimodal prompting with images and PDFs"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "python-dotenv", "pillow"]
# ///
from dotenv import load_dotenv

import daft
from daft.functions import decode_image, download, file, prompt

if __name__ == "__main__":
    load_dotenv()

    # Build a row with an image and a PDF from public HuggingFace datasets
    df = daft.from_pydict(
        {
            "prompt": ["Describe the image and summarize the PDF."],
            "image_url": ["hf://datasets/datasets-examples/doc-image-3/images/2.png"],
            "pdf_url": ["hf://datasets/Eventual-Inc/sample-files/papers/2102.04074v1.pdf"],
        }
    )

    # Decode the image and wrap the PDF as a daft.File
    df = df.with_column("my_image", decode_image(download(daft.col("image_url")))).with_column(
        "my_file", file(daft.col("pdf_url"))
    )

    # Prompt Usage for GPT-5 Responses
    df = df.with_column(
        "result",
        prompt(
            messages=[daft.col("prompt"), daft.col("my_image"), daft.col("my_file")],
            system_message="You are a helpful assistant.",
            model="gpt-5.1",
            provider="openai",
        ),
    )

    df.select("prompt", "result").show(format="fancy", max_width=120)
