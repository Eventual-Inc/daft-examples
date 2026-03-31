# /// script
# description = "Run vision models on Open Images dataset"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws,openai]>=0.7.6", "python-dotenv"]
# ///

import os

from dotenv import load_dotenv

import daft
from daft import col
from daft.functions import file, prompt

if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set - this example requires OpenAI API access")
        exit(1)

    daft.set_execution_config(enable_dynamic_batching=True)
    daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

    # Load images
    df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg").limit(5)
    df = df.with_column("image", file(col("path")))

    print("\n=== Image Captioning with GPT-4 Vision ===")
    df_captions = df.with_column(
        "caption",
        prompt(
            col("image"),
            system_message="Describe this image in one sentence",
            model="gpt-4o-mini",
        ),
    )
    df_captions.select("path", "caption").show(5)

    print("\n=== Object Detection with GPT-4 Vision ===")
    df_objects = df.with_column(
        "objects",
        prompt(
            col("image"),
            system_message="List all objects you see in this image, comma separated",
            model="gpt-4o-mini",
        ),
    )
    df_objects.select("path", "objects").show(5)

    print("\n=== Image Classification ===")
    df_classify = df.with_column(
        "category",
        prompt(
            col("image"),
            system_message="Classify this image into ONE category: person, animal, vehicle, building, nature, food, or other",
            model="gpt-4o-mini",
        ),
    )
    df_classify.select("path", "category").show(5)
