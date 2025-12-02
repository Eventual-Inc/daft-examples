"""
Image Understanding Evaluation Pipeline

Evaluates vision-language model (VLM) image understanding using:
1. Structured outputs with Pydantic models
2. Ablation study (with/without images)
3. Quadrant classification
4. VLM-as-a-Judge diagnostic feedback

Usage:
    # Set your API key and run
    export DAFT_API_KEY=your_key
    python eval_image_understanding.py

    # Override subset or limit via CLI
    python eval_image_understanding.py --subset chartqa --limit 100
"""

import os
import json
from pydantic import BaseModel, Field


import daft
from daft import col
from daft.functions import (
    prompt,
    unnest,
    when,
    format,
)

class ChoiceResponse(BaseModel):
    """Structured output for multiple choice answers."""

    choice: str = Field(
        ..., description="The letter of the correct choice (e.g., A, B, C, D)"
    )

class JudgeResponse(BaseModel):
    """Structured diagnostic feedback from the VLM judge."""

    reasoning: str = Field(
        ..., description="Why did the model choose the answer it did?"
    )
    hypothesis: str = Field(
        ..., description="What caused the divergence from the correct answer?"
    )
    attribution: str = Field(
        ...,
        description="Was this a 'question' issue or an 'image' understanding issue?",
    )

# System Prompts
SYSTEM_MESSAGE = (
    "Referencing the attached image, respond to the multiple choice question with just the letter corresponding to the correct answer. Do not include any other text."
)

def preprocess(df: daft.DataFrame, category: str, subset: str, model_id: str, system_message: str, params: dict) -> daft.DataFrame:
    """Preprocess images and text from The Cauldron format."""

    # Track Evaluation Inputs (Prompt Arguments) 
    df = df.with_columns(
        {
            "category": daft.lit(category), 
            "subset": daft.lit(subset),
            "model_id": daft.lit(model_id),
            "system_message": daft.lit(system_message),
            "params": daft.lit(json.dumps(params, indent=4)),
        }
    )
 
    df = (
        df
        .explode("texts")
        .select("*", unnest(col("texts"))) 
        # Create deterministic prompt_id from content only (allows tracking unique questions across models)
        .with_column(
            "text_hash",
            (
                daft.lit(SYSTEM_MESSAGE)
                + col("user")
                + col("assistant")
                + col("params")
            ).hash(hash_function="xxhash3_64")
        )
        .with_column(
            "image_hash",
            col("image").encode_image("PNG").hash(hash_function="xxhash3_64")
        )
        .with_column(
            "prompt_hash_64",
            (col("text_hash") + col("image_hash")).hash(hash_function="xxhash3_64")
        )
        .exclude("text_hash", "image_hash")
    )

    return df


def run_inference_and_check_correctness(df: daft.DataFrame, model_id: str, with_image: bool = True, system_message: str = SYSTEM_MESSAGE, params: dict = {}) -> daft.DataFrame:
    """Run structured output inference with or without images."""
    if with_image:
        messages = [col("image"), col("user")]
        result_col = "result"
        correct_col = "is_correct"
    else:
        messages = col("user")
        result_col = "result_no_image"
        correct_col = "is_correct_no_image"

    return (
        df
        .with_column(
            result_col,
            prompt(
                messages=messages,
                system_message=system_message,
                model=model_id,
                use_chat_completions=True,
                return_format=ChoiceResponse,
                **params,
            ),
        ).with_column(
            correct_col,
            col(result_col)["choice"].lstrip().rstrip()
            == col("answer").lstrip().rstrip(),
        )
    )

def classify_quadrants(df: daft.DataFrame) -> daft.DataFrame:
    """Classify results into diagnostic quadrants based on ablation results."""
    return df.with_column(
        "quadrant",
        when(
            (col("is_correct") == True) & (col("is_correct_no_image") == True),
            "Both Correct",
        )
        .when(
            (col("is_correct") == True) & (col("is_correct_no_image") == False),
            "Image Helped",
        )
        .when(
            (col("is_correct") == False) & (col("is_correct_no_image") == True),
            "Image Hurt",
        )
        .otherwise("Both Incorrect"),
    )


def run_full_pipeline(source_uri: str, category: str, subset: str, model_id: str, system_message: str, params: dict, limit: int = None) -> daft.DataFrame:
    """
    Run the complete evaluation pipeline.

    Args:
        subset: The Cauldron subset to evaluate (e.g., 'ai2d', 'chartqa')
        model_id: Model ID for the VLM
        limit: Optional row limit for testing

    Returns:
        Collected DataFrame with quadrant classifications
    """
    df = daft.read_parquet(source_uri)
    df = preprocess(df, category=category, subset=subset, model_id=model_id, system_message=system_message, params=params)
    df = run_inference_and_check_correctness(df, model_id, with_image=True)
    df = run_inference_and_check_correctness(df, model_id, with_image=False)
    df = classify_quadrants(df)

    if limit is not None:
        df = df.limit(limit)
   
    return df


if __name__ == "__main__":
    from daft.io import IOConfig, S3Config
    from dotenv import load_dotenv
    load_dotenv()

    CATEGORY = os.getenv("CATEGORY", "general_visual_qna")
    SUBSET = os.getenv("SUBSET", "hateful_memes")
    LIMIT = 10 #os.getenv("LIMIT", None)
    MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE", SYSTEM_MESSAGE)
    PARAMS = {"temperature": 0.7}

    SOURCE_URI = f"s3://daft-public-datasets/the_cauldron/original/{CATEGORY}/{SUBSET}/*.parquet"
    DEST_URI = "s3://daft-public-datasets/the_cauldron/evals/image_ablation/"
    
    daft.set_provider("daft")
    daft.set_planning_config(
        default_io_config=IOConfig(
            s3=S3Config(
                region_name="us-west-2",
                key_id=os.getenv("S3_ACCESS_KEY_ID"),
                access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                session_token=os.getenv("S3_SESSION_TOKEN"),
            )
        )
    )

    print(f"Running evaluation pipeline for {CATEGORY} {SUBSET} with {MODEL_ID} and {SYSTEM_MESSAGE} and {PARAMS} and {LIMIT}")

    df = run_full_pipeline(source_uri=SOURCE_URI, category=CATEGORY, subset=SUBSET, model_id=MODEL_ID, system_message=SYSTEM_MESSAGE, params=PARAMS, limit=LIMIT)

    df.write_parquet(DEST_URI, write_mode="append")

    daft.read_parquet(DEST_URI).show()
