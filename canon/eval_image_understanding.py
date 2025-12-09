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



JUDGE_SYSTEM_PROMPT = """
You are an impartial judge reviewing the results of a textbook academic questions multiple choice benchmark.
Inspect the attached image and provide high-signal feedback on why the model chose its answer.
First, reason about the model's answer with the image and the model's answer without the image.
Second, develop a hypothesis for why the model made the choice it did. 
Third, attribute the failure to a 'question' issue or an 'image' understanding issue.
Finally, assign whether the model's answer with the image is correct and whether the model's answer without the image is correct.
"""

class TextbookAcademicQuestions(BaseModel):
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
        description="Was this a 'question' issue or an 'image' understanding issue or 'other'?",
    )


def preprocess(df: daft.DataFrame, category: str, subset: str, model_id: str,  params: dict) -> daft.DataFrame:
    """Preprocess images and text from The Cauldron format."""

    # Track Evaluation Inputs (Prompt Arguments) 
    df = df.with_columns(
        {
            "category": daft.lit(category), 
            "subset": daft.lit(subset),
            "model_id": daft.lit(model_id),
            "params": daft.lit(json.dumps(params, indent=4)),
        }
    )

    df = (
        df
        .explode("texts")
        .with_column("answer", col("texts")["assistant"].regexp_replace("Answer: ", "").lstrip().rstrip())
    )

    return df


def run_inference(df: daft.DataFrame, model_id: str, with_image: bool = True, params: dict = {}) -> daft.DataFrame:
    """Run structured output inference with or without images."""
    if with_image:
        messages = [col("image"), col("texts")["user"]]
        result_col = "result"
        correct_col = "is_correct"
    else:
        messages = col("texts")["user"]
        result_col = "result_no_image"
        correct_col = "is_correct_no_image"

    return (
        df
        .with_column(
            result_col,
            prompt(
                messages=messages,
                model=model_id,
                use_chat_completions=True,
                return_format=TextbookAcademicQuestions,
                **params,
            ),
        ).with_column(
            correct_col,
            col(result_col)["choice"].lstrip().rstrip() == col("answer"),
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


def run_judge(df, model_id: str):
    """Run VLM-as-a-Judge on failure cases."""
    judge_template = format(
        """Given the image attached and the multiple choice question of <question>{}</question>, 
The model chose the following prediction <model_answer>{}</model_answer> and without the image, the model chose the following prediction <no_image_model_answer>{}</no_image_model_answer>, but the correct answer is <correct_answer>{}</correct_answer>.
""",
        col("texts")["user"],
        col("result")["choice"],
        col("result_no_image")["choice"],
        col("texts")["assistant"],
    )

    return df.where(
        (col("quadrant") == "Image Hurt") | (col("quadrant") == "Both Incorrect")
    ).with_column(
        "judge_response",
        prompt(
            messages=[col("image"), judge_template],
            system_message=JUDGE_SYSTEM_PROMPT,
            model=model_id,
            use_chat_completions=True,
            return_format=JudgeResponse,
        ),
    )


def run_full_pipeline(source_uri: str, category: str, subset: str, model_id: str, params: dict = {}, limit: int = None) -> daft.DataFrame:
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
    if limit is not None:
        limit = int(limit)
        df = df.limit(limit)

    df = preprocess(df, category=category, subset=subset, model_id=model_id, params=params)
    df = run_inference(df, model_id, with_image=True)
    df = run_inference(df, model_id, with_image=False)
    df = classify_quadrants(df)
    #df = run_judge(df, model_id)

    
    return df

if __name__ == "__main__":
    from daft.io import IOConfig, S3Config
    from dotenv import load_dotenv
    load_dotenv()

    CATEGORY = os.getenv("CATEGORY", "textbook_academic_questions")
    SUBSET = os.getenv("SUBSET", "scienceqa")
    LIMIT = os.getenv("LIMIT", None)
    MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    PARAMS = {"temperature": 0.0, "max_tokens": 2}
    USE_LOCAL = os.getenv("USE_LOCAL", "no")

    SOURCE_URI = f"s3://daft-public-datasets/the_cauldron/original/{CATEGORY}/{SUBSET}/*.parquet"
    DEST_URI = "s3://daft-public-datasets/the_cauldron/evals/image_ablation"
    
    if USE_LOCAL == "yes":
        MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
        daft.set_provider("openai", api_key=os.getenv("HF_TOKEN"), base_url="https://router.huggingface.co/v1")
    else:
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

    print(f"Running evaluation pipeline \n category: {CATEGORY} \n subset: {SUBSET} \n model: {MODEL_ID} \n params: {PARAMS} \n limit: {LIMIT}")

    df = run_full_pipeline(source_uri=SOURCE_URI, category=CATEGORY, subset=SUBSET, model_id=MODEL_ID, params=PARAMS, limit=LIMIT)

    df.write_parquet(DEST_URI, write_mode="append")

    df.show()
