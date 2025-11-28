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
from pydantic import BaseModel, Field

import daft
from daft import col
from daft.functions import (
    prompt,
    unnest,
    when,
    monotonically_increasing_id,
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


# ========================
# Configuration 
# ========================

# System Prompts
SYSTEM_PROMPT_WITH_IMAGE = (
    "Observe the attached image and respond to the multiple choice question "
    "with just the letter corresponding to the correct answer."
)

SYSTEM_PROMPT_NO_IMAGE = (
    "Respond to the multiple choice question with just the letter "
    "corresponding to the correct answer."
)

JUDGE_SYSTEM_PROMPT = """
You are an impartial judge reviewing the results of a visual Q&A benchmark.
Inspect the attached image and provide high-signal feedback on why the model chose its answer.
Focus on image understanding—your feedback should help improve accuracy when images are attached.
Do not propose improvements. Simply diagnose the failure mode.
"""

# ========================
# Pipeline Stages
# ========================


def load_cauldron(subset: str = "ai2d"):
    """Load a subset from The Cauldron dataset."""
    return daft.read_huggingface(f"HuggingFaceM4/the_cauldron/{subset}")


def preprocess(df):
    """Preprocess images and text from The Cauldron format."""
    return (
        df.explode("images")
        .with_column("image", col("images")["bytes"].decode_image())
        .explode("texts")
        .select(unnest(col("texts")), "image")
        .with_column("answer", col("assistant").regexp_replace("Answer: ", ""))
    )


def run_inference(df, model_id: str, with_image: bool = True):
    """Run structured output inference with or without images."""
    if with_image:
        messages = [col("image"), col("user")]
        system = SYSTEM_PROMPT_WITH_IMAGE
        result_col = "result"
        correct_col = "is_correct"
    else:
        messages = col("user")
        system = SYSTEM_PROMPT_NO_IMAGE
        result_col = "result_no_image"
        correct_col = "is_correct_no_image"

    return df.with_column(
        result_col,
        prompt(
            messages=messages,
            system_message=system,
            model=model_id,
            use_chat_completions=True,
            return_format=ChoiceResponse,
        ),
    ).with_column(
        correct_col,
        col(result_col)["choice"].lstrip().rstrip()
        == col("answer").lstrip().rstrip(),
    )


def classify_quadrants(df):
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
        """The model answered <predicted>{}</predicted> but the correct answer is <correct>{}</correct>.

Without the image, the model answered <no_image>{}</no_image>.

Question: {}

Analyze why the model failed.""",
        col("result")["choice"],
        col("answer"),
        col("result_no_image")["choice"],
        col("user"),
    )

    return df.where(
        (col("quadrant") == "Image Hurt") | (col("quadrant") == "Both Incorrect")
    ).with_column(
        "judge",
        prompt(
            messages=[col("image"), judge_template],
            system_message=JUDGE_SYSTEM_PROMPT,
            model=model_id,
            use_chat_completions=True,
            return_format=JudgeResponse,
        ),
    )


def run_full_pipeline(subset: str, model_id: str, limit: int = None):
    """
    Run the complete evaluation pipeline.

    Args:
        subset: The Cauldron subset to evaluate (e.g., 'ai2d', 'chartqa')
        model_id: Model ID for the VLM
        limit: Optional row limit for testing

    Returns:
        Collected DataFrame with quadrant classifications
    """
    df = load_cauldron(subset)
    df = preprocess(df)

    if limit:
        df = df.limit(limit)

    df = run_inference(df, model_id, with_image=True)
    df = run_inference(df, model_id, with_image=False)
    df = df.with_columns({
        "id":  monotonically_increasing_id(),
        "model_id": daft.lit(model_id),
    })
    df = classify_quadrants(df)

    return df.collect()


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    # ========================
    # Configuration 
    # ========================

    # Model & Provider
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    PROVIDER = "daft"  # "daft" or "openai"

    # Output
    OUTPUT_DIR = ".data/image_understanding_eval"
    SKIP_JUDGE = False

    # Defaults (can be overridden via CLI: --subset, --limit)
    DEFAULT_SUBSET = "ai2d"
    DEFAULT_LIMIT = None  # None = full dataset, set to int for testing

    # ========================
    # CLI Arguments 
    # ========================

    # CLI overrides for subset and limit 
    parser = argparse.ArgumentParser(
        description="Evaluate VLM image understanding with structured outputs"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help=f"The Cauldron subset to evaluate (default: {DEFAULT_SUBSET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit number of rows (default: no limit)",
    )
    args = parser.parse_args()

    # ========================
    # Provider Setup
    # ========================

    if os.environ.get("HF_TOKEN", None) is not None:
        daft.set_provider(
            "openai",
            api_key=os.environ["HF_TOKEN"],
            base_url="https://router.huggingface.co/v1",
        )
    else:
        daft.set_provider("daft")

    # ========================
    # Run Evaluation Pipeline
    # ========================

    # Run the main pipeline
    print(f"\nRunning evaluation pipeline for {MODEL_ID} on {args.subset}...")
    df_results = run_full_pipeline(args.subset, MODEL_ID, args.limit)

    # Calculate and display accuracy
    total = df_results.count_rows()
    accuracy_with_image = (
        df_results.where(col("is_correct")).count_rows() / total
    )
    accuracy_no_image = (
        df_results.where(col("is_correct_no_image")).count_rows() / total
    )

    print(f"\nResults ({total} rows):")
    print(f"  Accuracy with image:    {accuracy_with_image:.1%}")
    print(f"  Accuracy without image: {accuracy_no_image:.1%}")
    print(f"  Delta:                  {accuracy_with_image - accuracy_no_image:+.1%}")

    # Display quadrant distribution
    print("\nQuadrant Distribution:")
    df_results.groupby("quadrant").count().select(
        "quadrant", col("id").alias("count")
    ).show()

    # ========================
    # Run Judge Evaluation
    # ========================

    if not SKIP_JUDGE:
        print("\nRunning VLM-as-a-Judge on failures...")
        df_judge = run_judge(df_results, MODEL_ID).collect()

        judge_count = df_judge.count_rows()
        print(f"Judged {judge_count} failures")

        if judge_count > 0:
            print("\nSample Judge Feedback:")
            df_judge.select(
                "id",
                "quadrant",
                col("judge")["reasoning"].alias("reasoning"),
                col("judge")["hypothesis"].alias("hypothesis"),
                col("judge")["attribution"].alias("attribution"),
            ).show()

    # ========================
    # Persist Results
    # ========================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, f"{args.subset}_results.parquet")
    df_results.write_parquet(results_path)
    print(f"\nResults saved to {results_path}")

    if not SKIP_JUDGE:
        judge_path = os.path.join(OUTPUT_DIR, f"{args.subset}_judge.parquet")
        df_judge.write_parquet(judge_path)
        print(f"Judge feedback saved to {judge_path}")

    print("\nEvaluation complete!")