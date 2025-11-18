# /// script
# description = "Prompt a Gemini 3 model to review the code"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13","openai","pydantic","python-dotenv", "numpy"]
# ///
import os
import daft
from daft import col
from daft.functions import format, prompt, file, unnest
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

pwd = os.path.dirname(os.path.abspath(__file__))

daft.set_provider(
    "openai", 
    api_key=os.environ.get("GEMINI_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class Evaluation(BaseModel):
    quality_score: int = Field(description="The quality of the code on a scale of 1 to 10")
    improvements: list[str] = Field(description="Suggestions for improvements")
    reasoning: str = Field(description="The reasoning behind the evaluation")

# Create a dataframe with the quotes
df = (
    daft.from_glob_path(os.path.join(pwd, "*.py"))

    # Wrap the script path into a file object
    .with_column(
        "script",
        file(daft.col("path"))
    )

    # Prompt with Chat Completions
    .with_column(
        "response",
        prompt(
            messages = [daft.lit("Given the following script, evaluate the quality of the code and suggest improvements"), daft.col("script")],
            system_message="You are a helpful assistant that evaluates the quality of the code and suggests improvements.",
            model="gemini-3-pro-preview", 
            use_chat_completions=True,
            return_format=Evaluation,
        ),
    )

    .select("path", unnest(col("response")))
)

# Show the results
df.write_json("../../.data/prompt/prompt_gemini3_code_review.json")
df.show(format="fancy", max_width=60)