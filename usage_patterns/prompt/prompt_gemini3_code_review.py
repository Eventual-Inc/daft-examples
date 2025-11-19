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

class Evaluation(BaseModel):
    quality_score: int = Field(description="The quality of the code on a scale of 1 to 10")
    improvements: list[str] = Field(description="Suggestions for improvements")
    reasoning: str = Field(description="The reasoning behind the evaluation")


daft.set_provider(
    "openai", 
    api_key=os.environ.get("GEMINI_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

df = (
    # Discover Python Files
    daft.from_glob_path(os.path.join(pwd, "*.py"))

    # Prompt the Gemini 3 model to review the code using the chat completions API
    .with_column(
        "review",
        prompt(
            messages = [
                daft.lit("Evaluate the quality of the following python code and suggest improvements"), 
                file(daft.col("path"))
            ],
            system_message="You are a principal python developer.",
            model="gemini-3-pro-preview", 
            use_chat_completions=True,
            return_format=Evaluation,
        ),
    )

    .select("path", unnest(col("review")))
)

# Write the results to a JSON file
df = df.write_json("../../.data/prompt/prompt_gemini3_code_review.json")

# Show the results
df.show(format="fancy", max_width=60)