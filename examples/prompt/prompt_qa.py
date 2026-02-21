# /// script
# description = "Synthetic Q&A generation pipeline with question generation, answering, and verification"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "python-dotenv", "openai", "pydantic"]
# ///
import daft
from daft.functions import prompt, format
from dotenv import load_dotenv

load_dotenv()

# Start with seed topics
df = daft.from_pydict(
    {
        "topic": ["math", "physics", "python programming"] * 3,
    }
)

df = (
    df.with_column(
        "question",
        prompt(
            format(
                "Generate an orthogonal out of band question about {} ",
                daft.col("topic"),
            ),
            model="gpt-5-nano",
            reasoning={"effort": "low"},
        ),
    )
    .with_column(
        "answer",
        prompt(
            format("Answer this question in detail: {}", daft.col("question")),
            model="gpt-5-mini",
            reasoning={"effort": "high"},
        ),
    )
    .with_column(
        "judgement",
        prompt(
            format(
                "Is the answer to the question correct? Use your tools to verify the answer. <question>{}</question>  <answer>{}</answer>",
                daft.col("question"),
                daft.col("answer"),
            ),
            model="gpt-5.1",
            reasoning={"effort": "high"},
        ),
    )
)

df = df.collect()
df.write_parquet(".data/udfs/synthetic-qa-pairs.parquet")

df.show(9)
