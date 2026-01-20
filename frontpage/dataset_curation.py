# /// script
# description = "This example highlights how to use Daft Pipelines to enrich samples, remove duplicates, apply quality filters, and generate new examples where coverage is low. With incremental updates and reproducible execution, Daft keeps curated datasets fresh and aligned with evolving source data while minimizing manual labeling and compute cost."
# dependencies = ["daft[openai]"]
# ///
import daft
from daft import col
from daft.functions import prompt, embed_text, cosine_distance, concat
from pydantic import BaseModel

class Eval(BaseModel):
    realism_score: float
    pass_rate_threshold: float
    synthetic_examples: list[str]

# Given a Source Dataset
source_data = daft.read_huggingface("nvidia/Nemotron-Personas-USA")

# And a destination
#curated_data = daft.read_parquet("s3://personas/curated_personas.parquet")

# And desired filters
df = (
    source_data
    # Target 21-55 year olds with a STEM degree. 
    .where(col("age").between(21, 55))
    .where(col("bachelors_field") == "stem_related")
) 

# Check Distribution of Skills by Education
distribution_df = (
    df
    .explode("skills_and_expertise_list")
    .groupby(col("education_level"))
    .list_agg_distinct(col("skills_and_expertise_list"))
)
print(distribution_df.to_pydict())
    
# Use LLM as a Judge to evaluate the distribution of skills and expertise. 


# Then generate synthetic examples to augment the dataset.

df_augmented = df.with_column(
    "synthetic_comment",
    prompt(
        format("Generate a realistic comment for r/{}", col("subreddit")), 
        model="gpt-4o-mini"
    ),
).select("subreddit", col("synthetic_comment").alias("body"))

# Combine original and synthetic, write curated dataset
df_final = df.select("professional_persona", "body").join(
    df_augmented.select("subreddit"),
    on="subreddit",
    how="anti",
    strategy="hash",
)