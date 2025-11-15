# /// script
# description = "Prompt an OpenAI model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13","openai","pydantic","python-dotenv", "numpy"]
# ///
import daft
from daft.functions import format, prompt

daft.set_provider(
    "openai", 
    api_key="none", 
    base_url="http://127.0.0.1:1234/v1" # Local LM Studio Server
)

# Create a dataframe with the quotes
df = (
    daft.read_huggingface("nvidia/Nemotron-Personas-USA")

    # Craft Prompt Template
    .with_column(
        "prompt", 
        format(
            "Asumming the persona of <persona>{}</persona> answer the following question: <question>{}</question> ",
            daft.col("professional_persona"),
            daft.lit("Who will win the race to AGI?")
        )
    )

    # Prompt with Chat Completions
    .with_column(
        "response",
        prompt(
            messages = daft.col("prompt"),
            system_message="Impersonating the persona provided, authentically represent your perspective to the prompt posed.",
            model="google/gemma-3-4b", # Make sure LM Studio Server is running with the model loaded
            use_chat_completions=True,
        ),
    )
)

# Show the results
df.select("professional_persona", "response").show(format="fancy", max_width=60)