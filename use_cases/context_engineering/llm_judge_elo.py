# /// script
# description = "Context Engineering: LLM-as-a-Judge ELO Ranking Pipeline"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "openai", "pydantic", "python-dotenv", "numpy"]
# ///

import os
import daft
from daft import col, lit
from daft.functions import monotonically_increasing_id, prompt, format, monotonically_increasing_id
from dotenv import load_dotenv
import numpy as np

    

if __name__ == "__main__":

    # Load environment variables
    load_dotenv()

    # Configure OpenRouter (or OpenAI compatible provider)
    # Ensure OPENROUTER_API_KEY is set in your .env or environment
    daft.set_provider(
        "openai",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # 1. Define Prompts, Models, and Perspectives to Test
    questions = [
        # Coding
        "Write a Python function to check if a string is a palindrome, ignoring case.",
        "Explain the difference between a process and a thread in operating systems.",
        "How do I vertically and horizontally center a div in CSS?",
        # Creative
        "Write a short story about a time traveler who can only go back 5 minutes.",
        "Compose a haiku about a distributed database.",
        "Write a polite email declining a job offer due to salary mismatch.",
        # Reasoning
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        # Knowledge / Explanation
        "Explain the concept of quantum entanglement to a 5-year-old.",
        "What are the primary differences between a relational and a non-relational database?",
        "Summarize the main themes of the book '1984' by George Orwell."
    ]

    models = [
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/gpt-5",
        "openai/gpt-5.1",
        "x-ai/grok-code-fast-1",
        "x-ai/grok-4-fast",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "minimax/minimax-m2",
        "openrouter/sherlock-think-alpha",
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-chat-v3.1",
        "z-ai/glm-4.6",
        "qwen/qwen3-235b-a22b-2507",
        "deepseek/deepseek-v3.2-exp",
    ]

    perspectives = [
        "your",
        "a neutral",
        "a nuanced",
        "an instinctual",
        "an opposite",
        "an orthogonal",
        "an inverted",
        "an adjacent",
        "an objective",
        "a subjective",
        "an abjective",
        "a superjective",
    ]

    # Parameters
    max_tokens = 200
    judge_model = "openai/gpt-5" # More robust instruction following for judging
    
    # 1. Explode to create a full cartesian product of questions, models, and perspectives
    df = daft.from_pydict({
        "questions": list(questions),
        "perspectives": list(perspectives),
    }).explode("questions").explode("perspectives") 

    
    # 2. Generate Responses (M Models x N Perspectives x P Questions)
    responses = []
    for i, model_id in enumerate(models):

        df_responses = df.with_column(
            "response",
            prompt(
                messages=format("Generate a response to the following question in under {} tokens: {}", lit(max_tokens), col("questions")),
                model=model_id,
                max_tokens=max_tokens,
                use_chat_completions=True
            )
        ).with_column("model", lit(model_id))
        responses.append(df_responses)

    df_responses.collect().concat()
    

    # Add Monotonic ID Column
    df = df.with_column(
        "id",
        monotonically_increasing_id()
    )

    print(f"Num Perspectives: {len(perspectives)}")
    print(f"Num Models: {len(models)}")
    print(f"Num Questions: {len(questions)}")
    print(f"Total Responses: {df.count_rows()}")
    print(f"Expected Responses: {len(models) * len(perspectives) * len(questions)}")

    # 4. Generate Response Pairs
    # Self-join to create pairs (Model A vs Model B) for the same question
    # We filter to ensure we don't compare a model to itself, and we can do one-way comparisons 
    # (A < B) to reduce judge calls, or both ways to handle position bias. 
    # For simplicity/cost, we'll do A < B (combinations).
    
    df_left = df_responses.select(
        col("id").alias("id"), 
        col("question").alias("question"), 
        col("model").alias("model_a"), 
        col("response").alias("response_a")
    )
    
    df_right = df_responses.select(
        col("id").alias("id_r"), 
        col("model").alias("model_b"), 
        col("response").alias("response_b")
    )

    df_pairs = df_left.join(df_right, left_on="id", right_on="id_r")
    
    # Filter for unique pairs (Model A < Model B) to avoid duplicates and self-comparisons
    df_pairs = df_pairs.where(col("model_a") < col("model_b"))
    
    print(f"Total pairs to judge: {df_pairs.count_rows()}")
    if df_pairs.count_rows() == 0:
        print("No pairs created. Check your model names and join logic.")
        return

    # 5. Grade with LLM-as-a-Judge
    # Judge Model
    judge_model = "openai/gpt-4o-mini" # More robust instruction following for judging
    
    print(f"Judging pairs using {judge_model}...")

    judge_prompt_template = f"""
    You are an impartial judge evaluating the quality of two AI responses to the same user question.
    
    Question: {}
    
    [Response A]
    {}
    
    [Response B]
    {}
    
    Which response is better? Consider accuracy, precision, helpfulness, and clarity from {} perspective.
    Output EXACTLY 'A' if Response A is better, 'B' if Response B is better, or 'TIE' if they are equal.
    Do not provide explanations, just the single word verdict.
    """
    
    df_pairs = df_pairs.with_column(
        "judge_prompt",
        format(
            judge_prompt_template,
            col("question"),
            col("response_a"),
            col("response_b"),
            col("perspective")
        )
    )

    df_pairs = df_pairs.with_column(
        "verdict",
        prompt(
            messages=col("judge_prompt"),
            model=judge_model,
            max_tokens=10,
            use_chat_completions=True,
            system_message="You are a helpful assistant acting as a judge. Output only A, B, or TIE."
        )
    )

    # Clean verdict (remove extra whitespace/punctuation just in case)
    # Daft doesn't have extensive regex replace yet easily exposed? 
    # We'll trust the model or clean in python.
    
    # Collect results to Python for ELO calculation
    # (ELO is sequential/iterative, harder to do in pure parallel dataframe ops)
    print("Computing ELO ratings...")
    results = df_pairs.select("model_a", "model_b", "verdict").to_pydict()
    
    if len(results["verdict"]) > 0:
        print(f"Sample verdicts: {results['verdict'][:5]}")
    else:
        print("No verdicts collected.")
    
    # 6. Compute ELO
    elo_ratings = {m: 1000.0 for m in models}
    k_factor = 32

    models_a = results["model_a"]
    models_b = results["model_b"]
    verdicts = results["verdict"]

    for ma, mb, v in zip(models_a, models_b, verdicts):
        v_clean = v.strip().upper().replace(".", "")
        
        ra = elo_ratings[ma]
        rb = elo_ratings[mb]
        
        # Expected scores
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 / (1 + 10 ** ((ra - rb) / 400))
        
        sa = 0.0
        sb = 0.0
        
        if v_clean == 'A':
            sa = 1.0
        elif v_clean == 'B':
            sb = 1.0
        else: # TIE
            sa = 0.5
            sb = 0.5
            
        # Update
        elo_ratings[ma] = ra + k_factor * (sa - ea)
        elo_ratings[mb] = rb + k_factor * (sb - eb)

    # 7. Rank and Display
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Final ELO Rankings ---")
    print(f"{'Rank':<5} {'Model':<40} {'ELO':<10}")
    print("-" * 60)
    for i, (model, rating) in enumerate(sorted_ratings, 1):
        print(f"{i:<5} {model:<40} {rating:.2f}")

    # Create a Daft dataframe for the final results and show it
    final_data = {
        "rank": [i for i in range(1, len(sorted_ratings) + 1)],
        "model": [x[0] for x in sorted_ratings],
        "elo": [x[1] for x in sorted_ratings]
    }
    daft.from_pydict(final_data).show()
