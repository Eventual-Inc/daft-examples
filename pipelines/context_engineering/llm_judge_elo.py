# /// script
# description = "Context Engineering: LLM-as-a-Judge ELO Ranking Pipeline"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "pydantic", "python-dotenv", "numpy"]
# ///

import os

from dotenv import load_dotenv

import daft
from daft import col, lit
from daft.functions import format, monotonically_increasing_id, prompt

if __name__ == "__main__":
    load_dotenv()

    daft.set_provider(
        "openai",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # ── 1. Define questions, models, and perspectives ─────────────────

    questions = [
        "Write a Python function to check if a string is a palindrome, ignoring case.",
        "Explain the difference between a process and a thread in operating systems.",
        "Write a short story about a time traveler who can only go back 5 minutes.",
        "Compose a haiku about a distributed database.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Explain the concept of quantum entanglement to a 5-year-old.",
    ]

    models = [
        "openai/gpt-5-mini",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-chat-v3-0324",
    ]

    perspectives = [
        "a neutral",
        "a nuanced",
        "an objective",
    ]

    max_tokens = 200
    judge_model = "openai/gpt-5-mini"

    # ── 2. Build question x perspective grid ──────────────────────────

    df_questions = daft.from_pydict({"question": questions})
    df_perspectives = daft.from_pydict({"perspective": perspectives})
    df_grid = df_questions.join(df_perspectives, how="cross")

    # ── 3. Generate responses for each model ──────────────────────────

    responses = []
    for model_id in models:
        df_resp = (
            df_grid.with_column(
                "response",
                prompt(
                    messages=format(
                        "Generate a response to the following question in under {} tokens: {}",
                        lit(max_tokens),
                        col("question"),
                    ),
                    model=model_id,
                ),
            )
            .with_column("model", lit(model_id))
            .with_column("id", monotonically_increasing_id())
        )
        responses.append(df_resp)

    # Union all model responses
    df_all = responses[0]
    for r in responses[1:]:
        df_all = df_all.union_all(r)

    print(f"Perspectives: {len(perspectives)}, Models: {len(models)}, Questions: {len(questions)}")
    print(f"Total responses: {df_all.count_rows()}")

    # ── 4. Create response pairs (Model A vs Model B) ─────────────────

    df_left = df_all.select(
        col("question"),
        col("perspective"),
        col("model").alias("model_a"),
        col("response").alias("response_a"),
    )

    df_right = df_all.select(
        col("question").alias("question_r"),
        col("perspective").alias("perspective_r"),
        col("model").alias("model_b"),
        col("response").alias("response_b"),
    )

    df_pairs = df_left.join(
        df_right,
        left_on=["question", "perspective"],
        right_on=["question_r", "perspective_r"],
    ).where(col("model_a") < col("model_b"))

    num_pairs = df_pairs.count_rows()
    print(f"Pairs to judge: {num_pairs}")

    if num_pairs == 0:
        print("No pairs created.")
        exit()

    # ── 5. Judge with LLM ─────────────────────────────────────────────

    judge_template = (
        "You are an impartial judge evaluating two AI responses to the same question.\n\n"
        "Question: {}\n\n"
        "[Response A]\n{}\n\n"
        "[Response B]\n{}\n\n"
        "Which response is better from {} perspective? "
        "Consider accuracy, helpfulness, and clarity.\n"
        "Output EXACTLY 'A', 'B', or 'TIE'. No explanation."
    )

    df_judged = df_pairs.with_column(
        "verdict",
        prompt(
            messages=format(
                judge_template,
                col("question"),
                col("response_a"),
                col("response_b"),
                col("perspective"),
            ),
            model=judge_model,
            max_output_tokens=10,
            system_message="Output only A, B, or TIE.",
        ),
    )

    # ── 6. Compute ELO ratings ────────────────────────────────────────

    results = df_judged.select("model_a", "model_b", "verdict").collect().to_pydict()

    elo_ratings = {m: 1000.0 for m in models}
    k_factor = 32

    for ma, mb, v in zip(results["model_a"], results["model_b"], results["verdict"]):
        v_clean = v.strip().upper().replace(".", "")
        ra, rb = elo_ratings[ma], elo_ratings[mb]

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 / (1 + 10 ** ((ra - rb) / 400))

        if v_clean == "A":
            sa, sb = 1.0, 0.0
        elif v_clean == "B":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        elo_ratings[ma] = ra + k_factor * (sa - ea)
        elo_ratings[mb] = rb + k_factor * (sb - eb)

    # ── 7. Display rankings ───────────────────────────────────────────

    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    daft.from_pydict(
        {
            "rank": list(range(1, len(sorted_ratings) + 1)),
            "model": [x[0] for x in sorted_ratings],
            "elo": [x[1] for x in sorted_ratings],
        }
    ).show()
