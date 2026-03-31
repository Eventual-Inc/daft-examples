# /// script
# description = "Context Engineering: Few-Shot Example Selection Pipeline"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5", "python-dotenv", "pydantic"]
# ///

import os
import daft
from daft import col, lit, Window
from daft.functions import embed_text, prompt, format, rank
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv()

    daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

    # ══════════════════════════════════════════════════════════════════════
    # 1. Build an example bank of labeled Q&A pairs
    # ══════════════════════════════════════════════════════════════════════

    example_bank = daft.from_pydict(
        {
            "example_question": [
                "Write a Python function that reverses a linked list in place.",
                "Explain the difference between a stack and a queue.",
                "Write a SQL query to find the second highest salary in a table.",
                "What is the derivative of x^3 + 2x^2 - 5x + 7?",
                "Solve for x: 3x + 7 = 22.",
                "What is the probability of rolling two sixes with two fair dice?",
                "Write a short poem about the ocean at sunset.",
                "Come up with a creative name for a coffee shop on the moon.",
                "Describe a futuristic city in three sentences.",
                "What are the trade-offs between recursion and iteration?",
            ],
            "ideal_answer": [
                "def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev",
                "A stack is LIFO (Last In, First Out) — the most recently added element is removed first. A queue is FIFO (First In, First Out) — the earliest added element is removed first. Stacks are used for undo operations and call stacks; queues are used for task scheduling and BFS.",
                "SELECT MAX(salary) AS second_highest_salary FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);",
                "The derivative is 3x^2 + 4x - 5, found by applying the power rule to each term.",
                "Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5.",
                "Each die has a 1/6 chance of landing on six. The probability of both showing six is (1/6) * (1/6) = 1/36.",
                "Golden light melts into the waves,\nThe horizon hums a lullaby.\nSalt and color fill the air,\nAs the sun slips softly by.",
                "How about 'Lunar Drip' — evoking both the low gravity and fresh coffee dripping into your cup among the stars.",
                "Towers of translucent glass spiral above cloud-level gardens connected by silent mag-lev bridges. Autonomous drones weave between buildings delivering everything from groceries to medical supplies in minutes. Beneath the streets, a network of hyperloops links the city to every continent on Earth.",
                "Recursion offers elegant, readable solutions for problems with natural sub-structure (trees, divide-and-conquer) but risks stack overflow and repeated work without memoization. Iteration uses explicit loops and constant stack space, making it more memory-efficient. The best choice depends on the problem structure and performance constraints.",
            ],
            "category": [
                "coding",
                "coding",
                "coding",
                "math",
                "math",
                "math",
                "creative",
                "creative",
                "creative",
                "coding",
            ],
        }
    )

    # ══════════════════════════════════════════════════════════════════════
    # 2. Define new test queries
    # ══════════════════════════════════════════════════════════════════════

    test_queries = daft.from_pydict(
        {
            "query": [
                "Write a Python function to check if a binary tree is balanced.",
                "What is the integral of 2x * e^(x^2)?",
                "Write a haiku about a rainy afternoon.",
            ],
        }
    )

    # ══════════════════════════════════════════════════════════════════════
    # 3. Embed example bank questions and test queries
    # ══════════════════════════════════════════════════════════════════════

    EMBED_MODEL = "text-embedding-3-small"

    example_bank = example_bank.with_column(
        "example_embedding",
        embed_text(col("example_question"), provider="openai", model=EMBED_MODEL),
    )

    test_queries = test_queries.with_column(
        "query_embedding",
        embed_text(col("query"), provider="openai", model=EMBED_MODEL),
    )

    # ══════════════════════════════════════════════════════════════════════
    # 4. Cross join queries with examples and compute cosine distance
    # ══════════════════════════════════════════════════════════════════════

    df_cross = test_queries.join(example_bank, how="cross")

    df_with_distance = df_cross.with_column(
        "distance",
        col("query_embedding").cosine_distance(col("example_embedding")),
    )

    # ══════════════════════════════════════════════════════════════════════
    # 5. Rank examples per query and select top-K (K=3) nearest
    # ══════════════════════════════════════════════════════════════════════

    K = 3

    window = Window().partition_by("query").order_by(col("distance").asc())

    df_top_k = df_with_distance.with_column(
        "rank", rank().over(window),
    ).where(col("rank") <= K)

    # ══════════════════════════════════════════════════════════════════════
    # 6. Assemble few-shot context string per query
    # ══════════════════════════════════════════════════════════════════════

    # Build a formatted example string for each selected pair
    df_formatted = df_top_k.with_column(
        "formatted_example",
        format(
            "Q: {}\nA: {}",
            col("example_question"),
            col("ideal_answer"),
        ),
    )

    # Group by query and join the selected examples into a single context block
    df_context = (
        df_formatted.groupby("query")
        .agg(col("formatted_example").alias("examples_list"))
        .with_column(
            "few_shot_context",
            col("examples_list").list_join(delimiter="\n\n"),
        )
    )

    # ══════════════════════════════════════════════════════════════════════
    # 7. Send assembled prompt to the LLM
    # ══════════════════════════════════════════════════════════════════════

    GENERATION_MODEL = "gpt-5-mini"

    df_result = df_context.with_column(
        "response",
        prompt(
            messages=format(
                "Here are some examples of high-quality question-answer pairs:\n\n{}\n\nNow answer the following question in the same style and level of detail:\n\nQ: {}",
                col("few_shot_context"),
                col("query"),
            ),
            model=GENERATION_MODEL,
            provider="openai",
            system_message="You are a helpful assistant. Use the provided examples to guide the style, depth, and format of your answer.",
        ),
    )

    # ══════════════════════════════════════════════════════════════════════
    # 8. Show final results
    # ══════════════════════════════════════════════════════════════════════

    df_result.select("query", "few_shot_context", "response").show()
