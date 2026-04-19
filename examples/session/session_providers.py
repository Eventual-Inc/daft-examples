# /// script
# description = "Session providers - attach, switch, and use multiple AI providers"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "python-dotenv"]
# ///
import os

from dotenv import load_dotenv

import daft
from daft import Session
from daft.ai.openai.provider import OpenAIProvider
from daft.functions import prompt

QUESTIONS = daft.from_pydict(
    {"question": ["What is the capital of France?", "What is 2 + 2?"]}
)


def global_provider() -> None:
    """Simplest setup: load a built-in provider by name and use it as the default."""
    daft.set_provider("openai", api_key=os.environ["OPENAI_API_KEY"])
    QUESTIONS.with_column("answer", prompt(daft.col("question"), model="gpt-4o-mini")).show()


def custom_endpoint_provider() -> None:
    """Point at any OpenAI-compatible service (OpenRouter, Gemini, vLLM, Ollama) via base_url."""
    daft.set_provider(
        "openai",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )


def named_provider_objects() -> Session:
    """Attach multiple named Provider instances to a session for fine-grained control."""
    sess = Session()
    sess.attach_provider(
        OpenAIProvider(name="OpenAI", api_key=os.environ["OPENAI_API_KEY"])
    )
    sess.attach_provider(
        OpenAIProvider(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    )
    return sess


def switch_active_provider(sess: Session) -> None:
    """Switch the session's active provider without detaching or reconfiguring."""
    sess.set_provider("OpenAI")
    sess.set_provider("OpenRouter")


def override_provider_per_call(sess: Session) -> None:
    """Override the session default by passing a provider directly to prompt/embed."""
    QUESTIONS.with_column(
        "answer",
        prompt(
            daft.col("question"),
            provider=sess.get_provider("OpenAI"),
            model="gpt-4o-mini",
        ),
    ).show()


def provider_introspection(sess: Session) -> None:
    """Inspect and detach providers attached to a session."""
    sess.has_provider("OpenAI")
    sess.current_provider()
    sess.detach_provider("OpenRouter")


if __name__ == "__main__":
    load_dotenv()

    global_provider()
    custom_endpoint_provider()

    sess = named_provider_objects()
    switch_active_provider(sess)
    override_provider_per_call(sess)
    provider_introspection(sess)
