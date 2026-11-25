"""Model configuration for the Agent Harnesses chapter.

Every example in this project does `from models import model` and hands that
`model` straight to `create_deep_agent(...)`. Keeping the model in one place means
you swap providers or model names here once, and all five examples follow.

Default: OpenAI `gpt-6.2-sol`. Requires `OPENAI_API_KEY` in your `.env` (see
`.env.example`). To use a different model, change the string below — for example
`"openai:gpt-6-luna"` for a cheaper run, or an Anthropic model such as
`"anthropic:claude-haiku-4-5"` (set `ANTHROPIC_API_KEY` instead).
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from langchain.chat_models import init_chat_model

# The single model shared by every example in this project.
# Requires OPENAI_API_KEY in .env.
model = init_chat_model("openai:gpt-6.2-sol", reasoning_effort="none")
