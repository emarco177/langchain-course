"""Chat-model configuration for the System One Models chapter.

Jev never generates text, so the middleware examples (05 and 06) still need an
ordinary chat model to *do* the work. Jev's job in those examples is to make the
decisions around that model: which model to run, and whether a tool call is safe.

Keeping the chat models in one place means you swap providers or model names
here once, and both middleware examples follow. Requires `OPENAI_API_KEY` in
your `.env` (see `.env.example`).
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# Model strings in the `provider:model` form accepted by `init_chat_model`.
# The router (05) picks between these two; the tool gate (06) uses FAST_MODEL.
FAST_MODEL = "openai:gpt-5.6-luna"
POWERFUL_MODEL = "openai:gpt-5.6-sol"
