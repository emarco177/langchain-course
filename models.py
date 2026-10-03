"""Chat models for 05 and 06. Swap providers here."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# `provider:model` strings. 05 routes between both; 06 uses FAST_MODEL.
FAST_MODEL = "openai:gpt-6-luna"
POWERFUL_MODEL = "openai:gpt-6-sol"
