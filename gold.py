"""The gold set: labelled tool calls, split into dev (tune on it) and heldout (report on it once)."""

import json
import os
from collections import Counter
from pathlib import Path

from costs import LABELS

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold_tool_calls.jsonl"
RESULTS_DIR = ROOT / "results"
DATASET_NAME = os.getenv("GOLD_DATASET", "evaluating-jev-gold")
SPLITS = ("dev", "heldout")


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gold(path: Path | str = GOLD_PATH) -> list[dict]:
    """Load and sanity-check the gold set."""
    rows = read_jsonl(path)
    ids = Counter(r["id"] for r in rows)
    if dupes := [i for i, n in ids.items() if n > 1]:
        raise ValueError(f"duplicate ids: {dupes}")
    for r in rows:
        if r["label"] not in LABELS or r["split"] not in SPLITS:
            raise ValueError(f"{r['id']}: bad label or split")
    # Variants of one scenario must share a split, or dev leaks into heldout.
    splits_per_group: dict[str, set] = {}
    for r in rows:
        splits_per_group.setdefault(r["group"], set()).add(r["split"])
    if leaks := [g for g, s in splits_per_group.items() if len(s) > 1]:
        raise ValueError(f"groups in both splits: {leaks}")
    return rows


def split(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if name == "all" or r["split"] == name]


def to_langsmith_example(row: dict) -> dict:
    """One gold row → one LangSmith example. Inputs are what the gate sees; outputs are the answer key."""
    return {
        "inputs": row["inputs"],
        "outputs": {"label": row["label"]},
        "metadata": {k: row[k] for k in ("id", "group", "split", "tags", "note")},
        "split": row["split"],
    }


def require_keys(*names: str) -> None:
    """Exit with a clear message instead of a stack trace when a live step has no keys."""
    if missing := [n for n in names if not os.getenv(n)]:
        raise SystemExit(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and set them. "
            "This step makes live API calls."
        )
