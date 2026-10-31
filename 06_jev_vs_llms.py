"""06 · Jev vs LLM judges on the same gold set: accuracy, latency, and $ per 1k decisions.

Same policy text, same state, same evaluators. The LLMs come from env vars
(ANTHROPIC_JUDGE_MODEL, OPENAI_JUDGE_MODEL), so a model rename is a config change.

Needs TYPESAFE_API_KEY and LANGSMITH_API_KEY, plus ANTHROPIC_API_KEY and/or OPENAI_API_KEY.
A baseline whose key is missing is skipped. Reuses Jev's results from 03 when they exist.
"""

import argparse
import os

from dotenv import load_dotenv

from experiment import (
    JEV_EVALUATORS,
    ROW_EVALUATORS,
    compare_table,
    load_results,
    report,
    results_path,
    run_experiment,
)
from gate import JEV_MODEL, LLM_BASELINES, jev_gate, llm_gate
from gold import require_keys

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
split = parser.parse_args().split

require_keys("TYPESAFE_API_KEY", "LANGSMITH_API_KEY")

reports = {}
if results_path(JEV_MODEL, split).exists():
    reports[JEV_MODEL] = report(load_results(JEV_MODEL, split))
else:
    rows = run_experiment(jev_gate(JEV_MODEL), name=JEV_MODEL, split=split,
                          evaluators=JEV_EVALUATORS, metadata={"model": JEV_MODEL})
    reports[JEV_MODEL] = report(rows)

KEY_FOR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
for model in LLM_BASELINES:
    key = KEY_FOR.get(model.split(":", 1)[0])
    if key and not os.getenv(key):
        print(f"skipping {model}: set {key} to include it")
        continue
    rows = run_experiment(llm_gate(model), name=model, split=split,
                          evaluators=ROW_EVALUATORS, metadata={"model": model},
                          max_concurrency=4)
    if errors := sum(r.get("parse_error", False) for r in rows):
        print(f"  {model}: {errors} unparseable answers sent to a human")
    reports[model] = report(rows)

print("\n" + compare_table(reports))
print("\n$ / 1k uses the *_USD_PER_1M_IN / _OUT prices in costs.py; n/a means no price is set.")
print("The LLMs return a label with no confidence, so you can't cut lanes on them the way 05 does.")
print("To price 05's 'bigger LLM' lane with real decisions:")
for model in reports:
    if model != JEV_MODEL:
        print(f"  uv run python 05_threshold_from_cost.py --llm-results {model}")
