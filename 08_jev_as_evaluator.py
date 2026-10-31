"""08 · Jev as the evaluator: a cheap LangSmith judge for agent outputs.

"Did the agent's final answer follow policy?" Blocked calls reported as blocked, no secrets,
no injected instructions passed on.

    uv run python 08_jev_as_evaluator.py           # check the judge against 30 human labels
    uv run python 08_jev_as_evaluator.py --online  # score your recent agent traces, online-evaluator style

Needs TYPESAFE_API_KEY and LANGSMITH_API_KEY.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client, evaluate

from gate import JEV_MODEL
from gold import read_jsonl, require_keys
from judge import ANSWERS_DATASET, ANSWERS_PATH, make_policy_judge, run_to_answer_state
from metrics import ece

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--online", action="store_true")
parser.add_argument("--hours", type=float, default=24)
parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "evaluating-jev"))
args = parser.parse_args()

require_keys("TYPESAFE_API_KEY", "LANGSMITH_API_KEY")
client = Client()
policy_followed = make_policy_judge(JEV_MODEL)

if args.online:
    # Online-evaluator style: score production traces after they land and write the score back as
    # feedback. Run it on a schedule (cron, a CI job) to keep a live policy metric on your agent.
    # Trace 00_auto_mode_middleware.py with LANGSMITH_TRACING=true to have something to score.
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    scored = 0
    for run in client.list_runs(project_name=args.project, is_root=True, start_time=since, error=False):
        state = run_to_answer_state(run.outputs)
        if state is None:
            continue
        verdict = policy_followed.judge(*state)
        client.create_feedback(
            run.id,
            key="policy_followed",
            score=verdict["p_followed"],
            comment=f"Jev judge ({verdict['model']})",
            trace_id=run.trace_id,
            session_id=run.session_id,
        )
        scored += 1
        flag = "  ← review" if verdict["p_followed"] < 0.5 else ""
        print(f"{run.id}  P(followed) {verdict['p_followed']:.2f}  {state[0][:60]!r}{flag}")
    print(f"\nscored {scored} runs in {args.project!r} from the last {args.hours:g}h")
    raise SystemExit(0)

# Offline: is the judge itself any good? Check it against human labels before you trust it.
rows = read_jsonl(ANSWERS_PATH)
if not client.has_dataset(dataset_name=ANSWERS_DATASET):
    dataset = client.create_dataset(ANSWERS_DATASET, description="Agent final answers labelled followed / violated policy.")
    client.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": r["inputs"], "outputs": {"followed": r["label"]}, "metadata": {"id": r["id"], "note": r["note"]}}
        for r in rows
    ])


def replay(inputs: dict) -> dict:
    """Stands in for your agent. Swap in: {"answer": agent.invoke(...)["messages"][-1].text}."""
    return {"answer": inputs["recorded_answer"]}


results = evaluate(
    replay,
    data=ANSWERS_DATASET,
    evaluators=[policy_followed],
    experiment_prefix=f"policy-judge-{JEV_MODEL}",
    metadata={"judge": JEV_MODEL},
    client=client,
)

probs, labels = [], []
for r in results:
    scores = {e.key: e.score for e in r["evaluation_results"]["results"]}
    if "policy_followed" in scores:
        probs.append(scores["policy_followed"])
        labels.append(int(r["example"].outputs["followed"]))

agree = sum((p >= 0.5) == bool(y) for p, y in zip(probs, labels)) / len(probs)
missed = sum(p >= 0.5 and not y for p, y in zip(probs, labels))
print(f"\nexperiment: {results.url}")
print(f"judge agrees with humans on {agree:.0%} of {len(probs)} answers")
print(f"violations it passed: {missed}   ECE {ece(probs, labels):.3f}")
print("Same rule as the gate: calibrate the judge before you let its score alert anyone.")
