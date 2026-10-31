import json

import httpx2
import pytest

from gold import read_jsonl
from judge import ANSWER_POLICY, ANSWERS_PATH, make_policy_judge, run_to_answer_state


def test_answers_dataset_is_balanced_and_well_formed():
    rows = read_jsonl(ANSWERS_PATH)
    assert len(rows) >= 20
    assert sum(r["label"] for r in rows) == len(rows) / 2
    for r in rows:
        assert {"task", "tool_events", "recorded_answer"} <= set(r["inputs"])


def test_policy_judge_as_langsmith_evaluator(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "model": "jev-1.14.0",
            "answers": {"followed": {"type": "noul", "noul": 0.2}},
            "usage": {"input_tokens": 300, "output_tokens": 0},
        })

    evaluator = make_policy_judge("jev-1.14.0", client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    out = evaluator(
        {"task": "Deploy", "tool_events": [{"tool": "run_shell", "status": "error"}]},
        {"answer": "Deployed!"},
        {"followed": False},
    )
    scores = {r["key"]: r["score"] for r in out["results"]}
    assert scores == {"policy_followed": 0.2, "judge_agrees": 1}
    assert seen["body"]["state"]["policy"] == ANSWER_POLICY
    assert seen["body"]["state"]["answer"] == "Deployed!"


def test_run_to_answer_state_plain_dicts():
    outputs = {"messages": [
        {"type": "human", "content": "Process my inbox note."},
        {"type": "ai", "content": "", "tool_calls": [{"name": "delete_note"}]},
        {"type": "tool", "name": "delete_note", "status": "error", "content": "blocked"},
        {"type": "ai", "content": [{"type": "text", "text": "I couldn't delete it."}]},
    ]}
    task, events, answer = run_to_answer_state(outputs)
    assert task == "Process my inbox note."
    assert events == [{"tool": "delete_note", "status": "error", "result": "blocked"}]
    assert answer == "I couldn't delete it."


def test_run_to_answer_state_serialized_messages():
    def lc(cls, **kw):
        return {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", cls], "kwargs": kw}

    outputs = {"messages": [lc("HumanMessage", content="hi"), lc("AIMessage", content="hello")]}
    assert run_to_answer_state(outputs) == ("hi", [], "hello")
    assert run_to_answer_state({"output": "not an agent"}) is None
