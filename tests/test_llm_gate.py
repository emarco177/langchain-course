import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from experiment import compare_table
from gate import LLM_PROMPT, POLICY, Decision, llm_gate, llm_state
from gold import load_gold

INPUTS = next(r for r in load_gold() if len(r["inputs"]["messages"]) > 1)["inputs"]


class FakeChatModel:
    """Stands in for init_chat_model(...): returns what with_structured_output(include_raw=True) returns."""

    def __init__(self, parsed):
        self.parsed = parsed

    def with_structured_output(self, schema, include_raw=False):
        assert schema is Decision and include_raw
        raw = AIMessage("", usage_metadata={"input_tokens": 800, "output_tokens": 12, "total_tokens": 812})
        return RunnableLambda(lambda messages: {"raw": raw, "parsed": self.parsed, "parsing_error": None})


def test_prompt_and_state_carry_the_same_policy_and_call():
    assert all(text in LLM_PROMPT for text in POLICY.values())
    state = json.loads(llm_state(INPUTS))
    assert state["tool_call"]["name"] == INPUTS["tool_call"]["name"]
    assert [m["role"] for m in state["messages"]] == [m["role"] for m in INPUTS["messages"]]


def test_llm_gate_target(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_USD_PER_1M_IN", "1")
    monkeypatch.setenv("ANTHROPIC_USD_PER_1M_OUT", "5")
    out = llm_gate("anthropic:some-model", chat_model=FakeChatModel(Decision(action="block")))(INPUTS)
    assert out["action"] == "block" and not out["parse_error"]
    assert out["input_tokens"] == 800
    assert out["cost_usd"] == (800 * 1 + 12 * 5) / 1e6


def test_unparseable_answer_goes_to_a_human():
    out = llm_gate("openai:some-model", chat_model=FakeChatModel(None))(INPUTS)
    assert out["action"] == "approve" and out["parse_error"]


def test_compare_table():
    rep = {"accuracy": 0.9, "attacks_allowed": 1, "bill_usd": 503.2, "p50_ms": 40.0, "p95_ms": 90.0, "usd_per_1k": None}
    table = compare_table({"jev-1.14.0": rep, "openai:gpt-6-luna": {**rep, "usd_per_1k": 0.5}})
    assert "jev-1.14.0" in table and "n/a" in table and "0.5000" in table
