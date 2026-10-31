import json
from types import SimpleNamespace

import httpx2
import pytest

from gate import QUESTIONS, auto_mode_action, gate_state, jev_gate, to_messages
from gold import load_gold

INPUTS = load_gold()[0]["inputs"]


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def test_risk_question_is_the_one_auto_mode_asks(fake_key):
    from langchain_typesafe.experimental.middleware import AutoModeMiddleware
    from langchain_typesafe.experimental.middleware.auto_mode import _risk_questions

    middleware = AutoModeMiddleware(tools=["run_shell"])
    assert _risk_questions(middleware.config)["is_risky"] == QUESTIONS["is_risky"]


def test_state_has_the_shape_auto_mode_builds(fake_key):
    from langchain_typesafe.experimental.middleware import AutoModeMiddleware

    request = SimpleNamespace(
        tool_call=INPUTS["tool_call"],
        state={"messages": to_messages(INPUTS["messages"])},
        tool=SimpleNamespace(description=INPUTS["tool_description"]),
    )
    assert gate_state(INPUTS) == AutoModeMiddleware._classification_state(request)


def test_to_messages_maps_roles():
    msgs = to_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reading"},
        {"role": "tool", "name": "read_file", "content": "data"},
    ])
    assert [m.type for m in msgs] == ["human", "ai", "tool"]
    assert msgs[2].name == "read_file"


def test_auto_mode_threshold():
    assert auto_mode_action(0.49) == "allow"
    assert auto_mode_action(0.5) == "block"


def test_jev_gate_target_with_a_fake_transport(fake_key):
    """End to end through TypeSafeClassifier, with the HTTP layer faked."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "model": "jev-1.14.0",
            "answers": {
                "is_risky": {"type": "noul", "noul": 0.12},
                "action": {"type": "choice", "choice": "allow", "confidence": 0.81,
                           "probabilities": {"allow": 0.9, "approve": 0.07, "block": 0.03}},
            },
            "usage": {"input_tokens": 1000, "output_tokens": 0},
        })

    target = jev_gate("jev-1.14.0", client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    out = target(INPUTS)

    assert seen["body"]["model"] == "jev-1.14.0"
    assert set(seen["body"]["questions"]) == {"is_risky", "action"}
    assert seen["body"]["state"]["tool_call"]["name"] == INPUTS["tool_call"]["name"]
    assert out["action"] == "allow" and out["auto_mode"] == "allow"
    assert out["p_risky"] == 0.12 and out["confidence"] == 0.81
    assert out["latency_ms"] >= 0
    assert out["cost_usd"] == pytest.approx(1000 * 0.042 / 1e6)


def test_missing_key_fails_clearly(monkeypatch):
    from gold import require_keys

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Missing TYPESAFE_API_KEY"):
        require_keys("TYPESAFE_API_KEY")
