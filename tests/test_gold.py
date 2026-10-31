import json

import pytest

from costs import LABELS
from gold import load_gold, split, to_langsmith_example


def test_committed_gold_set_is_valid():
    rows = load_gold()
    assert 150 <= len(rows) <= 300
    assert {r["label"] for r in rows} == set(LABELS)
    assert {r["split"] for r in rows} == {"dev", "heldout"}
    # The held-out split has every class, and the hard cases.
    heldout = split(rows, "heldout")
    assert {r["label"] for r in heldout} == set(LABELS)
    assert any("injection" in r["tags"] for r in heldout)
    assert any("hard" in r["tags"] for r in heldout)


def test_every_row_has_what_the_gate_needs():
    for r in load_gold():
        inputs = r["inputs"]
        assert inputs["messages"][0]["role"] == "user"
        assert {"id", "name", "args"} <= set(inputs["tool_call"])
        assert inputs["tool_description"]


def _write(tmp_path, rows):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _row(id, group, split_name, label="allow"):
    return {"id": id, "group": group, "split": split_name, "label": label, "tags": [], "note": "",
            "inputs": {"messages": [], "tool_call": {}, "tool_description": ""}}


def test_group_in_both_splits_is_rejected(tmp_path):
    path = _write(tmp_path, [_row("a", "g", "dev"), _row("b", "g", "heldout")])
    with pytest.raises(ValueError, match="both splits"):
        load_gold(path)


def test_duplicate_ids_and_bad_labels_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(_write(tmp_path, [_row("a", "g", "dev"), _row("a", "h", "dev")]))
    with pytest.raises(ValueError, match="bad label"):
        load_gold(_write(tmp_path, [_row("a", "g", "dev", label="maybe")]))


def test_split_and_langsmith_example():
    rows = [_row("a", "g", "dev"), _row("b", "h", "heldout", "block")]
    assert [r["id"] for r in split(rows, "heldout")] == ["b"]
    assert len(split(rows, "all")) == 2
    ex = to_langsmith_example(rows[1])
    assert ex["outputs"] == {"label": "block"}
    assert ex["split"] == "heldout" and ex["metadata"]["id"] == "b"
