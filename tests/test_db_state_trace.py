from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from evotau.db_state_trace import (
    DBStateTraceError,
    capture_mutation_events,
    write_trace_once,
)
from evotau.tau_provenance import canonical_json
from evotau.web.view_models import db_state_trace_view


class FakeMessage:
    def __init__(
        self, role, *, calls=(), turn_idx=None, tool_id=None, content="", error=False
    ):
        self.role = role
        self.tool_calls = list(calls)
        self.turn_idx = turn_idx
        self.id = tool_id
        self.content = content
        self.error = error
        self.tool_messages = None

    def is_tool_call(self):
        return bool(self.tool_calls)


class FakeEnvironment:
    def __init__(self, *, mutating=True):
        self.tools = SimpleNamespace(
            db={
                "orders": {
                    "#W1": {
                        "status": "delivered",
                        "return_items": None,
                        "return_payment_method_id": None,
                    },
                },
                "products": {},
                "users": {"u1": {"email": "before@example.test"}},
            }
        )
        self.mutating = mutating
        self.provider_calls = 0

    def set_state(self, **_kwargs):
        return None

    def _has_tool(self, name):
        return name in {"return", "set_user", "lookup"}

    def _is_mutating_tool(self, name):
        return self.mutating and name in {"return", "set_user"}

    def get_db_hash(self):
        value = canonical_json(self.tools.db).encode()
        return hashlib.sha256(value).hexdigest()

    def get_response(self, call):
        if call.name == "return":
            row = self.tools.db["orders"][call.arguments["order_id"]]
            row["status"] = "return requested"
            row["return_items"] = sorted(call.arguments["item_ids"])
            row["return_payment_method_id"] = call.arguments["payment_method_id"]
        elif call.name == "set_user":
            self.tools.db["users"][call.arguments["user_id"]]["email"] = call.arguments[
                "email"
            ]
        return SimpleNamespace(
            role="tool",
            id=call.id,
            error=False,
            content=json.dumps({"ok": call.id}, sort_keys=True),
        )


def _call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, name=name, arguments=arguments, requestor="assistant"
    )


def _tool_result(call_id):
    return FakeMessage(
        "tool",
        tool_id=call_id,
        content=json.dumps({"ok": call_id}, sort_keys=True),
    )


def _fixture_messages():
    first = _call(
        "write-1",
        "return",
        {
            "order_id": "#W1",
            "item_ids": ["item-b", "item-a"],
            "payment_method_id": "card-1",
        },
    )
    read = _call("read-1", "lookup", {"order_id": "#W1"})
    second = _call(
        "write-2",
        "set_user",
        {
            "user_id": "u1",
            "email": "after@example.test",
        },
    )
    return [
        FakeMessage("assistant", calls=[first], turn_idx=4),
        _tool_result("write-1"),
        FakeMessage("assistant", calls=[read], turn_idx=5),
        _tool_result("read-1"),
        FakeMessage("assistant", calls=[second], turn_idx=8),
        _tool_result("write-2"),
    ]


def test_write_calls_emit_ordered_sorted_business_field_changes_without_provider_calls():
    environment = FakeEnvironment()
    result = capture_mutation_events(environment, _fixture_messages())

    assert result["write_tool_call_count"] == 2
    assert [event["tool_call_id"] for event in result["events"]] == [
        "write-1",
        "write-2",
    ]
    assert result["events"][0]["turn_idx"] == 4
    assert [change["field_path"] for change in result["events"][0]["changes"]] == [
        "return_items",
        "return_payment_method_id",
        "status",
    ]
    assert result["events"][0]["changes"][2]["before"] == "delivered"
    assert result["events"][0]["changes"][2]["after"] == "return requested"
    assert result["events"][1]["changes"][0]["field_path"] == "email"
    assert environment.provider_calls == 0


def test_read_only_tool_produces_no_empty_mutation_event():
    environment = FakeEnvironment(mutating=False)
    result = capture_mutation_events(environment, _fixture_messages()[2:4])
    assert result["write_tool_call_count"] == 0
    assert result["events"] == []


def test_replay_is_deterministic_and_does_not_mutate_source_messages():
    messages = _fixture_messages()
    before = deepcopy(messages)
    first = capture_mutation_events(FakeEnvironment(), messages)
    second = capture_mutation_events(FakeEnvironment(), messages)
    assert first == second
    assert [
        (row.role, row.id, row.content) for row in messages if row.role == "tool"
    ] == [(row.role, row.id, row.content) for row in before if row.role == "tool"]


def test_parallel_episode_replays_use_independent_mutable_state():
    messages = _fixture_messages()

    def replay(_index):
        environment = FakeEnvironment()
        result = capture_mutation_events(environment, messages)
        return result, environment.tools.db

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(replay, range(4)))
    assert all(result == rows[0][0] for result, _state in rows)
    assert len({id(state) for _result, state in rows}) == 4
    assert all(
        state["orders"]["#W1"]["status"] == "return requested"
        for _result, state in rows
    )


def test_tool_replay_mismatch_fails_closed():
    messages = _fixture_messages()
    messages[1].content = '{"ok":"unexpected"}'
    with pytest.raises(DBStateTraceError, match="tool_replay_mismatch"):
        capture_mutation_events(FakeEnvironment(), messages)


def test_trace_sidecar_write_is_atomic_and_never_overwrites(tmp_path):
    path = tmp_path / "db-state-trace.json"
    write_trace_once(path, {"status": "complete", "events": []})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_trace_once(path, {"status": "replacement", "events": []})
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_actual_db_mismatch_is_displayed_without_becoming_service_failure():
    view = db_state_trace_view(
        {
            "status": "complete",
            "events": [],
            "summary": {
                "db_match": False,
                "write_tool_call_count": 1,
                "mutation_event_count": 1,
                "field_change_count": 2,
                "final_comparison": {
                    "db_match": False,
                    "gold_state_available": True,
                    "differences": [
                        {
                            "entity_type": "order",
                            "entity_id": "#W1",
                            "field_path": "status",
                            "actual": "delivered",
                            "expected": "return requested",
                        }
                    ],
                },
            },
        }
    )
    assert view["available"] is True
    assert view["summary"]["db_match"] is False
    difference = view["summary"]["final_comparison"]["differences"][0]
    assert difference["simple_entity_label"] == "Order"
    assert difference["simple_field_label"] == "Status"
    assert difference["actual"] == "delivered"
    assert difference["expected"] == "return requested"
