"""Deterministic, evidence-based graders for saved and live conversations."""

import json

from .metrics import summarize_metrics


def evaluate_state(state, expectations=None):
    expectations = expectations or {}
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    calls = state.get("tool_calls", [])
    history = state.get("history", [])
    ids = [c["call_id"] for c in calls]
    check("unique_tool_ids", len(ids) == len(set(ids)), ids)
    outputs = [i.get("call_id") for i in history if i.get("type") == "function_call_output"]
    completed = [c["call_id"] for c in calls if c.get("completed")]
    check("tool_output_pairing", sorted(outputs) == sorted(completed), {"outputs": outputs, "completed": completed})
    known_paths = set()
    ordered = True
    for call in calls:
        if call["name"] == "find_text" and call.get("status") == "completed":
            paths = json.loads(call["arguments"]).get("paths", [])
            ordered &= bool(paths) and set(paths).issubset(known_paths)
        if call["name"] == "find_file" and call.get("status") == "completed":
            known_paths.update(call["result"]["data"])
    check("file_search_order", ordered, "Successful text searches must use previously discovered paths")
    if "turns" in state:
        check("turn_count", state["turn_count"] == sum(t["kind"] == "user" for t in state["turns"]), state["turn_count"])
    if "status" in expectations:
        check("status", state.get("status") == expectations["status"], state.get("status"))
    if "min_compactions" in expectations:
        check("compactions", len(state.get("compactions", [])) >= expectations["min_compactions"], len(state.get("compactions", [])))
    for expected in expectations.get("turns", []):
        number = expected["number"]
        turn = next((t for t in state.get("turns", []) if t["number"] == number), None)
        turn_steps = {s["number"] for s in state.get("steps", []) if s.get("turn") == number}
        turn_calls = [c for c in calls if c["step"] in turn_steps]
        names = [c["name"] for c in turn_calls]
        check(f"turn_{number}_completed", turn is not None and turn["status"] == "completed", names)
        if "tools" in expected:
            check(f"turn_{number}_tools", names == expected["tools"], names)
        for name in expected.get("forbidden_tools", []):
            check(f"turn_{number}_forbid_{name}", name not in names, names)
        if expected.get("tools_ok", True):
            check(f"turn_{number}_tools_ok", all(c.get("result", {}).get("ok") is True for c in turn_calls), names)
        for text in expected.get("reply_contains", []):
            check(f"turn_{number}_reply_{text}", turn is not None and text.casefold() in turn["reply"].casefold(), text)
        results = json.dumps([c.get("result") for c in turn_calls], ensure_ascii=False)
        for text in expected.get("result_contains", []):
            check(f"turn_{number}_result_{text}", text in results, text)
    passed = sum(c["passed"] for c in checks)
    return {"passed": passed == len(checks), "score": passed / len(checks), "checks": checks,
            "metrics": summarize_metrics(state),
            "scope": "Rule and trajectory checks; answer quality requires task expectations or human review"}


def validate_dataset(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a nonempty JSON list")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"id", "messages", "expect", "compact_threshold", "compact_mode"}:
            raise ValueError("Invalid case or unknown case field")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in names:
            raise ValueError("Case ids must be nonempty and unique")
        names.add(identifier)
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, str) and m.strip() for m in messages):
            raise ValueError("messages must contain nonempty strings")
        expected = case.get("expect")
        if not isinstance(expected, dict) or set(expected) - {"status", "turns", "min_compactions"}:
            raise ValueError("Each case needs explicit expect rules; unknown rules are rejected")
        if not expected:
            raise ValueError("Empty expectations would only measure structural consistency")
        if "status" in expected and not isinstance(expected["status"], str):
            raise ValueError("Expected status must be a string")
        if "min_compactions" in expected and (type(expected["min_compactions"]) is not int or expected["min_compactions"] < 0):
            raise ValueError("min_compactions must be a nonnegative integer")
        if not isinstance(expected.get("turns", []), list):
            raise ValueError("Expected turns must be a list")
        for turn in expected.get("turns", []):
            if not isinstance(turn, dict) or set(turn) - {"number", "tools", "forbidden_tools", "tools_ok", "reply_contains", "result_contains"}:
                raise ValueError("Unknown turn expectation")
            if "tools_ok" in turn and type(turn["tools_ok"]) is not bool:
                raise ValueError("tools_ok must be a boolean")
            if type(turn.get("number")) is not int or not 0 <= turn["number"] <= 10:
                raise ValueError("Expected turn number must be between 0 and 10")
            for key in ("tools", "forbidden_tools", "reply_contains", "result_contains"):
                if key in turn and (not isinstance(turn[key], list) or not all(isinstance(x, str) for x in turn[key])):
                    raise ValueError(f"{key} must be a list of strings")
        if type(case.get("compact_threshold", 12000)) is not int or case.get("compact_threshold", 12000) < 1:
            raise ValueError("compact_threshold must be a positive integer")
        if case.get("compact_mode", "local") not in {"none", "local", "summary", "standalone"}:
            raise ValueError("Invalid compact_mode")
    return cases
