"""Evaluation must distinguish unknown cache usage, failed tools and retries."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_core.evaluation import evaluate_state, validate_dataset
from agent_core.metrics import measured_request, summarize_metrics, usage_metrics


class MetricsTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_usage_and_missing_counters(self):
        self.assertEqual(usage_metrics({})["cached_tokens"], None)
        self.assertEqual(usage_metrics({"usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}}})["cached_tokens"], 0)
        self.assertEqual(usage_metrics({"usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60}})["cache_hit_ratio"], .6)
        self.assertEqual(usage_metrics(SimpleNamespace(usage=SimpleNamespace(input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=20))))["cached_tokens"], 20)
        self.assertIsNone(usage_metrics({"usage": {"input_tokens": 2, "input_tokens_details": {"cached_tokens": 3}}})["cached_tokens"])

    def test_weighted_hit_rate_excludes_unknown_usage(self):
        state = {"model_requests": [
            {"purpose": "conversation", "input_tokens": 100, "cached_tokens": 90},
            {"purpose": "summary", "input_tokens": 900, "cached_tokens": 0},
            {"purpose": "conversation", "input_tokens": 10000, "cached_tokens": None},
        ]}
        metrics = summarize_metrics(state)
        self.assertEqual(metrics["cache_hit_ratio"], .09)
        self.assertEqual(metrics["cache_observed_requests"], 2)
        self.assertEqual(metrics["by_purpose"]["conversation"]["cache_hit_ratio"], .9)
        self.assertIsNone(summarize_metrics({})["cache_hit_ratio"])

    async def test_retry_and_cancel_preserve_separate_attempts(self):
        state = {}
        response = SimpleNamespace(status="completed", usage={"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}})
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=[RuntimeError("fail"), response, asyncio.CancelledError()])))
        kwargs = dict(purpose="conversation", model="test", instructions="fixed", tools=[], input=[])
        with self.assertRaises(RuntimeError):
            await measured_request(client, state, **kwargs)
        await measured_request(client, state, **kwargs)
        with self.assertRaises(asyncio.CancelledError):
            await measured_request(client, state, **kwargs)
        records = state["model_requests"]
        self.assertEqual([r["status"] for r in records], ["error", "completed", "interrupted"])
        self.assertTrue(all(r["elapsed_seconds"] >= 0 for r in records))
        self.assertEqual(records[0]["prefix_hash"], records[1]["prefix_hash"])


class EvaluationTests(unittest.TestCase):
    def state(self):
        return {"status": "waiting_user", "turn_count": 1,
                "turns": [{"number": 1, "kind": "user", "status": "completed", "reply": "found"}],
                "steps": [{"number": 1, "turn": 1}], "history": [{"type": "function_call_output", "call_id": "x"}],
                "tool_calls": [{"call_id": "x", "step": 1, "name": "find_text", "arguments": '{"paths": ["unseen"]}',
                                "status": "completed", "completed": True, "result": {"ok": True, "data": []}}]}

    def test_unseen_paths_and_wrong_tool_fail_even_with_good_reply(self):
        report = evaluate_state(self.state(), {"turns": [{"number": 1, "tools": ["search_saved_skills"], "reply_contains": ["found"]}]})
        self.assertFalse(report["passed"])
        failures = [c["name"] for c in report["checks"] if not c["passed"]]
        self.assertIn("file_search_order", failures)
        self.assertIn("turn_1_tools", failures)

    def test_duplicate_outputs_and_failed_tools_fail(self):
        state = self.state()
        state["history"] *= 2
        state["tool_calls"][0].update(name="search_saved_skills", status="failed", result={"ok": False})
        report = evaluate_state(state, {"turns": [{"number": 1, "tools": ["search_saved_skills"]}]})
        self.assertFalse(report["passed"])
        self.assertFalse(next(c["passed"] for c in report["checks"] if c["name"] == "tool_output_pairing"))
        self.assertFalse(next(c["passed"] for c in report["checks"] if c["name"] == "turn_1_tools_ok"))

    def test_missing_expected_turn_and_unknown_rules_fail(self):
        self.assertFalse(evaluate_state(self.state(), {"turns": [{"number": 2} ]})["passed"])
        with self.assertRaises(ValueError):
            validate_dataset([{"id": "x", "messages": ["hello"], "expect": {"statsu": "closed"}}])
        with self.assertRaises(ValueError):
            validate_dataset([{"id": "x", "messages": ["hello"], "expect": {}}])
