"""A smaller prompt can cost more after losing cache or paying for summaries."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from agent_core.context import compact_context
from agent_core.conversation_state import ConversationStore
from agent_core.costs import compare_usage, price_usage, totals_from_state


PRICES = json.loads((Path(__file__).parent / "evals" / "deepseek_flash_prices.json").read_text(encoding="utf-8"))


class BillingTests(unittest.TestCase):
    def test_half_as_many_inputs_can_cost_more_after_cache_loss(self):
        before = {"input_tokens": 100000, "cached_tokens": 90000, "output_tokens": 5000}
        after = {"input_tokens": 50000, "cached_tokens": 10000, "output_tokens": 6000}
        result = compare_usage(before, after, PRICES, "off_peak")
        self.assertEqual(result["baseline"]["cost"]["total"], "0.031800000000")
        self.assertEqual(result["compressed"]["cost"]["total"], "0.064200000000")
        self.assertEqual(result["input_tokens_saved"], 50000)
        self.assertLess(float(result["cost_saved"]), 0)
        self.assertEqual(price_usage(before, PRICES, "peak")["total"], "0.063600000000")

    def test_unknown_cache_yields_bounds_instead_of_assuming_zero(self):
        cost = price_usage({"input_tokens": 1000000, "output_tokens": 1000000}, PRICES, "off_peak")
        self.assertFalse(cost["exact"])
        self.assertIsNone(cost["total"])
        self.assertEqual(cost["lower"], "4.020000000000")
        self.assertEqual(cost["upper"], "5.000000000000")

    def test_summary_input_and_output_are_included_once(self):
        totals = totals_from_state({"model_requests": [
            {"purpose": "conversation", "input_tokens": 1000, "cached_tokens": 800, "output_tokens": 100},
            {"purpose": "summary", "input_tokens": 2000, "cached_tokens": 100, "output_tokens": 200},
        ]})
        self.assertEqual(totals["input_tokens"], 3000)
        self.assertEqual(totals["output_tokens"], 300)
        self.assertEqual(totals["by_purpose"]["summary"]["input_tokens"], 2000)
        self.assertEqual(price_usage(totals, PRICES, "off_peak")["total"], "0.003318000000")

    def test_partial_observation_and_invalid_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            totals_from_state({"model_requests": [{"purpose": "conversation", "input_tokens": None, "output_tokens": 10}]})
        for cached in (-1, 11, True):
            with self.assertRaises(ValueError):
                price_usage({"input_tokens": 10, "output_tokens": 1, "cached_tokens": cached}, PRICES, "off_peak")
        bad = copy.deepcopy(PRICES)
        bad["profiles"]["off_peak"]["output"] = "NaN"
        with self.assertRaises(ValueError):
            price_usage({"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, bad, "off_peak")
        with self.assertRaises(ValueError):
            compare_usage({"model": "different-model", "input_tokens": 1, "output_tokens": 1}, {"input_tokens": 1, "output_tokens": 1}, PRICES, "off_peak")


class NoCompressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_none_mode_keeps_context_even_when_threshold_is_exceeded(self):
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "state.json")
            state = store.start("deepseek-flash", Path(directory) / "skills.json", 1, compact_mode="none")
            state["context"] = [{"role": "user", "content": "large context " * 1000}]
            state["history"] = copy.deepcopy(state["context"])
            original = copy.deepcopy(state)
            client = AsyncMock()
            self.assertFalse(await compact_context(client, state, store, force=True))
            self.assertEqual(state, original)
            client.responses.create.assert_not_awaited()
            client.responses.compact.assert_not_awaited()
            store.save(state)
            self.assertEqual(store.load()["compact_mode"], "none")
