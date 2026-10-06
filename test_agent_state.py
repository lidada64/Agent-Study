"""Offline integration tests for search progression and durable recovery."""

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, Tool
from openai.types.responses import ResponseFunctionToolCall

from agent_core.runner import run_agent
from agent_core.state import StateStore


def response(*calls, text=""):
    return SimpleNamespace(output=list(calls), output_text=text)


def call(name, arguments, call_id):
    return ResponseFunctionToolCall(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id,
    )


def model_with(*responses):
    return SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=list(responses))))


class PersistentAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "state.json"
        self.store = StateStore(self.path)
        self.file = self.root / "example.txt"
        self.file.write_text("first\nlidada\nlast lidada\n", encoding="utf-8")
        patcher = patch("agent_core.local_tools.ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[])))
        self.tools = [Tool(name="search_skills", inputSchema={})]

    async def run_task(self, model, prompt="find lidada", max_steps=8, **kwargs):
        return await run_agent(model, self.client, self.tools, prompt, "test", max_steps,
                               state_path=self.path, **kwargs)

    async def test_keyword_tool_is_available_from_first_turn(self):
        initial_tools = None

        async def respond(**kwargs):
            nonlocal initial_tools
            if initial_tools is None:
                initial_tools = {tool["name"] for tool in kwargs["tools"]}
                return response(call("find_file", {"name": "example.txt"}, "files"))
            if "find_text" not in initial_tools:
                return response(text="Cannot search: find_text is unavailable")
            if not any(item.get("call_id") == "text" for item in kwargs["input"]):
                return response(call("find_text", {"paths": [str(self.file)], "text": "lidada"}, "text"))
            return response(text="Found two matches")

        model = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=respond)))
        await self.run_task(model)
        state = self.store.load()
        self.assertEqual([c["name"] for c in state["tool_calls"]], ["find_file", "find_text"])
        self.assertEqual([hit["line_number"] for hit in state["tool_calls"][1]["result"]["data"]], [2, 3])

    async def test_real_keyword_search_is_multi_step_and_each_transition_is_saved(self):
        snapshots = []
        save = StateStore.save

        def capture(store, state):
            save(store, state)
            snapshots.append(json.loads(store.path.read_text(encoding="utf-8")))

        model = model_with(
            response(call("find_file", {"name": "example.txt"}, "files")),
            response(call("find_text", {"paths": [str(self.file)], "text": "lidada"}, "text")),
            response(text="Found two matches"),
        )
        with patch.object(StateStore, "save", capture):
            self.assertEqual(await self.run_task(model), ("completed", "Found two matches"))
        state = self.store.load()
        self.assertEqual(state["phase"], "done")
        self.assertEqual(len(state["steps"]), 3)
        self.assertEqual([hit["line_number"] for hit in state["tool_calls"][1]["result"]["data"]], [2, 3])
        for identifier in ("files", "text"):
            statuses = [c["status"] for s in snapshots for c in s["tool_calls"] if c["call_id"] == identifier]
            self.assertIn("pending", statuses)
            self.assertIn("running", statuses)
            self.assertIn("completed", statuses)
        first_request, second_request, _ = model.responses.create.call_args_list
        self.assertIn("find_text", [tool["name"] for tool in first_request.kwargs["tools"]])
        self.assertIn("find_text", [tool["name"] for tool in second_request.kwargs["tools"]])
        self.assertTrue(all(c["called"] and c["completed"] and not c["running"] for c in state["tool_calls"]))

    async def test_resume_after_step_limit_does_not_repeat_file_discovery(self):
        first = model_with(response(call("find_file", {"name": "example.txt"}, "files")))
        self.assertEqual(await self.run_task(first, max_steps=1), ("max_range", ""))
        original_id = self.store.load()["task_id"]
        second = model_with(
            response(call("find_text", {"paths": [str(self.file)], "text": "lidada"}, "text")),
            response(text="done"),
        )
        self.assertEqual(await self.run_task(second, prompt=None, resume=True), ("completed", "done"))
        state = self.store.load()
        self.assertEqual(state["task_id"], original_id)
        self.assertEqual(state["tool_calls"][0]["attempts"], 1)
        self.assertEqual(second.responses.create.call_args_list[0].kwargs["input"][-1]["call_id"], "files")

    async def test_cancel_mid_batch_only_unfinished_tool_is_retried_before_model(self):
        entered = asyncio.Event()

        async def interrupted_tool(name, arguments):
            state = self.store.load()
            self.assertTrue(state["tool_calls"][1]["running"])
            entered.set()
            await asyncio.Event().wait()

        self.client.call_tool.side_effect = interrupted_tool
        first = model_with(response(
            call("find_file", {"name": "example.txt"}, "files"),
            call("skills_search_skills", {"query": "python"}, "skills"),
        ))
        task = asyncio.create_task(self.run_task(first))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        state = self.store.load()
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual([c["status"] for c in state["tool_calls"]], ["completed", "interrupted"])
        self.assertFalse(state["tool_calls"][1]["running"])
        self.client.call_tool.side_effect = None
        resumed = model_with(response(text="done"))
        self.assertEqual(await self.run_task(resumed, prompt=None, resume=True), ("completed", "done"))
        state = self.store.load()
        self.assertEqual([c["attempts"] for c in state["tool_calls"]], [1, 2])
        outputs = [i for i in resumed.responses.create.call_args.kwargs["input"] if i.get("type") == "function_call_output"]
        self.assertEqual([i["call_id"] for i in outputs], ["files", "skills"])

    async def test_hard_exit_snapshot_with_running_tool_recovers(self):
        # Simulate abrupt process termination by failing the checkpoint observer
        # immediately after the on-disk running transition.
        class ProcessKilled(BaseException):
            pass

        save = StateStore.save

        def kill_after_running(store, state):
            save(store, state)
            if any(c["running"] for c in state["tool_calls"]):
                raise ProcessKilled()

        first = model_with(response(call("skills_search_skills", {"query": "python"}, "skills")))
        with patch.object(StateStore, "save", kill_after_running):
            with self.assertRaises(ProcessKilled):
                await self.run_task(first)
        self.assertTrue(self.store.load()["tool_calls"][0]["running"])
        second = model_with(response(text="recovered"))
        self.assertEqual(await self.run_task(second, prompt=None, resume=True), ("completed", "recovered"))
        self.client.call_tool.assert_awaited_once()
        self.assertEqual(self.store.load()["tool_calls"][0]["attempts"], 2)

    async def test_model_failure_can_resume_with_saved_tool_outputs(self):
        first = model_with(
            response(call("find_file", {"name": "example.txt"}, "files")), RuntimeError("offline"),
        )
        self.assertEqual(await self.run_task(first), ("model_error", ""))
        self.assertEqual(self.store.load()["last_error"]["message"], "offline")
        second = model_with(response(text="done"))
        self.assertEqual(await self.run_task(second, prompt=None, resume=True), ("completed", "done"))
        state = self.store.load()
        self.assertEqual(len(state["steps"]), 2)
        self.assertEqual(state["tool_calls"][0]["attempts"], 1)

    async def test_cancel_model_request_retains_previous_progress(self):
        first = model_with(response(call("find_file", {"name": "example.txt"}, "files")), asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.run_task(first)
        self.assertEqual(self.store.load()["steps"][-1]["status"], "interrupted")
        await self.run_task(model_with(response(text="done")), prompt=None, resume=True)
        self.assertEqual(len(self.store.load()["steps"]), 2)

    async def test_error_output_is_persisted_and_not_reexecuted_on_resume(self):
        first = model_with(response(call("find_text", {"paths": [str(self.file)], "text": "lidada"}, "bad")))
        await self.run_task(first, max_steps=1)
        failed = self.store.load()["tool_calls"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["completed"])
        second = model_with(response(text="tool failed"))
        await self.run_task(second, prompt=None, resume=True)
        self.assertEqual(self.store.load()["tool_calls"][0]["attempts"], 1)
        self.assertFalse(json.loads(second.responses.create.call_args.kwargs["input"][-1]["output"])["ok"])

    async def test_completed_resume_returns_saved_answer_without_model_request(self):
        await self.run_task(model_with(response(text="done")))
        second = model_with()
        self.assertEqual(await self.run_task(second, prompt=None, resume=True), ("completed", "done"))
        second.responses.create.assert_not_awaited()

    async def test_different_prompt_cannot_overwrite_unfinished_task_without_new(self):
        await self.run_task(model_with(response(call("find_file", {"name": "example.txt"}, "files"))), max_steps=1)
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "--new"):
            await self.run_task(model_with(), prompt="different")
        self.assertEqual(self.path.read_bytes(), original)
        await self.run_task(model_with(response(text="new task")), prompt="different", new_task=True)
        self.assertEqual(self.store.load()["prompt"], "different")

    def test_atomic_write_failure_keeps_previous_valid_checkpoint(self):
        state = self.store.start("find lidada", "test")
        original = self.path.read_bytes()
        state["status"] = "running"
        with patch("agent_core.state.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.store.save(state)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_corrupt_checkpoint_is_never_silently_replaced(self):
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.start("find lidada", "test")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{broken")

    def test_resume_requires_existing_state(self):
        with self.assertRaisesRegex(ValueError, "No saved task"):
            self.store.start(None, "test", resume=True)


if __name__ == "__main__":
    unittest.main()
