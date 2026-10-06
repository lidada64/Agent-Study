"""Offline behavior tests for conversations, downloads, compaction and recovery."""

import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from mcp import StdioServerParameters
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI

from agent_core.config import SKILL_SYSTEM_PROMPT
from agent_core.context import compact_context
from agent_core.conversation import SkillConversation, interactive_loop
from agent_core.conversation_state import ConversationStore
from agent_core.mcp_connection import connect_mcp
from agent_core.mcp_tools import discover_skill_tools
from agent_core.skill_library import SkillLibrary
from agent_core.state import StateStore
from test_agent_state import call


def answer(text, *, status="completed"):
    return SimpleNamespace(status=status, output_text=text, output=[{
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }])


def tool_response(*calls):
    return SimpleNamespace(status="completed", output_text="", output=list(calls))


def model_with(*responses):
    return SimpleNamespace(responses=SimpleNamespace(
        create=AsyncMock(side_effect=list(responses)), compact=AsyncMock(),
    ))


DOCUMENT = "# Python Testing\nSource: fixture\n\n---\n\n---\nname: python-testing\n---\nUse PyTest to test Python code.\n"


class SkillConversationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = ConversationStore(self.root / "conversation.json")
        self.library = SkillLibrary(self.root / "skills.json")
        self.state = self.store.start("test", self.library.path, 12000, compact_mode="standalone")
        self.client = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text=DOCUMENT)],
        )))
        from mcp.types import Tool
        self.discovered = [Tool(name="search_skills", inputSchema={}), Tool(name="get_skill", inputSchema={})]

    def conversation(self, model, state=None, **kwargs):
        return SkillConversation(model, self.client, self.discovered, state or self.state, self.store, self.library, **kwargs)

    async def test_intro_then_ten_user_turns_and_no_eleventh_request(self):
        model = model_with(answer("可以检索、下载和查询 Skill"), *(answer(f"回复 {i}") for i in range(10)))
        inputs = iter([f"问题 {i}" for i in range(12)])
        output = []
        conversation = self.conversation(model)
        result = await interactive_loop(conversation, read_input=lambda _: next(inputs), write_output=output.append)
        self.assertEqual(result, 0)
        self.assertEqual(self.state["turn_count"], 10)
        self.assertEqual(self.state["status"], "closed")
        self.assertEqual(self.state["close_reason"], "turn_limit")
        self.assertEqual(model.responses.create.await_count, 11)
        self.assertEqual(model.responses.create.call_args_list[0].kwargs["tools"], [])
        self.assertEqual(model.responses.create.call_args_list[0].kwargs["instructions"], SKILL_SYSTEM_PROMPT)
        self.assertIn("自动退出", output[-1])
        self.assertEqual(self.store.load()["turn_count"], 10)
        self.assertEqual(await conversation.submit("额外问题"), ("closed", ""))

    async def test_blank_input_and_exit_do_not_count_as_user_turns(self):
        model = model_with(answer("介绍"), answer("回复"))
        inputs = iter([" ", "查询 Python", "退出！"])
        await interactive_loop(self.conversation(model), read_input=lambda _: next(inputs), write_output=lambda _: None)
        self.assertEqual(self.state["turn_count"], 1)
        self.assertEqual(self.state["close_reason"], "user_exit")
        self.assertEqual(model.responses.create.await_count, 2)

    async def test_eof_closes_and_saves_conversation(self):
        await interactive_loop(self.conversation(model_with(answer("介绍"))), read_input=lambda _: (_ for _ in ()).throw(EOFError()), write_output=lambda _: None)
        self.assertEqual(self.store.load()["close_reason"], "eof")

    async def test_multiple_tools_are_one_user_turn_and_download_uses_mcp_content(self):
        model = model_with(answer("介绍"),
            tool_response(call("skills_search_skills", {"query": "python"}, "search")),
            tool_response(call("download_skill", {"id": "python-testing"}, "save")),
            answer("已录入"),
            tool_response(call("search_saved_skills", {"query": "pytest"}, "local")),
            answer("找到已下载的 Skill"))
        conversation = self.conversation(model)
        await conversation.introduce()
        self.assertEqual(await conversation.submit("搜索并下载 Python testing"), ("waiting_user", "已录入"))
        self.assertEqual(self.state["turn_count"], 1)
        self.assertEqual(self.library.read("python-testing")["data"]["content"], DOCUMENT)
        await conversation.submit("查询本地 pytest")
        self.assertEqual(self.state["turn_count"], 2)
        local = self.state["tool_calls"][-1]["result"]
        self.assertEqual(local["data"][0]["id"], "python-testing")
        self.assertEqual(self.state["history"], self.state["context"])
        second_user_input = model.responses.create.call_args_list[4].kwargs["input"]
        self.assertIn({"role": "user", "content": "搜索并下载 Python testing"}, second_user_input)

    async def test_library_is_searchable_after_restart_and_download_is_idempotent(self):
        await self.library.download("python-testing", self.client, {"skills_get_skill": "get_skill"})
        reloaded = SkillLibrary(self.library.path)
        result = await reloaded.download("python-testing", self.client, {"skills_get_skill": "get_skill"})
        self.assertTrue(result["already_saved"])
        self.client.call_tool.assert_awaited_once_with("get_skill", arguments={"id": "python-testing"})
        self.assertEqual(reloaded.search("PYTHON pytest")["count"], 1)
        self.assertEqual(reloaded.search("missing")["count"], 0)
        self.assertEqual(reloaded.search("")["count"], 1)

    async def test_mcp_error_or_empty_content_is_not_saved(self):
        self.client.call_tool.return_value = CallToolResult(content=[], isError=True)
        self.assertFalse((await self.library.download("bad", self.client, {"skills_get_skill": "get_skill"}))["ok"])
        self.assertFalse(self.library.path.exists())
        self.client.call_tool.return_value = CallToolResult(content=[])
        with self.assertRaisesRegex(ValueError, "no Skill document"):
            await self.library.download("bad", self.client, {"skills_get_skill": "get_skill"})
        self.assertFalse(self.library.path.exists())

    async def test_compaction_preserves_entire_window_history_ledger_and_system_prompt(self):
        model = model_with(answer("介绍"), tool_response(call("download_skill", {"id": "python-testing"}, "save")), answer("保存成功"))
        conversation = self.conversation(model)
        await conversation.introduce()
        self.state["compact_threshold"] = 1
        retained = {"role": "user", "content": "保留用户需求"}
        opaque = {"type": "compaction", "encrypted_content": "opaque-do-not-edit", "id": "cmp_test"}
        window = [retained, opaque]
        model.responses.compact.return_value = SimpleNamespace(id="compact_test", output=window)
        await conversation.submit("下载 python-testing")
        self.assertEqual(model.responses.create.call_args.kwargs["input"], window)
        self.assertEqual(model.responses.compact.call_args.kwargs["instructions"], SKILL_SYSTEM_PROMPT)
        compact_input = model.responses.compact.call_args.kwargs["input"]
        self.assertEqual(compact_input[-1]["type"], "function_call_output")
        self.assertEqual(compact_input[-1]["call_id"], "save")
        self.assertTrue(self.state["tool_calls"][0]["completed"])
        self.assertTrue(any(item.get("call_id") == "save" for item in self.state["history"]))
        self.assertEqual(self.state["context"][:2], window)
        reloaded = self.store.load()
        self.assertEqual(reloaded["context"][:2], window)
        self.assertEqual(len(reloaded["compactions"]), 2)

    async def test_compact_failure_keeps_original_context_and_can_resume_same_turn(self):
        first = model_with(answer("介绍"))
        conversation = self.conversation(first)
        await conversation.introduce()
        self.state["compact_threshold"] = 1
        first.responses.compact.side_effect = RuntimeError("compact endpoint unavailable")
        self.assertEqual(await conversation.submit("查询"), ("compact_error", ""))
        saved = self.store.load()
        self.assertEqual(saved["context"], saved["history"])
        second = model_with(answer("继续回复"))
        second.responses.compact.return_value = SimpleNamespace(output=[{"type": "compaction", "encrypted_content": "opaque"}])
        resumed = self.conversation(second, saved)
        await resumed.introduce()
        self.assertEqual(resumed.state["turn_count"], 1)
        self.assertEqual(resumed.state["turns"][-1]["reply"], "继续回复")
        self.assertEqual(first.responses.create.await_count, 1)

    async def test_compaction_save_failure_does_not_replace_in_memory_context(self):
        model = model_with(answer("介绍"))
        await self.conversation(model).introduce()
        original = copy.deepcopy(self.state)
        model.responses.compact.return_value = SimpleNamespace(output=[{"type": "compaction", "encrypted_content": "opaque"}])
        with patch.object(self.store, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await compact_context(model, self.state, self.store, force=True)
        self.assertEqual(self.state, original)

    async def test_incomplete_or_empty_response_does_not_finish_turn(self):
        for response in (answer("partial", status="incomplete"), answer("")):
            with self.subTest(response=response.output_text):
                state = self.store.start("test", self.library.path, 12000, new=True)
                conversation = self.conversation(model_with(answer("介绍"), response), state)
                await conversation.introduce()
                self.assertEqual(await conversation.submit("查询"), ("model_error", ""))
                self.assertEqual(state["turns"][-1]["status"], "running")
                self.assertFalse(any(item.get("content") == response.output[0]["content"] for item in state["history"]))
                await self.conversation(model_with(answer("完整回复")), self.store.load()).resume_turn()
                self.assertEqual(self.store.load()["turn_count"], 1)

    async def test_step_limit_resume_finishes_turn_without_repeating_download(self):
        first = model_with(answer("介绍"), tool_response(call("download_skill", {"id": "python-testing"}, "save")))
        conversation = self.conversation(first, max_steps=1)
        await conversation.introduce()
        self.assertEqual(await conversation.submit("下载"), ("max_steps", ""))
        second = model_with(answer("下载完成"))
        resumed = self.conversation(second, self.store.load())
        await resumed.introduce()
        self.assertEqual(resumed.state["turn_count"], 1)
        self.client.call_tool.assert_awaited_once()
        self.assertEqual(second.responses.create.call_args.kwargs["input"][-1]["call_id"], "save")

    async def test_failed_tenth_turn_resumes_then_closes_without_an_eleventh_turn(self):
        first = model_with(answer("介绍"), *(answer("回复") for _ in range(9)), RuntimeError("offline"))
        conversation = self.conversation(first)
        await conversation.introduce()
        for _ in range(9):
            await conversation.submit("查询")
        self.assertEqual(await conversation.submit("最后一个问题"), ("model_error", ""))
        self.assertEqual(self.state["turn_count"], 10)
        second = model_with(answer("最后一个回复"))
        resumed = self.conversation(second, self.store.load())
        self.assertEqual(await resumed.introduce(), ("closed", "最后一个回复"))
        self.assertEqual(resumed.state["turn_count"], 10)
        self.assertEqual(resumed.state["close_reason"], "turn_limit")
        second.responses.create.assert_awaited_once()

    async def test_pending_tool_calls_prevent_compaction(self):
        model = model_with(answer("介绍"))
        await self.conversation(model).introduce()
        self.state["tool_calls"].append({"completed": False})
        with self.assertRaisesRegex(ValueError, "pending"):
            await compact_context(model, self.state, self.store, force=True)
        model.responses.compact.assert_not_awaited()

    async def test_crash_after_library_write_before_checkpoint_recovers_without_refetch(self):
        class ProcessKilled(BaseException):
            pass

        model = model_with(answer("介绍"), tool_response(call("download_skill", {"id": "python-testing"}, "save")))
        conversation = self.conversation(model)
        await conversation.introduce()
        save = StateStore.save

        def fail_after_download(store, state):
            if any(call["completed"] for call in state["tool_calls"]):
                raise ProcessKilled()
            save(store, state)

        with patch.object(ConversationStore, "save", fail_after_download):
            with self.assertRaises(ProcessKilled):
                await conversation.submit("下载")
        saved = self.store.load()
        self.assertTrue(saved["tool_calls"][0]["running"])
        self.assertTrue(self.library.path.exists())
        second = model_with(answer("已保存"))
        resumed = self.conversation(second, saved)
        await resumed.resume_turn()
        self.client.call_tool.assert_awaited_once()
        self.assertEqual(saved["tool_calls"][0]["attempts"], 2)
        self.assertTrue(saved["tool_calls"][0]["result"]["already_saved"])
        self.assertEqual(sum(item.get("type") == "function_call_output" for item in saved["history"]), 1)

    async def test_cancel_model_preserves_turn_and_resume_does_not_add_another(self):
        first = model_with(answer("介绍"), asyncio.CancelledError())
        conversation = self.conversation(first)
        await conversation.introduce()
        with self.assertRaises(asyncio.CancelledError):
            await conversation.submit("查询")
        self.assertEqual(self.store.load()["status"], "interrupted")
        await self.conversation(model_with(answer("继续")), self.store.load()).resume_turn()
        self.assertEqual(self.store.load()["turn_count"], 1)

    def test_corrupt_library_or_conversation_is_not_silently_replaced(self):
        self.library.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.library.search("python")
        self.assertEqual(self.library.path.read_text(encoding="utf-8"), "{broken")
        del self.state["context"]
        self.store.save(self.state)
        with self.assertRaisesRegex(ValueError, "context"):
            self.store.load()

    async def test_actual_stdio_mcp_download_and_local_query(self):
        parameters = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "test_fixtures" / "skills_server.py")])
        async with connect_mcp(parameters) as client:
            discovered = await discover_skill_tools(client)
            model = model_with(answer("介绍"),
                tool_response(call("skills_search_skills", {"query": "python"}, "search")),
                tool_response(call("download_skill", {"id": "python-testing"}, "save")), answer("已保存"))
            conversation = SkillConversation(model, client, discovered, self.state, self.store, self.library)
            await conversation.introduce()
            await conversation.submit("搜索并下载 Python testing")
        self.assertEqual(self.state["tool_calls"][0]["result"]["ok"], True)
        self.assertEqual(self.library.search("pytest")["count"], 1)
        self.assertIn("Write meaningful pytest tests", self.library.read("python-testing")["data"]["content"])

    async def test_real_openai_sdk_uses_standalone_compact_http_route_and_replays_window(self):
        requests = []
        window = [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "保留"}]}, {"type": "compaction", "id": "cmp_test", "encrypted_content": "opaque"}]

        def handler(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/compact"):
                return httpx.Response(200, json={"id": "resp_compact", "object": "response.compaction", "created_at": 1, "output": window})
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 1,
                "model": "test", "status": "completed", "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [], "output": [{
                    "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "回复", "annotations": []}],
                }],
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            # MockTransport handles every request locally; no credential is needed.
            async with AsyncOpenAI(api_key="", base_url="https://fixture.invalid/v1", http_client=http_client) as model:
                conversation = self.conversation(model)
                await conversation.introduce()
                self.state["compact_threshold"] = 1
                await conversation.submit("查询")
        self.assertEqual([path for path, _ in requests], ["/v1/responses", "/v1/responses/compact", "/v1/responses"])
        self.assertEqual(requests[2][1]["input"], window)
        self.assertFalse(requests[2][1]["store"])
        self.assertEqual(requests[2][1]["instructions"], SKILL_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
