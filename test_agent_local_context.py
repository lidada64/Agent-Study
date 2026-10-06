"""Local memory must reduce input without a remote compression request or lost ledger."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_core.config import SKILL_SYSTEM_PROMPT
from agent_core.context import compact_context, context_size, local_window, read_history
from agent_core.conversation import SkillConversation
from agent_core.conversation_state import ConversationStore
from agent_core.skill_library import SkillLibrary
from test_agent_conversation import answer, model_with, tool_response, DOCUMENT
from test_agent_state import call
from types import SimpleNamespace
from unittest.mock import AsyncMock
from mcp.types import CallToolResult, TextContent, Tool


class LocalContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = ConversationStore(self.root / "conversation.json")
        self.library = SkillLibrary(self.root / "skills.json")
        self.state = self.store.start("deepseek-flash", self.library.path, 4000)
        self.mcp = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text=DOCUMENT)])))
        self.tools = [Tool(name="search_skills", inputSchema={}), Tool(name="get_skill", inputSchema={})]

    def conversation(self, model, state=None):
        return SkillConversation(model, self.mcp, self.tools, state or self.state, self.store, self.library)

    async def test_local_default_reduces_history_keeps_two_recent_turns_and_uses_no_endpoint(self):
        model = model_with(answer("介绍"), answer("旧资料 " * 3000), answer("第二轮回复"), answer("第三轮回复"))
        conversation = self.conversation(model)
        await conversation.introduce()
        await conversation.submit("第一个查询")
        await conversation.submit("第二个查询")
        await conversation.submit("第三个查询")
        model.responses.compact.assert_not_awaited()
        self.assertEqual(model.responses.create.await_count, 4)
        self.assertEqual(self.state["compact_mode"], "local")
        self.assertLess(context_size(self.state["context"]), context_size(self.state["history"]))
        window = model.responses.create.call_args.kwargs["input"]
        memory = json.loads(window[0]["content"].split("\n", 1)[1])
        self.assertTrue(memory["lossy"])
        self.assertTrue(any(record.get("excerpt") == "第一个查询" for record in memory["records"]))
        self.assertIn({"role": "user", "content": "第二个查询"}, window[1:])
        self.assertIn({"role": "user", "content": "第三个查询"}, window[1:])
        self.assertTrue(any(item.get("content") == answer("第二轮回复").output[0]["content"] for item in window[1:]))
        self.assertEqual(self.state["turn_count"], 3)
        self.assertEqual(self.state["compactions"][0]["mode"], "local")
        self.assertEqual(self.store.load()["context"], self.state["context"])
        for request in model.responses.create.call_args_list:
            self.assertFalse(request.kwargs["store"])
            self.assertNotIn("previous_response_id", request.kwargs)
            self.assertEqual(request.kwargs["instructions"], SKILL_SYSTEM_PROMPT)

    async def test_old_skill_ids_and_original_results_are_available_after_compression(self):
        skill_id = "fixture/skills/python-testing/SKILL.md"
        results = "描述 " * 2000 + f"\nid: {skill_id}\n"
        self.mcp.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text=results)])
        model = model_with(answer("介绍"), tool_response(call("skills_search_skills", {"query": "pytest"}, "search_old")),
            answer("第一轮结果"), answer("第二轮结果"), answer("第三轮结果"))
        conversation = self.conversation(model)
        await conversation.introduce()
        await conversation.submit("搜索")
        await conversation.submit("第二轮")
        await conversation.submit("第三轮")
        memory = json.loads(self.state["context"][0]["content"].split("\n", 1)[1])
        record = next(record for record in memory["records"] if record.get("type") == "function_call_output")
        self.assertIn(skill_id, record["skill_ids"])
        self.assertTrue(record["truncated"])
        original = await conversation.execute_tool("read_conversation_history", {"start": record["history_index"], "limit": 1})
        full = json.loads(original["data"][0]["item"]["output"])
        self.assertEqual(full["data"]["content"][0]["text"], results)
        self.assertEqual(self.state["tool_calls"][0]["attempts"], 1)
        model.responses.compact.assert_not_awaited()

    async def test_recent_tool_batch_is_replayed_with_all_matching_results(self):
        model = model_with(answer("介绍"), answer("旧资料 " * 2000),
            tool_response(call("download_skill", {"id": "python-testing"}, "download"),
                          call("search_saved_skills", {"query": "python"}, "local_search")),
            answer("第二轮结果"), answer("第三轮结果"))
        conversation = self.conversation(model)
        await conversation.introduce()
        await conversation.submit("旧查询")
        await conversation.submit("下载和查询")
        await conversation.submit("第三轮")
        window = model.responses.create.call_args.kwargs["input"]
        calls = [item["call_id"] for item in window if item.get("type") == "function_call"]
        outputs = [item["call_id"] for item in window if item.get("type") == "function_call_output"]
        self.assertEqual(calls, ["download", "local_search"])
        self.assertEqual(outputs, calls)
        self.assertEqual(self.library.search("pytest")["count"], 1)

    async def test_archive_tool_can_recover_old_content_through_model_loop_after_restart(self):
        first = model_with(answer("介绍"), answer("完整旧内容 " * 1000), answer("第二轮"), answer("第三轮"))
        conversation = self.conversation(first)
        await conversation.introduce()
        for text in ("最早的问题", "第二个问题", "第三个问题"):
            await conversation.submit(text)
        original_index = next(index for index, item in enumerate(self.state["history"]) if item.get("content") == "最早的问题")
        second = model_with(tool_response(call("read_conversation_history", {"start": original_index, "limit": 1}, "archive")), answer("已读取旧问题"))
        resumed = self.conversation(second, self.store.load())
        await resumed.submit("回看第一个问题")
        output = json.loads(second.responses.create.call_args.kwargs["input"][-1]["output"])
        self.assertEqual(output["data"][0]["item"]["content"], "最早的问题")
        self.assertEqual(resumed.state["turn_count"], 4)
        second.responses.compact.assert_not_awaited()

    async def test_local_save_failure_keeps_original_state(self):
        self.state["history"] = [{"role": "user", "content": "第一轮"}, answer("旧资料 " * 2000).output[0],
                                 {"role": "user", "content": "第二轮"}, {"role": "user", "content": "第三轮"}]
        self.state["context"] = copy.deepcopy(self.state["history"])
        original = copy.deepcopy(self.state)
        model = model_with()
        with patch.object(self.store, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await compact_context(model, self.state, self.store, force=True)
        self.assertEqual(self.state, original)
        model.responses.compact.assert_not_awaited()

    async def test_no_old_turn_to_compress_keeps_current_evidence_even_above_trigger(self):
        self.state["history"] = [{"role": "user", "content": "巨大但仍需读取的证据 " * 2000}]
        self.state["context"] = copy.deepcopy(self.state["history"])
        original = copy.deepcopy(self.state["context"])
        model = model_with()
        self.assertFalse(await compact_context(model, self.state, self.store, force=True))
        self.assertEqual(self.state["context"], original)
        self.assertEqual(self.state["compactions"], [])

    def test_mode_switch_and_migration_work_on_existing_checkpoint(self):
        state = self.store.start("deepseek-flash", self.library.path, 4000, new=True, compact_mode="standalone")
        original_id = state["conversation_id"]
        resumed = self.store.start("deepseek-flash", self.library.path, 4000, resume=True, compact_mode="local")
        self.assertEqual(resumed["compact_mode"], "local")
        self.assertEqual(resumed["conversation_id"], original_id)
        del resumed["compact_mode"]
        self.store.save(resumed)
        self.assertEqual(self.store.load()["compact_mode"], "local")

    def test_archive_read_is_paged_and_rejects_invalid_indices(self):
        self.state["history"] = [{"role": "user", "content": str(i)} for i in range(12)]
        first = read_history(self.state, 0, 10)
        self.assertEqual(first["next_start"], 10)
        self.assertEqual(len(first["data"]), 10)
        self.assertIsNone(read_history(self.state, 10, 10)["next_start"])
        for start, limit in ((-1, 1), (12, 1), (0, 11), (True, 1)):
            with self.assertRaises(ValueError):
                read_history(self.state, start, limit)


if __name__ == "__main__":
    unittest.main()
