"""Model-generated summaries replace working input lists without changing user turns."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, TextContent, Tool

from agent_core.context import compact_context, context_size
from agent_core.conversation import SkillConversation
from agent_core.conversation_state import ConversationStore
from agent_core.skill_library import SkillLibrary
from test_agent_conversation import answer, model_with, tool_response, DOCUMENT
from test_agent_state import call


class SummaryContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = ConversationStore(self.root / "conversation.json")
        self.library = SkillLibrary(self.root / "skills.json")
        self.state = self.store.start("deepseek-flash", self.library.path, 4000, compact_mode="summary")
        self.mcp = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text=DOCUMENT)])))
        self.tools = [Tool(name="search_skills", inputSchema={}), Tool(name="get_skill", inputSchema={})]

    def conversation(self, model, state=None):
        return SkillConversation(model, self.mcp, self.tools, state or self.state, self.store, self.library)

    async def populate_old_turns(self, conversation):
        await conversation.introduce()
        await conversation.submit("第一个查询的关键词是 pytest")
        await conversation.submit("第二个查询")

    async def test_summary_request_is_separate_and_replacement_list_is_used_as_input(self):
        model = model_with(answer("介绍"), answer("旧内容 " * 3000), answer("第二轮回答"),
                           answer("用户的关键词是 pytest。第一轮资料位于 history_index=3。"), answer("第三轮回答"))
        conversation = self.conversation(model)
        await self.populate_old_turns(conversation)
        await conversation.submit("继续第一轮关键词")
        requests = model.responses.create.call_args_list
        summary = requests[3].kwargs
        self.assertEqual(summary["tools"], [])
        self.assertFalse(summary["store"])
        self.assertNotIn("previous_response_id", summary)
        source = json.loads(summary["input"][0]["content"])
        self.assertIsNone(source["previous_summary"])
        self.assertTrue(any(item["item"].get("content") == "第一个查询的关键词是 pytest" for item in source["items"]))
        window = requests[4].kwargs["input"]
        self.assertIn("用户的关键词是 pytest", window[0]["content"])
        self.assertIn({"role": "user", "content": "第二个查询"}, window[1:])
        self.assertIn({"role": "user", "content": "继续第一轮关键词"}, window[1:])
        self.assertFalse(any(item.get("type") == "compaction" for item in window))
        self.assertEqual(self.state["turn_count"], 3)
        self.assertEqual(len(self.state["steps"]), 4)  # Introduction + 3 user replies.
        self.assertEqual(self.state["compactions"][0]["mode"], "summary")
        self.assertLess(context_size(self.state["context"]), context_size(self.state["history"]))
        self.assertEqual(self.store.load()["summary_memory"], self.state["summary_memory"])
        model.responses.compact.assert_not_awaited()

    async def test_rolling_summary_uses_previous_memory_and_only_new_archived_turns(self):
        model = model_with(answer("介绍"), answer("最早资料 " * 2000), answer("第二轮资料 " * 2000),
                           answer("摘要一：关键词 pytest。"), answer("第三轮回答"),
                           answer("摘要二：关键词 pytest，已确认第二轮资料。"), answer("第四轮回答"))
        conversation = self.conversation(model)
        await self.populate_old_turns(conversation)
        await conversation.submit("第三个查询")
        first_end = self.state["summary_memory"]["history_end"]
        await conversation.submit("第四个查询")
        source = json.loads(model.responses.create.call_args_list[5].kwargs["input"][0]["content"])
        self.assertEqual(source["previous_summary"], "摘要一：关键词 pytest。")
        self.assertEqual(source["new_history_range"][0], first_end)
        self.assertNotIn("最早资料", json.dumps(source["items"], ensure_ascii=False))
        self.assertIn("第二轮资料", json.dumps(source["items"], ensure_ascii=False))
        self.assertEqual(len(self.state["compactions"]), 2)
        self.assertEqual(self.state["turn_count"], 4)
        count = model.responses.create.await_count
        self.assertFalse(await compact_context(model, self.state, self.store, force=True))
        self.assertEqual(model.responses.create.await_count, count)

    async def test_failed_summary_keeps_context_and_resumes_the_same_user_turn(self):
        for bad in (answer("", status="completed"), answer("半段摘要", status="incomplete"),
                    tool_response(call("download_skill", {"id": "python-testing"}, "wrong"))):
            with self.subTest(response=bad.output_text):
                self.state = self.store.start("deepseek-flash", self.library.path, 4000, compact_mode="summary", new=True)
                model = model_with(answer("介绍"), answer("旧资料 " * 3000), answer("第二轮"), bad)
                conversation = self.conversation(model)
                await self.populate_old_turns(conversation)
                original = copy.deepcopy(self.state["context"])
                self.assertEqual(await conversation.submit("第三轮"), ("compact_error", ""))
                self.assertEqual(self.state["context"], original + [{"role": "user", "content": "第三轮"}])
                self.assertNotIn("summary_memory", self.state)
                self.assertEqual(self.state["tool_calls"], [])
                resumed_model = model_with(answer("旧资料摘要：关键词 pytest。"), answer("已恢复"))
                resumed = self.conversation(resumed_model, self.store.load())
                self.assertEqual(await resumed.resume_turn(), ("waiting_user", "已恢复"))
                self.assertEqual(resumed.state["turn_count"], 3)

    async def test_recent_tool_calls_and_results_survive_summary_replacement(self):
        model = model_with(answer("介绍"), answer("旧资料 " * 3000),
                           tool_response(call("download_skill", {"id": "python-testing"}, "save")), answer("第二轮已保存"),
                           answer("第一轮资料摘要。"), answer("第三轮继续"))
        conversation = self.conversation(model)
        await conversation.introduce()
        await conversation.submit("第一轮")
        await conversation.submit("下载")
        original_calls = copy.deepcopy(self.state["tool_calls"])
        await conversation.submit("第三轮")
        window = model.responses.create.call_args.kwargs["input"]
        self.assertEqual([item["call_id"] for item in window if item.get("type") == "function_call"], ["save"])
        self.assertEqual([item["call_id"] for item in window if item.get("type") == "function_call_output"], ["save"])
        self.assertEqual(self.state["tool_calls"], original_calls)
        self.assertEqual(self.library.search("pytest")["count"], 1)

    async def test_save_failure_keeps_input_list_and_previous_summary(self):
        model = model_with(answer("介绍"), answer("旧资料 " * 3000), answer("第二轮"), answer("摘要"))
        conversation = self.conversation(model)
        await self.populate_old_turns(conversation)
        conversation.append_items([{"role": "user", "content": "第三轮"}])
        original = copy.deepcopy(self.state)
        with patch.object(self.store, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await compact_context(model, self.state, self.store, force=True)
        self.assertEqual(self.state, original)

    async def test_larger_summary_does_not_replace_context(self):
        model = model_with(answer("介绍"), answer("旧资料 " * 1000), answer("第二轮"), answer("更大的摘要 " * 5000))
        conversation = self.conversation(model)
        await self.populate_old_turns(conversation)
        conversation.append_items([{"role": "user", "content": "第三轮"}])
        original = copy.deepcopy(self.state)
        self.assertFalse(await compact_context(model, self.state, self.store, force=True))
        self.assertEqual(self.state, original)

    def test_resume_can_switch_to_summary_and_reject_invalid_memory(self):
        resumed = self.store.start("deepseek-flash", self.library.path, 4000, compact_mode="summary", resume=True)
        self.assertEqual(resumed["compact_mode"], "summary")
        resumed["summary_memory"] = {"text": "bad", "history_end": -1}
        self.store.save(resumed)
        with self.assertRaisesRegex(ValueError, "summary memory"):
            self.store.load()


if __name__ == "__main__":
    unittest.main()
