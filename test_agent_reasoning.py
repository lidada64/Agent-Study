"""Reasoning survives SDK parsing, checkpoints, tools and context compaction."""

import asyncio
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from mcp.types import CallToolResult, Tool
from openai import AsyncOpenAI

from agent_core.conversation import SkillConversation
from agent_core.conversation_state import ConversationStore
from agent_core.context import compact_context, read_history
from agent_core.presentation import MarkdownHooks, transcript_markdown
from agent_core.reasoning import step_reasoning_text
from agent_core.runner import run_agent
from agent_core.skill_library import SkillLibrary
from agent_core.state import StateStore
from test_agent_conversation import answer, model_with, tool_response
from test_agent_state import call


def reasoning(text, **extra):
    return {"type": "reasoning", "id": "rs_fixture", "status": "completed",
            "content": [{"type": "reasoning_text", "text": text}], **extra}


def with_reasoning(response, text):
    response.output.insert(0, reasoning(text))
    return response


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.store = ConversationStore(root / "conversation.json")
        self.library = SkillLibrary(root / "skills.json")
        self.state = self.store.start("deepseek-flash", self.library.path, 100000)
        self.mcp = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[])))
        self.tools = [Tool(name="search_skills", inputSchema={}), Tool(name="get_skill", inputSchema={})]

    def conversation(self, model, state=None, **kwargs):
        return SkillConversation(model, self.mcp, self.tools, state or self.state, self.store, self.library, **kwargs)

    async def test_real_sdk_keeps_deepseek_plaintext_and_replays_after_restart(self):
        requests = []
        raw = reasoning("先确认用户要检索的对象")

        def handler(request):
            requests.append(json.loads(request.content))
            message = answer("回复").output[0]
            message["id"] = "msg_fixture"
            return httpx.Response(200, json={
                "id": "resp_fixture", "object": "response", "created_at": 1,
                "model": "deepseek-flash", "status": "completed", "tools": [],
                "parallel_tool_calls": True, "tool_choice": "auto", "output": [raw, message],
            })

        shown = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            async with AsyncOpenAI(api_key="", base_url="https://fixture.invalid", http_client=http_client) as model:
                conversation = self.conversation(model, on_reasoning=shown.append)
                await conversation.introduce()
                await conversation.submit("第一轮")
                saved = self.store.load()
                await self.conversation(model, saved).submit("第二轮")
        self.assertEqual(shown, [raw["content"][0]["text"]] * 2)
        for request in requests:
            self.assertEqual(request["reasoning"], {"effort": "high"})
            self.assertFalse(request["store"])
            self.assertNotIn("previous_response_id", request)
        self.assertEqual([item for item in requests[-1]["input"] if item.get("type") == "reasoning"], [raw, raw])
        self.assertEqual(step_reasoning_text(saved, saved["steps"][0]), shown[0])
        self.assertIn(shown[0], transcript_markdown(saved))

    async def test_tool_reasoning_commits_before_display_and_replays_in_order(self):
        raw = reasoning("先检索再回答")
        first = tool_response(raw, call("skills_search_skills", {"query": "python"}, "search"))
        model = model_with(answer("介绍"), first, with_reasoning(answer("完成"), "检索已完成"), answer("下一轮"))
        output = []

        def show(text):
            saved = self.store.load()
            self.assertEqual(saved["steps"][-1]["reasoning"]["status"], "completed")
            self.assertEqual(step_reasoning_text(saved, saved["steps"][-1]), text)
            output.append(text)

        conversation = self.conversation(model, on_reasoning=show)
        await conversation.introduce()
        await conversation.submit("搜索")
        await conversation.submit("继续")
        replay = model.responses.create.call_args_list[2].kwargs["input"]
        index = replay.index(raw)
        self.assertEqual([item["type"] for item in replay[index:index + 3]], ["reasoning", "function_call", "function_call_output"])
        self.assertEqual(output, ["先检索再回答", "检索已完成"])
        self.assertEqual(self.store.load()["context"], self.state["history"])
        exported = transcript_markdown(self.state)
        self.assertLess(exported.index("先检索再回答"), exported.index("检索已完成"))
        self.assertNotIn('"call_id"', exported)

    async def test_error_interrupt_resume_and_disabled_states(self):
        first = model_with(answer("介绍"), RuntimeError("offline"))
        conversation = self.conversation(first)
        await conversation.introduce()
        await conversation.submit("查询")
        self.assertEqual(self.store.load()["steps"][-1]["reasoning"]["status"], "model_error")
        interrupted = self.conversation(model_with(asyncio.CancelledError()), self.store.load())
        with self.assertRaises(asyncio.CancelledError):
            await interrupted.resume_turn()
        self.assertEqual(self.store.load()["steps"][-1]["reasoning"]["status"], "interrupted")
        saved = self.store.start("ignored", self.library.path, 1, resume=True, reasoning_effort="none")
        model = model_with(answer("继续完成"))
        await self.conversation(model, saved).resume_turn()
        self.assertEqual(saved["turn_count"], 1)
        self.assertEqual(saved["steps"][-1]["reasoning"]["status"], "disabled")
        self.assertEqual(model.responses.create.call_args.kwargs["reasoning"], {"effort": "none"})

    async def test_hook_failure_cannot_lose_reasoning_or_fail_turn(self):
        hook = Mock(side_effect=RuntimeError("renderer failed"))
        conversation = self.conversation(model_with(with_reasoning(answer("介绍"), "推理")), on_reasoning=hook)
        self.assertEqual(await conversation.introduce(), ("waiting_user", "介绍"))
        self.assertEqual(step_reasoning_text(self.store.load(), self.state["steps"][0]), "推理")

    async def test_compaction_keeps_recent_raw_items_and_archives_old_reasoning(self):
        for mode in ("local", "summary"):
            with self.subTest(mode=mode):
                state = self.store.start("deepseek-flash", self.library.path, 4000, new=True, compact_mode=mode)
                old = reasoning("旧推理决策 " * 2000)
                recent = reasoning("近期推理")
                state["history"] = [{"role": "user", "content": "旧问题"}, old, answer("旧回答").output[0],
                                    {"role": "user", "content": "第二轮"}, recent, answer("回复").output[0],
                                    {"role": "user", "content": "第三轮"}]
                state["context"] = copy.deepcopy(state["history"])
                model = model_with(answer("旧 reasoning 决策摘要"))
                self.assertTrue(await compact_context(model, state, self.store, force=True))
                self.assertIn(recent, state["context"])
                self.assertEqual(read_history(state, 1, 1)["data"][0]["item"], old)
                if mode == "local":
                    memory = json.loads(state["context"][0]["content"].split("\n", 1)[1])
                    record = next(record for record in memory["records"] if record["type"] == "reasoning")
                    self.assertTrue(record["truncated"])
                    self.assertEqual(record["history_index"], 1)
                else:
                    source = json.loads(model.responses.create.call_args.kwargs["input"][0]["content"])
                    self.assertIn(old, [entry["item"] for entry in source["items"]])

    async def test_legacy_task_replays_reasoning_after_step_limit_resume(self):
        path = self.store.path.with_name("state.json")
        raw = reasoning("先搜索")
        first = model_with(tool_response(raw, call("skills_search_skills", {"query": "python"}, "search")))
        result = await run_agent(first, self.mcp, self.tools, "查找", "test", 1, state_path=path, reasoning_effort="low")
        self.assertEqual(result[0], "max_range")
        second = model_with(answer("完成"))
        await run_agent(second, self.mcp, self.tools, None, "test", 1, state_path=path, resume=True)
        self.assertIn(raw, second.responses.create.call_args.kwargs["input"])
        self.assertEqual(second.responses.create.call_args.kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(StateStore(path).load()["steps"][0]["reasoning"]["status"], "completed")

    def test_migration_validation_and_visible_summary_without_ciphertext(self):
        del self.state["reasoning"]
        self.store.save(self.state)
        self.assertEqual(self.store.load()["reasoning"], {"effort": "high"})
        self.state["reasoning"] = {"effort": "invalid"}
        self.store.save(self.state)
        with self.assertRaisesRegex(ValueError, "reasoning"):
            self.store.load()
        output = []
        hooks = MarkdownHooks(self.store.path, mode="plain", write_output=output.append)
        hooks.on_reasoning("**推理内容**")
        self.assertEqual(output, ["助手推理：", "**推理内容**"])
        state = {"status": "waiting_user", "turn_count": 0, "turns": [{"number": 0, "kind": "introduction", "status": "completed", "reply": "回答"}],
                 "history": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "可见摘要"}], "encrypted_content": "secret"}]}
        self.assertIn("可见摘要", transcript_markdown(state))
        self.assertNotIn("secret", transcript_markdown(state))


if __name__ == "__main__":
    unittest.main()
