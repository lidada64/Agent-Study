"""Presentation must not affect durable conversation behavior."""

import asyncio
import copy
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent_core.conversation import interactive_loop
from agent_core.presentation import MarkdownHooks, MarkdownRenderer, transcript_markdown


def state():
    return {"status": "waiting_user", "turn_count": 0, "close_reason": None, "last_error": None,
            "turns": [{"kind": "introduction", "number": 0, "status": "completed", "reply": "# 欢迎\n\n**搜索 Skill**"}]}


class PresentationTests(unittest.TestCase):
    def test_glow_only_receives_local_markdown_on_stdin(self):
        output = []
        text = "# 中文\n\n$(danger) `echo nope` https://example.com"
        with patch("agent_core.presentation.shutil.which", return_value="glow.exe"), patch(
            "agent_core.presentation.subprocess.run", return_value=SimpleNamespace(stdout="rendered\n")
        ) as run:
            MarkdownRenderer("glow", write_output=output.append, width=80).render(text)
        self.assertEqual(output, ["rendered"])
        self.assertEqual(run.call_args.args[0], ["glow.exe", "-w", "80", "-"])
        self.assertEqual(run.call_args.kwargs["input"], text)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_glow_timeout_falls_back_without_losing_text(self):
        output = []
        with patch("agent_core.presentation.shutil.which", return_value="glow"), patch(
            "agent_core.presentation.subprocess.run", side_effect=subprocess.TimeoutExpired("glow", 15)
        ), patch.dict("sys.modules", {"rich.console": None}):
            MarkdownRenderer("glow", write_output=output.append).render("# hello")
        self.assertEqual(output, ["# hello"])

    def test_redirected_auto_output_is_original_markdown(self):
        output = []
        with patch("sys.stdout.isatty", return_value=False), patch("agent_core.presentation.subprocess.run") as run:
            MarkdownRenderer(write_output=output.append).render("**你好**")
        run.assert_not_called()
        self.assertEqual(output, ["**你好**"])

    def test_rich_renders_chinese_headings_lists_and_tables(self):
        output = []
        MarkdownRenderer("rich", write_output=output.append, width=80).render(
            "# 欢迎\n\n**搜索技能**\n\n- 下载\n\n| 名称 | 状态 |\n| --- | --- |\n| pytest | 已录入 |"
        )
        rendered = output[0]
        self.assertIn("欢迎", rendered)
        self.assertIn("pytest", rendered)
        self.assertNotIn("**搜索技能**", rendered)
        self.assertNotIn("| --- |", rendered)

    def test_export_does_not_overwrite_checkpoint_or_include_tool_payloads(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "conversation.md"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            saved = state()
            saved["history"] = [{"type": "reasoning", "text": "hidden reasoning"}]
            saved["tool_calls"] = [{"result": "hidden tool payload"}]
            saved["turns"].append({"number": 1, "kind": "user", "text": "查找 pytest", "status": "running", "reply": ""})
            hooks = MarkdownHooks(checkpoint, mode="plain", render_session_end=True, write_output=lambda _: None)
            hooks.on_session_end(saved)
            exported = hooks.transcript_path.read_text(encoding="utf-8")
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), "checkpoint")
            self.assertIn("查找 pytest", exported)
            self.assertIn("尚未完成", exported)
            self.assertNotIn("hidden", exported)


class HookLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def conversation(self):
        current = state()
        conversation = SimpleNamespace(state=current, store=SimpleNamespace(path="conversation.json"),
                                       introduce=AsyncMock(return_value=("waiting_user", "# 欢迎")))

        def close(reason):
            current.update(status="closed", close_reason=reason)

        conversation.close = close
        return conversation

    async def test_exit_eof_and_turn_limit_fire_session_end_once(self):
        for reason in ("user_exit", "eof", "turn_limit"):
            with self.subTest(reason=reason):
                conversation = self.conversation()
                hooks = SimpleNamespace(on_reply=Mock(), on_session_end=Mock())
                if reason == "eof":
                    reader = Mock(side_effect=EOFError)
                else:
                    def submit(text):
                        conversation.close(reason)
                        if reason == "turn_limit":
                            conversation.state["turn_count"] = 10
                        return "closed", "**完成**"
                    conversation.submit = AsyncMock(side_effect=submit)
                    reader = lambda _: "退出"
                code = await interactive_loop(conversation, read_input=reader, write_output=lambda _: None, hooks=hooks)
                self.assertEqual(code, 0)
                hooks.on_session_end.assert_called_once()
                self.assertEqual(hooks.on_session_end.call_args.args[0]["close_reason"], reason)
                hooks.on_reply.assert_any_call("# 欢迎")

    async def test_model_error_and_interrupt_export_without_closing_conversation(self):
        for interrupted in (False, True):
            conversation = self.conversation()
            hooks = SimpleNamespace(on_reply=Mock(), on_session_end=Mock())
            if interrupted:
                conversation.submit = AsyncMock(side_effect=asyncio.CancelledError)
                with self.assertRaises(asyncio.CancelledError):
                    await interactive_loop(conversation, read_input=lambda _: "test", hooks=hooks, write_output=lambda _: None)
            else:
                conversation.state.update(status="model_error", last_error={"message": "retry"})
                conversation.introduce.return_value = ("model_error", "")
                self.assertEqual(await interactive_loop(conversation, hooks=hooks, write_output=lambda _: None), 1)
            hooks.on_session_end.assert_called_once()
            self.assertNotEqual(conversation.state["status"], "closed")

    async def test_broken_hooks_cannot_change_state_or_exit_code(self):
        conversation = self.conversation()
        conversation.close("user_exit")
        original = copy.deepcopy(conversation.state)

        def broken(snapshot):
            snapshot["status"] = "corrupted"
            raise OSError("disk full")

        hooks = SimpleNamespace(on_reply=Mock(), on_session_end=broken)
        code = await interactive_loop(conversation, hooks=hooks, write_output=lambda _: None)
        self.assertEqual(code, 0)
        self.assertEqual(conversation.state, original)


if __name__ == "__main__":
    unittest.main()
