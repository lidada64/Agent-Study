"""Local Markdown presentation hooks; never change model input or checkpoints."""

import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkdownRenderer:
    def __init__(self, mode="auto", *, write_output=print, width=None):
        if mode not in {"auto", "glow", "rich", "plain"}:
            raise ValueError("Unknown Markdown renderer")
        self.mode, self.write_output = mode, write_output
        self.width = width or shutil.get_terminal_size((88, 24)).columns

    def render(self, text):
        if self.mode == "plain":
            self.write_output(text)
            return
        # Auto leaves redirected output machine-readable and free of ANSI escapes.
        if self.mode == "auto" and not sys.stdout.isatty():
            self.write_output(text)
            return
        glow = shutil.which("glow") if self.mode in {"auto", "glow"} else None
        if glow:
            try:
                result = subprocess.run(
                    [glow, "-w", str(self.width), "-"], input=text,
                    text=True, encoding="utf-8", errors="replace", capture_output=True,
                    timeout=15, check=True,
                )
                if not result.stdout.strip():
                    raise ValueError("Glow returned no output")
                self.write_output(result.stdout.rstrip("\r\n"))
                return
            except (OSError, subprocess.SubprocessError, ValueError):
                logger.warning("Glow rendering failed; falling back to Rich")
        elif self.mode == "glow":
            logger.warning("Glow is not on PATH; falling back to Rich")
        try:
            from rich.console import Console
            from rich.markdown import Markdown

            buffer = io.StringIO()
            Console(file=buffer, width=self.width, force_terminal=sys.stdout.isatty()).print(Markdown(text))
            self.write_output(buffer.getvalue().rstrip("\n"))
        except Exception:
            logger.warning("Markdown rendering failed; displaying original text")
            self.write_output(text)


def transcript_markdown(state):
    """Export user-visible messages, excluding reasoning and tool payloads."""
    parts = ["# Skill 助手对话", f"状态：{state['status']}；用户轮次：{state['turn_count']}/10"]
    for turn in state["turns"]:
        if turn["kind"] == "introduction":
            parts.append("## 助手介绍")
        else:
            parts.extend([f"## 第 {turn['number']} 轮", "### 用户", turn["text"], "### 助手"])
        if turn.get("reply"):
            parts.append(turn["reply"])
        elif turn["status"] != "completed":
            parts.append("（本轮尚未完成，可恢复继续。）")
    return "\n\n".join(parts) + "\n"


def save_transcript(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MarkdownHooks:
    def __init__(self, checkpoint_path, *, mode="auto", render_session_end=False, write_output=print):
        checkpoint_path = Path(checkpoint_path)
        # Append rather than replace a suffix: never overwrite the JSON checkpoint.
        self.transcript_path = checkpoint_path.with_name(checkpoint_path.name + ".md")
        self.write_output = write_output
        self.renderer = MarkdownRenderer(mode, write_output=write_output)
        self.render_session_end = render_session_end

    def on_reply(self, reply):
        self.write_output("助手：")
        self.renderer.render(reply)

    def on_session_end(self, state):
        text = transcript_markdown(state)
        save_transcript(self.transcript_path, text)
        self.write_output(f"对话 Markdown 已保存至 {self.transcript_path}")
        if self.render_session_end:
            self.renderer.render(text)
