"""Interactive Skill conversation, with durable tool recovery and explicit compaction."""

import asyncio
import copy
import logging

from .config import MAX_USER_TURNS
from .context import HISTORY_TOOLS, compact_context, read_history
from .mcp_tools import function_schema
from .runner import finish_pending, json_item
from .skill_library import LIBRARY_TOOLS
from .state import now, set_tool_status


EXIT_WORDS = {"退出", "请退出", "我要退出", "退出吧", "结束", "结束对话", "退出对话", "结束聊天", "停止对话", "再见", "拜拜", "不聊了", "exit", "quit", "bye", "/exit", "/quit", "q"}


def is_exit(text):
    return text.strip().casefold().rstrip("。.!！ ") in EXIT_WORDS


class SkillConversation:
    def __init__(self, model_client, mcp_client, discovered, state, store, library, *, max_steps=8):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model_client, self.mcp_client = model_client, mcp_client
        self.state, self.store, self.library = state, store, library
        self.max_steps = max_steps
        self.mcp_tools = {f"skills_{tool.name}": tool.name for tool in discovered}
        if "skills_search_skills" not in self.mcp_tools or "skills_get_skill" not in self.mcp_tools:
            raise ValueError("Skill conversations require MCP search_skills and get_skill")
        self.tools = LIBRARY_TOOLS + HISTORY_TOOLS + [function_schema(tool) for tool in discovered]

    def append_items(self, items):
        self.state["history"].extend(copy.deepcopy(items))
        self.state["context"].extend(copy.deepcopy(items))

    def close(self, reason):
        self.state.update(status="closed", close_reason=reason, phase="done")
        self.store.save(self.state)

    @property
    def unfinished_turn(self):
        turns = self.state["turns"]
        return turns[-1] if turns and turns[-1]["status"] == "running" else None

    async def introduce(self):
        if self.state["turns"]:
            if self.unfinished_turn:
                return await self.resume_turn()
            return "ready", ""
        self.state["turns"].append({"number": 0, "kind": "introduction", "status": "running", "reply": ""})
        self.append_items([{"role": "developer", "content": "请先向用户介绍你的 Skill 检索、下载录入、本地关键词查询能力和退出方法，然后等待用户输入。"}])
        self.store.save(self.state)
        return await self.resume_turn()

    async def submit(self, text):
        if self.state["status"] == "closed":
            return "closed", ""
        if is_exit(text):
            self.close("user_exit")
            return "closed", "对话已结束。"
        if not text.strip():
            return "ready", ""
        if self.unfinished_turn:
            raise ValueError("Resume the unfinished turn before accepting another user message")
        if not self.state["turns"]:
            raise ValueError("Introduce the assistant before accepting a user message")
        if self.state["turn_count"] >= MAX_USER_TURNS:
            self.close("turn_limit")
            return "closed", "已达到 10 轮，对话结束。"
        self.state["turn_count"] += 1
        self.state["turns"].append({
            "number": self.state["turn_count"], "kind": "user", "status": "running",
            "text": text, "reply": "", "created_at": now(),
        })
        self.append_items([{"role": "user", "content": text}])
        self.state.update(status="running", last_error=None)
        self.store.save(self.state)
        return await self.resume_turn()

    async def execute_tool(self, name, arguments):
        if name == "read_conversation_history":
            return read_history(self.state, arguments["start"], arguments["limit"])
        return await self.library.execute(name, arguments, self.mcp_client, self.mcp_tools)

    def fail(self, status, error, step=None):
        if step is not None:
            step["status"] = "model_error"
        self.state["status"] = status
        self.state["last_error"] = {"type": type(error).__name__, "message": str(error)}
        self.store.save(self.state)
        return status, ""

    async def resume_turn(self):
        turn = self.unfinished_turn
        if turn is None:
            return self.state["status"], ""
        state = self.state
        state.update(status="running", last_error=None)
        for call in state["tool_calls"]:
            if call["running"]:
                set_tool_status(call, "interrupted")
        self.store.save(state)
        try:
            await finish_pending(state, self.store, self.mcp_client, self.mcp_tools, tool_executor=self.execute_tool)
            for _ in range(self.max_steps):
                try:
                    await compact_context(self.model_client, state, self.store)
                except Exception as exc:
                    return self.fail("compact_error", exc)
                # Compaction swaps the state dictionary contents; obtain live records again.
                turn = self.unfinished_turn
                if state["steps"] and state["steps"][-1]["status"] in {"requesting_model", "interrupted", "model_error"}:
                    step = state["steps"][-1]
                    step["status"] = "requesting_model"
                else:
                    step = {"number": len(state["steps"]) + 1, "turn": turn["number"], "status": "requesting_model", "created_at": now()}
                    state["steps"].append(step)
                state["phase"] = "introduction" if turn["kind"] == "introduction" else "responding"
                self.store.save(state)
                try:
                    response = await self.model_client.responses.create(
                        model=state["model"], instructions=state["instructions"],
                        input=copy.deepcopy(state["context"]),
                        tools=[] if turn["kind"] == "introduction" else self.tools,
                        store=False, include=["reasoning.encrypted_content"],
                    )
                    if getattr(response, "status", "completed") != "completed":
                        raise ValueError(f"Model response is {response.status}; turn remains unfinished")
                    output = [json_item(item) for item in response.output]
                    calls = [item for item in output if item.get("type") == "function_call"]
                    if turn["kind"] == "introduction" and calls:
                        raise ValueError("The introduction must not call tools")
                    if not calls and not response.output_text.strip():
                        raise ValueError("Model returned no answer; turn remains unfinished")
                    seen = {call["call_id"] for call in state["tool_calls"]}
                    for item in calls:
                        if item["call_id"] in seen:
                            raise ValueError(f"Duplicate tool call id: {item['call_id']}")
                        seen.add(item["call_id"])
                except Exception as exc:
                    return self.fail("model_error", exc, step)
                self.append_items(output)
                for item in calls:
                    call = {"call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"], "step": step["number"], "attempts": 0, "result": None}
                    set_tool_status(call, "pending")
                    state["tool_calls"].append(call)
                step["status"] = "tools_pending" if calls else "completed"
                if not calls:
                    turn.update(status="completed", reply=response.output_text, completed_at=now())
                    state.update(status="waiting_user", phase="waiting_user")
                    if state["turn_count"] == MAX_USER_TURNS:
                        state.update(status="closed", close_reason="turn_limit", phase="done")
                self.store.save(state)
                if not calls:
                    return state["status"], turn["reply"]
                await finish_pending(state, self.store, self.mcp_client, self.mcp_tools, tool_executor=self.execute_tool)
            state["status"] = "max_steps"
            self.store.save(state)
            return "max_steps", ""
        except (asyncio.CancelledError, KeyboardInterrupt):
            state["status"] = "interrupted"
            for call in state["tool_calls"]:
                if call["running"]:
                    set_tool_status(call, "interrupted")
            if state["steps"] and state["steps"][-1]["status"] == "requesting_model":
                state["steps"][-1]["status"] = "interrupted"
            self.store.save(state)
            raise


async def interactive_loop(conversation, first_prompt=None, *, read_input=input, write_output=print, hooks=None):
    """Presentation hooks run outside the durable agent state machine."""
    def show_reply(reply):
        if hooks is not None:
            try:
                hooks.on_reply(reply)
                return
            except Exception:
                logging.getLogger(__name__).warning("Reply hook failed; displaying original text", exc_info=True)
        write_output(f"助手：{reply}")

    try:
        return await _interactive_loop(conversation, first_prompt, read_input=read_input,
                                       write_output=write_output, show_reply=show_reply)
    finally:
        if hooks is not None:
            try:
                # A hook gets a snapshot and cannot modify saved conversation state.
                hooks.on_session_end(copy.deepcopy(conversation.state))
            except Exception:
                logging.getLogger(__name__).warning("SessionEnd hook failed; checkpoint retained", exc_info=True)


async def _interactive_loop(conversation, first_prompt, *, read_input, write_output, show_reply):
    """Count user turns, not the introduction, tools, blank input or exit commands."""
    if conversation.state["status"] == "closed":
        write_output("该对话已结束，使用 --new 开始新对话。")
        return 0
    status, reply = await conversation.introduce()
    if reply:
        show_reply(reply)
    while status in {"ready", "waiting_user"}:
        if first_prompt is not None:
            text, first_prompt = first_prompt, None
        else:
            try:
                text = read_input(f"你（{conversation.state['turn_count'] + 1}/10）：")
            except EOFError:
                conversation.close("eof")
                write_output("输入已结束，对话已保存。")
                return 0
        status, reply = await conversation.submit(text)
        if reply:
            show_reply(reply)
    if status == "closed":
        if conversation.state["close_reason"] == "turn_limit":
            write_output("已完成 10 轮用户对话，自动退出。")
        return 0
    error = conversation.state.get("last_error")
    if error:
        write_output(f"本轮未完成（{status}）：{error['message']}")
    if status == "compact_error":
        write_output("压缩未完成，原上下文仍保留；DeepSeek 可用 local 或 summary 模式，standalone 需要对应端点。")
    write_output(f"对话已保存至 {conversation.store.path}，使用 --resume 继续本轮。")
    return 1
