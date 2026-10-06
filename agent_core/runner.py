import asyncio
import copy
import json
import logging

from .config import DEFAULT_PROMPT, DEFAULT_STATE_PATH, INSTRUCTIONS
from .local_tools import LOCAL_TOOLS
from .mcp_tools import execute_tool, function_schema
from .state import StateStore, now, set_tool_status

logger = logging.getLogger(__name__)


def json_item(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return vars(item).copy()


def discovered_paths(state):
    return {
        path for call in state["tool_calls"]
        if call["name"] == "find_file" and call["status"] == "completed"
        for path in call["result"]["data"]
    }


async def finish_pending(state, store, mcp_client, mcp_tools, *, tool_executor=None):
    if not any(not call["completed"] for call in state["tool_calls"]) and not any(
        step["status"] == "tools_pending" for step in state["steps"]
    ):
        return
    for call in state["tool_calls"]:
        if call["completed"]:
            continue
        call["attempts"] += 1
        set_tool_status(call, "running")
        state["status"] = "running"
        state["phase"] = "search" if call["name"] == "find_text" else "discover"
        store.save(state)
        logger.info("Tool started: step=%s tool=%s call_id=%s", call["step"], call["name"], call["call_id"])
        try:
            arguments = json.loads(call["arguments"])
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object")
            if call["name"] == "find_text":
                paths = arguments.get("paths")
                if not isinstance(paths, list) or not paths or not set(paths).issubset(discovered_paths(state)):
                    raise ValueError("find_text requires paths returned by a completed find_file call")
            if tool_executor is None:
                result = await execute_tool(call["name"], arguments, mcp_client, mcp_tools)
            else:
                result = await tool_executor(call["name"], arguments)
        except Exception as exc:
            logger.exception("Tool error: tool=%s", call["name"])
            result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        call["result"] = result
        set_tool_status(call, "completed" if result["ok"] else "failed")
        # Result and function output are committed together.
        output = {
            "type": "function_call_output", "call_id": call["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
        state["history"].append(output)
        if "context" in state:
            state["context"].append(copy.deepcopy(output))
        store.save(state)
        logger.info("Tool finished: tool=%s ok=%s", call["name"], result["ok"])
        if call["name"] in {"find_file", "find_text"} and result["ok"]:
            logger.info("Search result: tool=%s count=%s", call["name"], len(result["data"]))
    for step in state["steps"]:
        if step["status"] == "tools_pending":
            step["status"] = "completed"
    state["phase"] = "review"
    store.save(state)


async def run_agent(model_client, mcp_client, discovered, prompt, model, max_steps,
                    *, state_path=DEFAULT_STATE_PATH, resume=False, new_task=False):
    """Resume checkpoints first; max_steps limits new model requests this run."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    store = StateStore(state_path)
    if prompt is None and not resume and (new_task or not store.path.exists()):
        prompt = DEFAULT_PROMPT
    state = store.start(prompt, model, resume=resume, new_task=new_task)
    if state["status"] == "completed":
        return "completed", state["final_text"]
    mcp_tools = {f"skills_{tool.name}": tool.name for tool in discovered}
    state["status"] = "running"
    state["last_error"] = None
    for call in state["tool_calls"]:
        if call["running"]:
            set_tool_status(call, "interrupted")
    store.save(state)
    try:
        await finish_pending(state, store, mcp_client, mcp_tools)
        for _ in range(max_steps):
            # Keep declarations stable: some models retain their first-turn
            # understanding of capabilities in reasoning replayed in history.
            # Execution checks above still require real discovered file paths.
            tools = LOCAL_TOOLS + [function_schema(tool) for tool in discovered]
            if state["steps"] and state["steps"][-1]["status"] in {"requesting_model", "interrupted", "model_error"}:
                step = state["steps"][-1]
                step["status"] = "requesting_model"
            else:
                step = {"number": len(state["steps"]) + 1, "status": "requesting_model", "created_at": now()}
                state["steps"].append(step)
            state["status"] = "running"
            store.save(state)
            logger.info("Model step=%s available_tools=%s", step["number"], [tool["name"] for tool in tools])
            try:
                response = await model_client.responses.create(
                    model=state["model"], instructions=INSTRUCTIONS, tools=tools,
                    input=copy.deepcopy(state["history"]),
                )
            except Exception as exc:
                logger.exception("Model request failed")
                step["status"] = "model_error"
                state["status"] = "model_error"
                state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
                store.save(state)
                return "model_error", ""
            output = [json_item(item) for item in response.output]
            calls = [item for item in output if item["type"] == "function_call"]
            state["history"].extend(output)
            for item in calls:
                call = {
                    "call_id": item["call_id"], "name": item["name"],
                    "arguments": item["arguments"], "step": step["number"],
                    "attempts": 0, "result": None,
                }
                set_tool_status(call, "pending")
                state["tool_calls"].append(call)
            step["status"] = "tools_pending" if calls else "completed"
            if not calls:
                state.update(status="completed", phase="done", final_text=response.output_text)
            # Save the entire response before executing any tool.
            store.save(state)
            if not calls:
                return "completed", state["final_text"]
            await finish_pending(state, store, mcp_client, mcp_tools)
        state["status"] = "max_range"
        store.save(state)
        return "max_range", ""
    except (asyncio.CancelledError, KeyboardInterrupt):
        state["status"] = "interrupted"
        for call in state["tool_calls"]:
            if call["running"]:
                set_tool_status(call, "interrupted")
        if state["steps"] and state["steps"][-1]["status"] == "requesting_model":
            state["steps"][-1]["status"] = "interrupted"
        store.save(state)
        raise
