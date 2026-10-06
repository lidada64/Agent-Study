import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .config import (
    DEFAULT_PROMPT, DEFAULT_STATE_PATH, DEFAULT_CONVERSATION_PATH,
    DEFAULT_LIBRARY_PATH, DEFAULT_COMPACT_THRESHOLD, ROOT,
)
from .conversation import SkillConversation, interactive_loop
from .conversation_state import ConversationStore
from .mcp_connection import connect_mcp
from .mcp_tools import discover_skill_tools, execute_tool, skills_server_params
from .runner import run_agent
from .state import StateStore
from .skill_library import SkillLibrary
from .presentation import MarkdownHooks

logger = logging.getLogger(__name__)


async def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Interactive Skill search, download and automated input-list compaction.")
    parser.add_argument("prompt", nargs="?", help="Optional first user message after the introduction")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-flash"))
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum model requests per user turn or resume attempt (legacy: per invocation)")
    parser.add_argument("--state-file", type=Path, help="Conversation checkpoint (or legacy task checkpoint with --task)")
    parser.add_argument("--library-file", type=Path, default=DEFAULT_LIBRARY_PATH, help="JSON file containing downloaded Skill documents")
    parser.add_argument("--compact-threshold", type=int, default=DEFAULT_COMPACT_THRESHOLD, help="Compact before a model request when context reaches this many serialized UTF-8 bytes (not tokens)")
    parser.add_argument("--compact-mode", choices=("local", "summary", "standalone"), help="local: offline rules (default); summary: provider model generates a rolling summary; standalone: native compact endpoint; may be changed on resume")
    parser.add_argument("--task", action="store_true", help="Run the original resumable file/keyword search instead of a conversation")
    parser.add_argument("--markdown", choices=("auto", "glow", "rich", "plain"), default="auto", help="Reply renderer: auto prefers Glow, then optional Rich; redirected auto output stays plain")
    parser.add_argument("--render-session-end", action="store_true", help="Also render the exported conversation Markdown when the interactive session exits")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue the saved conversation or legacy task")
    mode.add_argument("--new", action="store_true", help="Start a new conversation or legacy task")
    parser.add_argument("--check-mcp", action="store_true", help="Search skills without calling the model")
    args = parser.parse_args()
    if args.state_file is None:
        args.state_file = DEFAULT_STATE_PATH if args.task else DEFAULT_CONVERSATION_PATH
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.compact_threshold < 1:
        parser.error("--compact-threshold must be positive")
    if args.resume and args.prompt is not None:
        parser.error("--resume continues the saved turn; omit the prompt")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "agent.log", encoding="utf-8")],
    )
    if not args.check_mcp and args.task:
        store = StateStore(args.state_file)
        prompt = args.prompt
        if prompt is None and not args.resume and (args.new or not store.path.exists()):
            prompt = DEFAULT_PROMPT
        try:
            state = store.start(prompt, args.model, resume=args.resume, new_task=args.new)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if state["status"] == "completed":
            print(state["final_text"])
            return 0
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY (and OPENAI_BASE_URL for a compatible provider)")
        logger.info("Task %s: %s; state=%s", state["task_id"], state["status"], store.path)
    elif not args.check_mcp:
        store = ConversationStore(args.state_file)
        try:
            state = store.start(args.model, args.library_file.resolve(), args.compact_threshold, compact_mode=args.compact_mode, resume=args.resume, new=args.new)
            library = SkillLibrary(state["library_file"])
            library.load()  # Validate before making any model or MCP requests.
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if state["status"] == "closed":
            print("该对话已结束，使用 --new 开始新对话。")
            return 0
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY and a provider/model supporting responses.create")
    try:
        async with connect_mcp(skills_server_params(), read_timeout_seconds=120) as mcp_client:
            discovered = await discover_skill_tools(mcp_client)
            logger.info("Discovered MCP tools: %s", [tool.name for tool in discovered])
            if args.check_mcp:
                result = await execute_tool(
                    "skills_search_skills", {"query": "python testing", "limit": 3},
                    mcp_client, {"skills_search_skills": "search_skills"},
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["ok"] else 1
            async with AsyncOpenAI() as model_client:
                if not args.task:
                    conversation = SkillConversation(model_client, mcp_client, discovered, state, store, library, max_steps=args.max_steps)
                    hooks = MarkdownHooks(store.path, mode=args.markdown, render_session_end=args.render_session_end)
                    return await interactive_loop(conversation, args.prompt, hooks=hooks)
                status, final_text = await run_agent(
                    model_client, mcp_client, discovered, state["prompt"], state["model"], args.max_steps,
                    state_path=args.state_file, resume=True,
                )
    except Exception:
        logger.exception("MCP connection or agent startup failed; saved progress is retained")
        return 1
    print(f"Final status: {status}")
    if final_text:
        print(final_text)
    if status != "completed":
        print(f"Progress saved in {args.state_file}. Continue with --resume.")
    return 0 if status == "completed" else 1


def entrypoint():
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Saved progress can be continued with --resume.")
        return 130
