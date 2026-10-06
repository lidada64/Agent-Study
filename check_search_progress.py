"""Inspect five search checkpoints and verify recovery without a model API or MCP.

Run from the project root: python code/check_search_progress.py --case all
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.runner import run_agent
from agent_core.state import StateStore

CASES = {
    "not-started": ("pending", None),
    "ff-running": ("running", None),
    "ff-completed": ("completed", None),
    "ft-running": ("completed", "running"),
    "both-completed": ("completed", "completed"),
}


class SimulatedExit(BaseException):
    """Stop immediately after an atomic checkpoint, like process termination."""


class SearchModel:
    """Choose the next step from saved history; only the model is simulated."""

    def __init__(self, target):
        self.target = target
        self.responses = self
        self.requests = 0

    async def create(self, **kwargs):
        self.requests += 1
        outputs = {
            item["call_id"]: json.loads(item["output"])
            for item in kwargs["input"] if item.get("type") == "function_call_output"
        }
        if "files" not in outputs:
            name, arguments, identifier = "find_file", {"name": self.target.name}, "files"
        elif "text" not in outputs:
            name, arguments, identifier = "find_text", {
                "paths": outputs["files"]["data"], "text": "lidada",
            }, "text"
        else:
            return SimpleNamespace(output=[], output_text="Found lidada on lines 2 and 3.")
        return SimpleNamespace(output=[{
            "type": "function_call", "name": name,
            "arguments": json.dumps(arguments), "call_id": identifier,
        }], output_text="")


def tool_statuses(state):
    calls = {call["name"]: call for call in state["tool_calls"]}
    return tuple(calls[name]["status"] if name in calls else None for name in ("find_file", "find_text"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


async def check_case(name, output_root):
    directory = (output_root / name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "sample.txt"
    target.write_text("first line\nlidada\nlast lidada\n", encoding="utf-8")
    state_path = directory / "state.json"
    checkpoint_path = directory / "checkpoint.json"
    handler = logging.FileHandler(directory / "agent.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("agent_core.runner")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    save = StateStore.save
    expected = CASES[name]

    def stop_at_checkpoint(store, state):
        save(store, state)
        if tool_statuses(state) == expected:
            checkpoint_path.write_bytes(store.path.read_bytes())
            logger.info("Simulated exit: case=%s ff=%s ft=%s", name, *expected)
            raise SimulatedExit()

    try:
        # Searches run against a small real file; checkpoints and the resume
        # loop use production code. No network, API key or existing task needed.
        with patch("agent_core.local_tools.ROOT", directory):
            with patch.object(StateStore, "save", stop_at_checkpoint):
                try:
                    await run_agent(SearchModel(target), None, [], "Find lidada in sample.txt", "offline", 3,
                                    state_path=state_path, new_task=True)
                except SimulatedExit:
                    pass
                else:
                    raise AssertionError(f"Checkpoint was not reached: {name}")
            checkpoint = StateStore(state_path).load()
            require(tool_statuses(checkpoint) == expected, "Unexpected checkpoint")
            resumed_model = SearchModel(target)
            status, _ = await run_agent(resumed_model, None, [], None, "offline", 3,
                                        state_path=state_path, resume=True)
        final = StateStore(state_path).load()
        require(status == "completed", "Resume did not complete")
        require(tool_statuses(final) == ("completed", "completed"), "Tools did not complete")
        require(final["task_id"] == checkpoint["task_id"], "Resume created another task")
        calls = {call["name"]: call for call in final["tool_calls"]}
        expected_attempts = {
            "find_file": 2 if name == "ff-running" else 1,
            "find_text": 2 if name == "ft-running" else 1,
        }
        for tool_name, attempts in expected_attempts.items():
            require(calls[tool_name]["attempts"] == attempts, f"Unexpected retries: {tool_name}")
        hits = calls["find_text"]["result"]["data"]
        require([hit["line_number"] for hit in hits] == [2, 3], "Incorrect search results")
        outputs = [item["call_id"] for item in final["history"] if item.get("type") == "function_call_output"]
        require(outputs == ["files", "text"], "Duplicate or missing outputs")
        logger.info("Recovery verified: attempts=%s model_requests=%s", expected_attempts, resumed_model.requests)
        print(f"PASS {name}: ff={expected[0]}, ft={expected[1] or 'not scheduled'} -> completed")
        print(f"  Before resume: {checkpoint_path}")
        print(f"  After resume:  {state_path}")
        print(f"  Log:           {directory / 'agent.log'}")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        handler.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "progress_checks")
    args = parser.parse_args()
    for name in CASES if args.case == "all" else [args.case]:
        await check_case(name, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
