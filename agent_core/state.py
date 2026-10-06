"""Atomic checkpoints. A state file belongs to one task and one running process."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import DEFAULT_STATE_PATH


def now():
    return datetime.now(timezone.utc).isoformat()


def set_tool_status(call, status):
    call.update(
        status=status,
        called=call["attempts"] > 0,
        running=status == "running",
        completed=status in {"completed", "failed"},
        updated_at=now(),
    )


def atomic_write_json(path, data):
    """Replace one JSON file only after its complete contents reach disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class StateStore:
    def __init__(self, path=DEFAULT_STATE_PATH):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return None
        # Never silently overwrite a broken or incompatible checkpoint.
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError("Unsupported state.json version")
        for field in ("prompt", "model", "status", "history", "steps", "tool_calls"):
            if field not in state:
                raise ValueError(f"Invalid state.json: missing {field}")
        return state

    def save(self, state):
        state["updated_at"] = now()
        atomic_write_json(self.path, state)

    def start(self, prompt, model, *, resume=False, new_task=False):
        existing = None if new_task else self.load()
        if resume and existing is None:
            raise ValueError("No saved task to resume")
        if existing and (resume or prompt is None or existing["status"] != "completed"):
            if prompt is not None and prompt != existing["prompt"]:
                raise ValueError("An unfinished task exists; use --new to start a different task")
            return existing
        if prompt is None:
            raise ValueError("A new task needs a prompt")
        state = {
            "version": 1, "task_id": str(uuid4()), "prompt": prompt, "model": model,
            "status": "pending", "phase": "discover", "created_at": now(),
            "history": [{"role": "user", "content": prompt}],
            "steps": [], "tool_calls": [], "final_text": "", "last_error": None,
        }
        self.save(state)
        return state
