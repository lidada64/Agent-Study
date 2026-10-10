from pathlib import Path

from .config import ROOT

LOCAL_TOOLS = [
    {
        "type": "function", "name": "find_file",
        "description": "Discover real files in the workspace by name or glob and return absolute paths. Call this before find_text so the keyword search uses actual discovered paths.",
        "parameters": {
            "type": "object", "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "type": "function", "name": "find_text",
        "description": "Search a keyword inside real file contents and return matching paths and line numbers. Only paths returned by a completed find_file call are accepted; this is not the downloaded Skill library search.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"},
            },
            "required": ["paths", "text"],
        },
    },
]


def find_file(name):
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    return sorted(str(path.resolve()) for path in ROOT.rglob(name) if path.is_file())


def find_text(paths, text):
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not isinstance(paths, list):
        raise TypeError("paths must be a list")
    locations = []
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                if text in line:
                    locations.append({"path": str(path), "line_number": line_number})
    return locations
