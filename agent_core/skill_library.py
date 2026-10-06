"""Download MCP Skill documents into one searchable, idempotent JSON library."""

import json
from pathlib import Path

from .config import DEFAULT_LIBRARY_PATH
from .mcp_tools import execute_tool
from .state import atomic_write_json, now


LIBRARY_TOOLS = [
    {
        "type": "function", "name": "download_skill",
        "description": "Download a full SKILL.md through MCP and save it in the local library. Repeated ids reuse the saved document. This does not install or execute a skill.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False},
    },
    {
        "type": "function", "name": "search_saved_skills",
        "description": "Search downloaded skills by keywords in their id, name and full document. An empty query lists saved skills.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
    },
    {
        "type": "function", "name": "read_saved_skill",
        "description": "Read the full document of a downloaded skill by its exact id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False},
    },
]


class SkillLibrary:
    def __init__(self, path=DEFAULT_LIBRARY_PATH):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return {"version": 1, "skills": {}}
        library = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(library, dict) or library.get("version") != 1 or not isinstance(library.get("skills"), dict):
            raise ValueError("Invalid Skill library; refusing to overwrite it")
        for identifier, skill in library["skills"].items():
            if not isinstance(skill, dict) or skill.get("id") != identifier or not all(
                isinstance(skill.get(key), str) for key in ("name", "content", "saved_at")
            ):
                raise ValueError(f"Invalid saved Skill: {identifier}")
        return library

    def summary(self, skill):
        return {key: skill[key] for key in ("id", "name", "saved_at")} | {"file": str(self.path.resolve())}

    def search(self, query):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        keywords = query.casefold().split()
        matches = []
        for skill in self.load()["skills"].values():
            haystack = "\n".join(skill[key] for key in ("id", "name", "content")).casefold()
            if all(keyword in haystack for keyword in keywords):
                matches.append(self.summary(skill))
        return {"ok": True, "data": matches, "count": len(matches)}

    def read(self, identifier):
        self.validate_id(identifier)
        skill = self.load()["skills"].get(identifier)
        if skill is None:
            raise ValueError(f"Skill is not saved: {identifier}")
        return {"ok": True, "data": skill}

    @staticmethod
    def validate_id(identifier):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("id must be a non-empty string")

    async def download(self, identifier, mcp_client, mcp_tools):
        self.validate_id(identifier)
        library = self.load()
        if identifier in library["skills"]:
            return {"ok": True, "data": self.summary(library["skills"][identifier]), "already_saved": True}
        if "skills_get_skill" not in mcp_tools:
            raise ValueError("The Skill MCP server does not expose get_skill")
        result = await execute_tool("skills_get_skill", {"id": identifier}, mcp_client, mcp_tools)
        if not result["ok"]:
            return result
        # The upstream server returns Markdown with its source header in text blocks.
        # Preserve that entire document; never let the model invent saved content.
        content = "\n\n".join(
            block["text"] for block in result["data"].get("content", [])
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        if not content.strip():
            raise ValueError("get_skill returned no Skill document; nothing was saved")
        name = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")), identifier)
        skill = {"id": identifier, "name": name, "content": content, "saved_at": now()}
        library["skills"][identifier] = skill
        atomic_write_json(self.path, library)
        return {"ok": True, "data": self.summary(skill), "already_saved": False}

    async def execute(self, name, arguments, mcp_client, mcp_tools):
        if name == "download_skill":
            return await self.download(arguments["id"], mcp_client, mcp_tools)
        if name == "search_saved_skills":
            return self.search(arguments["query"])
        if name == "read_saved_skill":
            return self.read(arguments["id"])
        return await execute_tool(name, arguments, mcp_client, mcp_tools)
