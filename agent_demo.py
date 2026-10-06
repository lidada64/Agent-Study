"""Interactive Skill library with MCP, conversation checkpoints and compaction.

Run: python code/agent_demo.py
Resume: python code/agent_demo.py --resume
Legacy search: python code/agent_demo.py --task "Find 1.txt and search for lidada"
MCP-only check: python code/agent_demo.py --check-mcp
Dependencies: mcp>=1.30,<3, openai, python-dotenv; Node.js/npx.
"""

# Keep existing imports compatible; implementation lives in agent_core.
from agent_core.cli import entrypoint, main
from agent_core.config import ROOT, SKILLS_PACKAGE, SKILL_TOOLS
from agent_core.local_tools import LOCAL_TOOLS, find_file, find_text
from agent_core.mcp_tools import discover_skill_tools, execute_tool, function_schema, skills_server_params
from agent_core.runner import run_agent
from agent_core.conversation import SkillConversation


if __name__ == "__main__":
    raise SystemExit(entrypoint())
