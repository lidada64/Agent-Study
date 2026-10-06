"""Run with: python -m unittest discover -s code -p test_agent_demo.py"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp import StdioServerParameters
from mcp.types import CallToolResult, ListToolsResult, Tool

from agent_demo import discover_skill_tools, execute_tool, function_schema, run_agent
from agent_core.mcp_connection import connect_mcp


class AgentDemoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state_path = Path(self.directory.name) / "state.json"

    async def test_sdk_search_and_schema(self):
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).parent / "test_fixtures" / "skills_server.py")],
        )
        async with connect_mcp(parameters) as client:
            discovered = await discover_skill_tools(client)
            schema = function_schema(discovered[0])
            self.assertEqual(schema["name"], "skills_search_skills")
            self.assertIn("query", schema["parameters"]["required"])
            result = await execute_tool(
                schema["name"], {"query": "python", "limit": 1},
                client, {schema["name"]: "search_skills"},
            )
            self.assertTrue(result["ok"])
            self.assertIn("python-testing", json.dumps(result))

    async def test_pagination_and_no_install_tool(self):
        client = SimpleNamespace(list_tools=AsyncMock(side_effect=[
            ListToolsResult(tools=[Tool(name="install_skill", inputSchema={})], nextCursor="page2"),
            ListToolsResult(tools=[Tool(name="search_skills", inputSchema={})]),
        ]))
        tools = await discover_skill_tools(client)
        self.assertEqual([tool.name for tool in tools], ["search_skills"])
        self.assertEqual(client.list_tools.call_args.kwargs, {"cursor": "page2"})

    async def test_server_error_is_not_success(self):
        client = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[], isError=True)))
        result = await execute_tool("skills_search_skills", {}, client, {"skills_search_skills": "search_skills"})
        self.assertFalse(result["ok"])

    async def test_loop_returns_all_results_before_next_model_request(self):
        calls = [
            SimpleNamespace(type="function_call", name="skills_search_skills", arguments='{"query":"python"}', call_id="a"),
            SimpleNamespace(type="function_call", name="find_text", arguments="[]", call_id="b"),
        ]
        captured = []

        async def respond(**kwargs):
            captured.append(list(kwargs["input"]))
            if len(captured) == 1:
                return SimpleNamespace(output=calls, output_text="")
            return SimpleNamespace(output=[], output_text="done")

        model = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=respond)))
        client = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(content=[])))
        status, text = await run_agent(
            model, client, [Tool(name="search_skills", inputSchema={})], "search", "test", 3,
            state_path=self.state_path,
        )
        self.assertEqual((status, text), ("completed", "done"))
        outputs = captured[1][-2:]
        self.assertEqual([output["call_id"] for output in outputs], ["a", "b"])
        self.assertTrue(json.loads(outputs[0]["output"])["ok"])
        self.assertFalse(json.loads(outputs[1]["output"])["ok"])
        client.call_tool.assert_awaited_once_with("search_skills", arguments={"query": "python"})


if __name__ == "__main__":
    unittest.main()
