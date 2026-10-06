import asyncio
import os

from mcp import StdioServerParameters

from .config import SKILL_TOOLS, SKILLS_PACKAGE
from .local_tools import find_file, find_text


def skills_server_params():
    args = ["-y", SKILLS_PACKAGE]
    if os.name == "nt":
        return StdioServerParameters(command="cmd", args=["/c", "npx", *args])
    return StdioServerParameters(command="npx", args=args)


async def discover_skill_tools(mcp_client):
    tools, cursor = [], None
    while True:
        page = await mcp_client.list_tools(cursor=cursor)
        tools.extend(tool for tool in page.tools if tool.name in SKILL_TOOLS)
        cursor = page.model_dump(by_alias=True).get("nextCursor")
        if cursor is None:
            break
    if not any(tool.name == "search_skills" for tool in tools):
        raise RuntimeError("The MCP server did not expose search_skills")
    return tools


def function_schema(tool):
    return {
        "type": "function", "name": f"skills_{tool.name}",
        "description": tool.description or tool.name,
        "parameters": tool.model_dump(by_alias=True)["inputSchema"],
    }


async def execute_tool(name, arguments, mcp_client, mcp_tools):
    if name == "find_file":
        return {"ok": True, "data": await asyncio.to_thread(find_file, arguments["name"])}
    if name == "find_text":
        return {"ok": True, "data": await asyncio.to_thread(find_text, arguments["paths"], arguments["text"])}
    if name not in mcp_tools:
        raise ValueError(f"Unknown tool: {name}")
    result = await mcp_client.call_tool(mcp_tools[name], arguments=arguments)
    data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {"ok": not data.get("isError", False), "data": data}
