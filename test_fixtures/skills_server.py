"""Local stdio MCP server for both SDK versions; requires no external services."""

import mcp

if hasattr(mcp, "Client"):
    from mcp.server import MCPServer
    server = MCPServer("test-skills")
else:
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("test-skills")


@server.tool()
def search_skills(query: str, limit: int = 3) -> dict:
    """Search the test skill catalog."""
    return {"skills": [{"id": "python-testing", "query": query}][:limit]}


@server.tool()
def get_skill(id: str) -> str:
    """Fetch a full test Skill document."""
    if id != "python-testing":
        raise ValueError(f"Skill not found: {id}")
    return "# Python Testing\nSource: offline fixture\n\n---\n\n---\nname: python-testing\ndescription: Python pytest testing\n---\n\nWrite meaningful pytest tests."


if __name__ == "__main__":
    server.run()
