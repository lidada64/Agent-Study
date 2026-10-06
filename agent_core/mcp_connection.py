"""MCP 1.x/2.x stdio connection with version-appropriate initialization."""

from contextlib import asynccontextmanager
from datetime import timedelta

import mcp


@asynccontextmanager
async def connect_mcp(parameters, read_timeout_seconds=120):
    if hasattr(mcp, "Client"):
        # SDK 2.x owns the transport and initialization handshake.
        async with mcp.Client(parameters, read_timeout_seconds=read_timeout_seconds) as client:
            yield client
    else:
        # SDK 1.x requires an explicit stdio transport and session handshake.
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(parameters) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=read_timeout_seconds),
            ) as session:
                await session.initialize()
                yield session
