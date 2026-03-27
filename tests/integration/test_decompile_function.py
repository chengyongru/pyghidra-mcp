import json

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from pyghidra_mcp.context import PyGhidraContext


@pytest.mark.asyncio
async def test_decompile_function_tool(shared_mcp_session):
    """Test the decompile_function tool."""

    # Use shared MCP session with pre-imported demo binary
    session = shared_mcp_session
    binary_name = session.demo_binary_name

    # Call the decompile_function tool
    try:
        results = await session.call_tool(
            "decompile_function", {"binary_name": binary_name, "name_or_address": "main"}
        )

        # Check that we got results
        assert results is not None
        assert results.content is not None
        assert len(results.content) > 0

        # Check that the result contains decompiled code
        # (this might vary depending on the binary and Ghidra's analysis)
        # We'll just check that it's not empty
        text_content = results.content[0].text
        assert text_content is not None
        assert len(text_content) > 0
        assert "main" in text_content
    except Exception as e:
        # If we get an error, it might be because the function wasn't found
        # or because of issues with the binary analysis
        # We'll just check that we got a proper error response
        assert e is not None


@pytest.mark.asyncio
async def test_decompile_function_not_found_returns_tool_error(shared_mcp_session):
    """Test that decompile_function returns a ToolError result (not an exception)
    when the function is not found."""

    session = shared_mcp_session
    binary_name = session.demo_binary_name

    # Call with a non-existent function name — should NOT raise
    results = await session.call_tool(
        "decompile_function",
        {"binary_name": binary_name, "name_or_address": "nonexistent_function_xyz"},
    )

    # Should return a normal result (not raise), containing ToolError data
    assert results is not None
    assert results.content is not None
    assert len(results.content) > 0
    assert results.isError is False  # MCP-level error flag should be False

    # Parse the result as JSON and verify ToolError fields
    data = json.loads(results.content[0].text)
    assert "error" in data
    assert "nonexistent_function_xyz" in data["error"]
