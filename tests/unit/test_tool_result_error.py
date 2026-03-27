import pytest

from pyghidra_mcp.tools import ToolResultError, handle_exceptions


class TestToolResultError:
    """Tests for the ToolResultError exception class."""

    def test_error_without_suggestions(self):
        """ToolResultError with only a message."""
        err = ToolResultError("Function 'foo' not found.")
        assert err.message == "Function 'foo' not found."
        assert err.suggestions == []
        assert str(err) == "Function 'foo' not found."

    def test_error_with_suggestions(self):
        """ToolResultError with message and suggestions."""
        suggestions = ["sub_140002C50 @ 0x140002C50", "sub_140002C60 @ 0x140002C60"]
        err = ToolResultError("Function 'sub_140002C58' not found.", suggestions=suggestions)
        assert err.message == "Function 'sub_140002C58' not found."
        assert err.suggestions == suggestions

    def test_suggestions_default_to_empty_list(self):
        """Suggestions default to empty list when None is passed."""
        err = ToolResultError("Symbol 'bar' not found.", suggestions=None)
        assert err.suggestions == []

    def test_is_exception(self):
        """ToolResultError can be raised and caught as Exception."""
        with pytest.raises(ToolResultError) as exc_info:
            raise ToolResultError("test error")
        assert exc_info.value.message == "test error"

    def test_does_not_match_value_error(self):
        """ToolResultError is NOT a ValueError subclass."""
        err = ToolResultError("test")
        assert not isinstance(err, ValueError)


class TestHandleExceptions:
    """Tests for the handle_exceptions decorator."""

    def test_tool_result_error_propagates_unchanged(self):
        """ToolResultError should propagate through the decorator unchanged."""

        @handle_exceptions
        def raise_tool_error():
            raise ToolResultError("user-facing error", suggestions=["alt"])

        with pytest.raises(ToolResultError) as exc_info:
            raise_tool_error()

        assert exc_info.value.message == "user-facing error"
        assert exc_info.value.suggestions == ["alt"]

    def test_generic_exception_propagates_unchanged(self):
        """Generic exceptions should propagate through the decorator (after logging)."""

        @handle_exceptions
        def raise_runtime_error():
            raise RuntimeError("internal error")

        with pytest.raises(RuntimeError, match="internal error"):
            raise_runtime_error()
