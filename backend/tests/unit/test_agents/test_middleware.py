import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Union
from langchain_core.messages import ToolMessage
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from pydantic import ValidationError
from langchain_core.tools import ToolException
from langgraph.types import Command

from app.agents.middleware import TrimmingMiddleware, ToolSafetyMiddleware

class TestTrimmingMiddleware:
    @patch('app.agents.middleware.trim_messages')
    def test_wrap_model_call(self, mock_trim):
        middleware = TrimmingMiddleware(max_tokens=100)
        mock_trim.return_value = ["trimmed"]
        
        request = MagicMock(spec=ModelRequest)
        request.messages = ["a", "b"]
        
        handler = MagicMock()
        handler.return_value = "response"
        
        res = middleware.wrap_model_call(request, handler)
        
        mock_trim.assert_called_once()
        assert request.messages == ["trimmed"]
        handler.assert_called_with(request)
        assert res == "response"

    @pytest.mark.asyncio
    @patch('app.agents.middleware.trim_messages')
    async def test_awrap_model_call(self, mock_trim):
        middleware = TrimmingMiddleware(max_tokens=100)
        mock_trim.return_value = ["trimmed"]
        
        request = MagicMock(spec=ModelRequest)
        request.messages = ["a"]
        
        async def handler(req): return "async response"
        
        res = await middleware.awrap_model_call(request, handler)
        
        mock_trim.assert_called_once()
        assert request.messages == ["trimmed"]
        assert res == "async response"

class TestToolSafetyMiddleware:
    def test_truncate_if_needed(self):
        middleware = ToolSafetyMiddleware()
        middleware.MAX_TOOL_OUTPUT_CHARS = 10
        assert middleware._truncate_if_needed("short") == "short"
        
        long_str = "this is very long indeed"
        res = middleware._truncate_if_needed(long_str)
        assert "this is ve" in res
        assert "[OUTPUT TRUNCATED" in res

    def test_handle_error(self):
        middleware = ToolSafetyMiddleware()
        
        e_valid = ValidationError.from_exception_data("title", [])
        res1 = middleware._handle_error(e_valid, "tool", uses_content_and_artifact=False)
        assert "Invalid argument" in res1
        
        res1_tuple = middleware._handle_error(e_valid, "tool", uses_content_and_artifact=True)
        assert isinstance(res1_tuple, tuple)
        assert "Invalid argument" in res1_tuple[0]
        
        e_tool = ToolException("tool failed")
        res2 = middleware._handle_error(e_tool, "tool", False)
        assert "Tool execution failed" in res2
        
        e_gen = Exception("generic error")
        res3 = middleware._handle_error(e_gen, "tool", False)
        assert "Unexpected error" in res3

    @patch('app.agents.middleware._extract_paper_sections')
    def test_process_result_mcp_fetch(self, mock_extract):
        middleware = ToolSafetyMiddleware()
        mock_extract.return_value = "extracted sections"
        
        # Tuple, fetch_content, list inside
        res = middleware._process_result((["desc", "x"*10001], None), "fetch_content", True)
        assert mock_extract.called
        assert res[0] == "desc\n\nextracted sections"
        
        # Normal content tuple
        mock_extract.reset_mock()
        res2 = middleware._process_result(("short content", "art"), "tool", True)
        assert res2 == ("short content", "art")
        
        # Empty
        res3 = middleware._process_result(([], "art"), "tool", True)
        assert "No results found" in res3[0]
        
        # fetch_content str > 10000
        res4 = middleware._process_result(("y"*10001, None), "fetch_content", True)
        assert res4[0] == "extracted sections"

    def test_process_result_mcp_malformed(self):
        middleware = ToolSafetyMiddleware()
        
        # string
        res = middleware._process_result("bad content", "tool", True)
        assert "malformed response" in res[0]
        
        # tuple size 1
        res2 = middleware._process_result(("bad",), "tool", True)
        assert "malformed response" in res2[0]

        # fetching with > 10000 chars string
        res3 = middleware._process_result("z"*10001, "fetch_content", True)
        assert "malformed response" in res3[0]

    @patch('app.agents.middleware._extract_paper_sections')
    def test_process_result_standard(self, mock_extract):
        middleware = ToolSafetyMiddleware()
        mock_extract.return_value = "extracted standard"
        middleware.MAX_TOOL_OUTPUT_CHARS = 10
        
        # None
        assert "No results found" in middleware._process_result(None, "tool", False)
        
        # Normal string
        res = middleware._process_result("short", "tool", False)
        assert res == "short"
        
        # Truncate
        res2 = middleware._process_result("very long string indeed", "tool", False)
        assert "[OUTPUT TRUNCATED" in res2
        
        # fetch_content > 10000
        res3 = middleware._process_result("x"*10001, "fetch_content", False)
        assert res3 == "extracted standard"

    @patch('app.agents.middleware._extract_paper_sections')
    def test_wrap_tool_call(self, mock_extract):
        middleware = ToolSafetyMiddleware()
        middleware.MAX_TOOL_OUTPUT_CHARS = 10
        mock_extract.return_value = "extracted_wrap"
        
        req = MagicMock()
        req.tool.name = "tool"
        req.tool_call = {"id": "123"}
        
        # Returns ToolMessage < max
        def h1(r): return ToolMessage(content="short", tool_call_id="123")
        res1 = middleware.wrap_tool_call(req, h1)
        assert res1.content == "short"
        
        # Returns ToolMessage > max
        def h2(r): return ToolMessage(content="very long string here", tool_call_id="123")
        res2 = middleware.wrap_tool_call(req, h2)
        assert "[OUTPUT TRUNCATED" in res2.content
        
        # Returns fetch_content > 10000
        req.tool.name = "fetch_content"
        def h3(r): return ToolMessage(content="x"*10001, tool_call_id="123")
        res3 = middleware.wrap_tool_call(req, h3)
        assert res3.content == "extracted_wrap"
        
        # Returns non-ToolMessage
        def h4(r): return Command(resume="xy")
        res4 = middleware.wrap_tool_call(req, h4)
        assert isinstance(res4, Command)
        
        # Exception
        def h5(r): raise Exception("boom")
        res5 = middleware.wrap_tool_call(req, h5)
        assert res5.status == "error"
        assert "Unexpected error" in res5.content

    @pytest.mark.asyncio
    @patch('app.agents.middleware._extract_paper_sections')
    async def test_awrap_tool_call(self, mock_extract):
        middleware = ToolSafetyMiddleware()
        middleware.MAX_TOOL_OUTPUT_CHARS = 10
        mock_extract.return_value = "extracted_awrap"
        
        req = MagicMock()
        req.tool.name = "tool"
        req.tool_call = {"id": "123"}
        
        async def h1(r): return ToolMessage(content="short", tool_call_id="123")
        res1 = await middleware.awrap_tool_call(req, h1)
        assert res1.content == "short"
        
        async def h2(r): return ToolMessage(content="very long data over max", tool_call_id="123")
        res2 = await middleware.awrap_tool_call(req, h2)
        assert "[OUTPUT" in res2.content
        
        req.tool.name = "fetch_content"
        async def h3(r): return ToolMessage(content="x"*10001, tool_call_id="123")
        res3 = await middleware.awrap_tool_call(req, h3)
        assert res3.content == "extracted_awrap"
        
        async def h4(r): raise ToolException("tool boom")
        res4 = await middleware.awrap_tool_call(req, h4)
        assert res4.status == "error"
        assert "tool boom" in res4.content
