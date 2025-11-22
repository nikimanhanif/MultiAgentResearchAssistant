"""API routes for the research assistant backend.

Future implementation:
- POST /api/v1/chat - Main chat endpoint
  - Handle multi-turn clarification flow
  - Route to scope agent, research agent, or report agent based on conversation state
  - Return clarification questions, research progress, or final report
  
- POST /api/v1/config/mcp-servers - Update MCP server configuration
  - Accept list of enabled MCP servers from frontend
  - Store in conversation context or user session
  
- GET /api/v1/config/mcp-servers - Get available MCP servers
  - Return list of configured MCP servers and their status
"""

