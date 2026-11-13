import os
import sys
from pathlib import Path

# Ensure UTF-8 for console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure BASE_DIR is in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.mcpserver import MCPServer

from src.mcp_server.tools.metadata_tools import register_metadata_tools
from src.mcp_server.tools.crawler_tools import register_crawler_tools
from src.mcp_server.tools.document_tools import register_document_tools
from src.mcp_server.resources.legal_resources import register_legal_resources

# Initialize MCP Server
mcp = MCPServer(
    name="Vietnam Legal & Katalon Assistant",
    instructions="MCP Server specialized in managing, searching, and collecting Vietnamese legal documents from vbpl.vn."
)

# Register modules
register_metadata_tools(mcp)
register_crawler_tools(mcp)
register_document_tools(mcp)
register_legal_resources(mcp)

if __name__ == "__main__":
    mcp.run()
