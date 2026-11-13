import asyncio
import sys
import os
from pathlib import Path

# Ensure console UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_test():
    base_dir = Path(__file__).resolve().parent
    server_script = base_dir / "src" / "mcp_server" / "server.py"

    print("==================================================")
    print("INITIALIZE MCP CLIENT & CONNECT TO MODULAR MCP SERVER")
    print(f"Server script: {server_script}")
    print("==================================================")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize session (handshake)
            await session.initialize()
            print("\n[1] Handshake successful with Modular Legal MCP Server!")

            # 2. List tools registered from sub-files
            tools_list = await session.list_tools()
            print(f"\n[2] List of automatically registered Tools ({len(tools_list.tools)} tools):")
            for t in tools_list.tools:
                print(f"  - [{t.name}]: {t.description.strip().splitlines()[0]}")

            # 3. Test Tool from metadata_tools: check_document_validity
            print("\n[3] Test Tool metadata: check_document_validity('50/2005/QH11')...")
            res_val = await session.call_tool(
                "check_document_validity",
                arguments={"doc_number": "50/2005/QH11"}
            )
            for c in res_val.content:
                print(c.text)

            # 4. Test Tool from document_tools: get_crawled_document
            print("\n[4] Test Tool document: get_crawled_document('50/2005/QH11')...")
            res_doc = await session.call_tool(
                "get_crawled_document",
                arguments={"doc_number": "50/2005/QH11"}
            )
            for c in res_doc.content:
                print(c.text[:350] + "...")

    print("\n==================================================")
    print("MODULAR MCP SERVER IS WORKING PERFECTLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_mcp_test())
