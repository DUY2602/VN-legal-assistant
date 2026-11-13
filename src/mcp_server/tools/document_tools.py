import re
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name).strip()

def register_document_tools(mcp):
    """Register tools for reading and checking stored document content on MCP Server."""

    @mcp.tool()
    def get_crawled_document(doc_number: str) -> Dict[str, Any]:
        """
        Read full text of downloaded and stored document by document number.
        Example: '50/2005/QH11', '186/2026/NĐ-CP'.
        """
        base_name = sanitize_filename(doc_number.strip())
        text_file = RAW_TEXT_DIR / f"{base_name}.txt"

        if not text_file.exists():
            return {
                "status": "not_found",
                "message": f"Document {doc_number} has not been downloaded. Please use tool crawl_vbpl_document to download first."
            }

        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "status": "success",
            "doc_number": doc_number,
            "length": len(content),
            "content": content
        }

