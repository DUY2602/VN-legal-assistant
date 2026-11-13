import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name).strip()

def register_legal_resources(mcp):
    """Register MCP Resource URIs to the server."""

    @mcp.resource("legal://documents/{doc_number}")
    def document_resource(doc_number: str) -> str:
        """
        MCP Resource that allows clients to read full document text directly via URI legal://documents/{doc_number}.
        """
        base_name = sanitize_filename(doc_number.strip())
        text_file = RAW_TEXT_DIR / f"{base_name}.txt"
        if text_file.exists():
            with open(text_file, "r", encoding="utf-8") as f:
                return f.read()
        return f"Document {doc_number} is not crawled yet."

