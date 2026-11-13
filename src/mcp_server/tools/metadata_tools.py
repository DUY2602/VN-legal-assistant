from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METADATA_CSV = BASE_DIR / "data" / "metadata" / "vbpl_master.csv"

def load_metadata_df() -> pd.DataFrame:
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    return pd.DataFrame()

def register_metadata_tools(mcp):
    """Register tools for searching catalog and legal validity of documents on MCP Server."""

    @mcp.tool()
    def list_available_laws(keyword: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List legal documents in metadata master catalog.
        Can filter by keyword (e.g., "intellectual property", "contract", "technology").
        """
        df = load_metadata_df()
        if df.empty:
            return []

        if keyword:
            kw = keyword.lower()
            mask = (
                df["Số ký hiệu"].astype(str).str.lower().str.contains(kw) |
                df["Tên văn bản"].astype(str).str.lower().str.contains(kw) |
                df["Lĩnh vực"].astype(str).str.lower().str.contains(kw)
            )
            filtered = df[mask]
        else:
            filtered = df

        results = []
        for _, row in filtered.head(limit).iterrows():
            results.append({
                "id": str(row.get("ID", "")),
                "doc_number": str(row.get("Số ký hiệu", "")),
                "title": str(row.get("Tên văn bản", "")),
                "validity_status": str(row.get("Tình trạng hiệu lực", "")),
                "effective_date": str(row.get("Ngày có hiệu lực", "")),
                "review_status": str(row.get("review_status", ""))
            })
        return results

    @mcp.tool()
    def check_document_validity(doc_number: str) -> Dict[str, Any]:
        """
        Quickly check validity status and issuance information of a specific document by document number.
        Example: '50/2005/QH11', '07/2022/QH15'.
        """
        df = load_metadata_df()
        if df.empty:
            return {"status": "error", "message": "Metadata data not found"}

        clean_no = doc_number.strip().lower()
        match = df[df["Số ký hiệu"].astype(str).str.strip().str.lower() == clean_no]
        
        if match.empty:
            return {
                "status": "not_found",
                "message": f"Validity information not found for document number: {doc_number}"
            }

        row = match.iloc[0]
        return {
            "status": "found",
            "doc_number": str(row.get("Số ký hiệu", "")),
            "title": str(row.get("Tên văn bản", "")),
            "issuer": str(row.get("Cơ quan ban hành", "")),
            "validity_status": str(row.get("Tình trạng hiệu lực", "")),
            "issue_date": str(row.get("Ngày ban hành", "")),
            "effective_date": str(row.get("Ngày có hiệu lực", "")),
            "affected_documents_count": str(row.get("Số lượng văn bản bị tác động hiệu lực", "0"))
        }

