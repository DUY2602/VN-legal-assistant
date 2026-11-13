import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METADATA_CSV = BASE_DIR / "data" / "metadata" / "vbpl_master.csv"
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"
RAW_HTML_DIR = BASE_DIR / "data" / "raw_html"
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdf"

class VBPLCrawlerEngine:
    """Engine responsible for connecting and extracting data from vbpl.vn (Next.js SSR)."""
    BASE_URL = "https://vbpl.vn"
    DETAIL_URL = "https://vbpl.vn/van-ban/chi-tiet/{doc_id}"

    def __init__(self, timeout: int = 20, delay_seconds: float = 0.3):
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def fetch_document(self, doc_id: str) -> Dict[str, Any]:
        url = self.DETAIL_URL.format(doc_id=doc_id.strip())
        res = {"title": "", "html": "", "text": "", "pdf_links": [], "success": False, "error": None}
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                res["error"] = f"HTTP {resp.status_code}"
                return res

            raw_html = resp.text
            soup = BeautifulSoup(resp.content, "html.parser")
            if soup.title and soup.title.string:
                res["title"] = soup.title.string.strip()

            # Extract full text from Next.js payload (__next_f)
            matches = re.findall(r'self\.__next_f\.push\(\[1,\"(.*?)\"\]\)', raw_html)
            full_html = ""
            for m in matches:
                if "\\u003chtml" in m or "<html" in m or "\\u003cdiv" in m or "Điều 1" in m:
                    try:
                        full_html += json.loads(f'"{m}"')
                    except Exception:
                        full_html += m

            if full_html:
                content_soup = BeautifulSoup(full_html, "html.parser")
                res["html"] = full_html
                res["text"] = content_soup.get_text(separator="\n", strip=True)
                res["success"] = True
            else:
                body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                if len(body_text) > 500:
                    res["text"] = body_text
                    res["success"] = True
                else:
                    res["error"] = "Document content not found"

            time.sleep(self.delay_seconds)
            return res
        except Exception as e:
            res["error"] = str(e)
            return res

    def download_pdf(self, url: str, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            if path.exists():
                path.unlink()
            return False

crawler_engine = VBPLCrawlerEngine()

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name).strip()

def load_metadata_df() -> pd.DataFrame:
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    return pd.DataFrame()

def register_crawler_tools(mcp):
    """Đăng ký các tools crawler vào MCP Server."""

    @mcp.tool()
    def crawl_vbpl_document(doc_identifier: str) -> Dict[str, Any]:
        """
        Thu thập toàn văn văn bản pháp luật từ vbpl.vn theo Số ký hiệu (VD: '50/2005/QH11', '186/2026/NĐ-CP') 
        hoặc ID trên vbpl.vn. Dữ liệu tải về sẽ được lưu vào data/raw_text/ và data/raw_html/.
        """
        df = load_metadata_df()
        doc_id = None
        doc_no = None

        if not df.empty:
            match_no = df[df["Số ký hiệu"].astype(str).str.strip().str.lower() == doc_identifier.strip().lower()]
            if not match_no.empty:
                doc_id = str(match_no.iloc[0]["ID"]).strip()
                doc_no = str(match_no.iloc[0]["Số ký hiệu"]).strip()
            else:
                match_id = df[df["ID"].astype(str).str.strip() == doc_identifier.strip()]
                if not match_id.empty:
                    doc_id = str(match_id.iloc[0]["ID"]).strip()
                    doc_no = str(match_id.iloc[0]["Số ký hiệu"]).strip()

        if not doc_id:
            doc_id = doc_identifier.strip()
            doc_no = doc_identifier.strip()

        RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

        base_name = sanitize_filename(doc_no if doc_no and doc_no.lower() != "không số" else doc_id)
        text_file = RAW_TEXT_DIR / f"{base_name}.txt"
        html_file = RAW_HTML_DIR / f"{base_name}.html"

        result = crawler_engine.fetch_document(doc_id)
        if not result["success"]:
            return {"status": "error", "message": f"Error crawling document {doc_identifier}: {result.get('error')}"}

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(result["text"])

        if result["html"]:
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(result["html"])

        return {
            "status": "success",
            "doc_number": doc_no,
            "title": result["title"],
            "saved_text_path": str(text_file.relative_to(BASE_DIR)),
            "text_length": len(result["text"]),
            "preview": result["text"][:300] + "..."
        }

    @mcp.tool()
    def crawl_all_vbpl_documents(status_filter: str = "include", max_items: Optional[int] = None) -> Dict[str, Any]:
        """
        Trigger crawl of all documents in master CSV file from vbpl.vn (filter by review_status, default "include").
        Full text and html will be saved to data/raw_text/ and data/raw_html/.
        """
        df = load_metadata_df()
        if df.empty:
            return {"status": "error", "message": "Không tìm thấy file metadata CSV"}

        if status_filter and "review_status" in df.columns:
            targets = df[df["review_status"] == status_filter]
        else:
            targets = df

        if max_items:
            targets = targets.head(max_items)

        RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

        success_count = 0
        skipped_count = 0
        failed_count = 0
        failed_docs = []

        for _, row in targets.iterrows():
            doc_id = str(row.get("ID", "")).strip()
            doc_no = str(row.get("Số ký hiệu", "")).strip()
            base_name = sanitize_filename(doc_no if doc_no and doc_no.lower() != "không số" else doc_id)

            text_file = RAW_TEXT_DIR / f"{base_name}.txt"
            html_file = RAW_HTML_DIR / f"{base_name}.html"

            if text_file.exists() and text_file.stat().st_size > 500:
                skipped_count += 1
                continue

            res = crawler_engine.fetch_document(doc_id)
            if res["success"]:
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(res["text"])
                if res["html"]:
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(res["html"])
                success_count += 1
            else:
                failed_count += 1
                failed_docs.append(f"{doc_no} (ID: {doc_id})")

        return {
            "status": "completed",
            "total_targets": len(targets),
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "failed_docs": failed_docs
        }

