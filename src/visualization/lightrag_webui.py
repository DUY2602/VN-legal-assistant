import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
WORKING_DIR = BASE_DIR / "data" / "lightrag_storage"
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"


def _validate_index() -> None:
    graph_files = list(WORKING_DIR.glob("graph_chunk_entity_relation*.graphml"))
    if not graph_files:
        raise RuntimeError(
            f"LightRAG index not found in {WORKING_DIR}. "
            "Please ingest documents first."
        )

    status_path = WORKING_DIR / "kv_store_doc_status.json"
    if status_path.exists():
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(status_data, dict) and any(
            isinstance(entry, dict) and entry.get("status") == "processing"
            for entry in status_data.values()
        ):
            raise RuntimeError(
                "LightRAG is still ingesting documents. Please wait for ingestion to complete "
                "before opening WebUI."
            )


def _configure_server() -> None:
    load_dotenv(BASE_DIR / ".env", override=False)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured in .env file.")

    provider = os.getenv("LIGHTRAG_EMBEDDING_PROVIDER", "local").strip().lower()
    model = os.getenv("LIGHTRAG_EMBEDDING_MODEL", "BAAI/bge-m3").strip().lower()
    dimension = os.getenv("LIGHTRAG_EMBEDDING_DIMENSION", "1024").strip()
    if (
        provider != "local"
        or model not in {"baai/bge-m3", "bge-m3"}
        or dimension != "1024"
    ):
        raise RuntimeError(
            "WebUI is currently configured for local BGE-M3 1024-dim index. "
            "Embedding configuration in .env does not match."
        )

    os.environ.update(
        {
            "LLM_BINDING": "gemini",
            "LLM_BINDING_API_KEY": api_key,
            "LLM_MODEL": os.getenv(
                "LIGHTRAG_LLM_MODEL", "gemini-3.5-flash-lite"
            ),
            "EMBEDDING_BINDING": "ollama",
            "EMBEDDING_BINDING_HOST": os.getenv(
                "LIGHTRAG_WEBUI_OLLAMA_HOST", "http://localhost:11434"
            ),
            "EMBEDDING_MODEL": "bge-m3",
            "EMBEDDING_DIM": "1024",
            "WORKSPACE": "",
            "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
            "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
            "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
        }
    )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Run official LightRAG WebUI on existing knowledge graph."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server bind address (default: access only on this machine).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9621,
        help="LightRAG WebUI port.",
    )
    args = parser.parse_args()

    _validate_index()
    _configure_server()
    sys.argv = [
        "lightrag-server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--working-dir",
        str(WORKING_DIR),
        "--input-dir",
        str(RAW_TEXT_DIR),
        "--default-ui",
        "webui",
        "--llm-binding",
        "gemini",
        "--embedding-binding",
        "ollama",
    ]

    from lightrag.api.lightrag_server import main as lightrag_server_main

    lightrag_server_main()


if __name__ == "__main__":
    main()
