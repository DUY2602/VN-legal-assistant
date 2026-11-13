# Vietnam Legal Assistant - Capstone Project

Legal & Internal Document Support System (RAG Assistant) for Katalon Legal Department.

## 1. Data Directory Structure & Current Ingestion

```text
VN-legal-assistant/
├── data/
│   ├── metadata/
│   │   └── vbpl_master.csv           # Vietnamese legal document catalog (Source of Truth)
│   ├── raw_pdf/                      # Contains PDF files downloaded from vbpl.vn
│   └── parsed/                       # Contains text/JSON after extracting Articles/Clauses
│
├── src/
│   └── ingestion/
│       ├── crawler/
│       │   └── vbpl_crawler.py       # Module to search and download documents from vbpl.vn
│       └── scripts/
│           └── crawl_all_pdf.py      # Script to batch download all PDFs by catalog
│
└── requirements.txt                  # Basic dependencies
```

## 2. Crawler PDF Execution Guide

1. **Install libraries:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run crawl PDF:**
   ```bash
   python -m src.ingestion.scripts.crawl_all_pdf
   ```
   *By default, the script filters documents with `review_status == "include"` column and saves files to `data/raw_pdf/`.*

## 3. Create Knowledge Graph with LightRAG

LightRAG uses LLM to extract entities and semantic relationships from full texts in `data/raw_text/`. By default, Gemini is used for extraction and BGE-M3 runs locally for embedding. Enter the API key from Google AI Studio into `GEMINI_API_KEY` in the `.env` file (this file is excluded from Git).

```bash
# Open .env and set GEMINI_API_KEY=...; then:
python -m src.visualization.lightrag_graph --ingest
```

By default, `gemini-3.5-flash-lite` is used for extraction via Gemini's OpenAI-compatible endpoint and `BAAI/bge-m3` (1024 dims) for local embedding via FlagEmbedding; GPU is used automatically if available. Since embedding runs locally, no Gemini embedding quota is consumed. If the model is not in cache, the first run will download the model from Hugging Face. You can change `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_EMBEDDING_PROVIDER`, `LIGHTRAG_EMBEDDING_MODEL`, `LIGHTRAG_EMBEDDING_DIMENSION`, and `GEMINI_OPENAI_BASE_URL` in `.env`. Changing embedding provider/model requires creating a new index because vectors have different dimensions. Try one document first:

```bash
python -m src.visualization.lightrag_graph --ingest --limit 1
```

After ingestion is complete, run the official LightRAG knowledge graph WebUI:

```bash
python -m src.visualization.lightrag_webui
```

Open `http://127.0.0.1:9621` to view the graph. The server binds to localhost by default and uses the index directly in `data/lightrag_storage/;` no re-ingestion or Gemini calls just to display the graph. WebUI uses Ollama with model `bge-m3` for search/query operations; if using those operations, install the model with `ollama pull bge-m3` and ensure Ollama is running. You can change port or bind address with `--port` and `--host`.

You can export a separate HTML viewer (not the official WebUI) from the existing index without calling LLM:

```bash
python -m src.visualization.lightrag_graph
```

When running `--ingest`, LightRAG records in failed status (except records identified as duplicate) will be retried document by document before ingesting new documents. Gemini rate limit errors are retried with exponential backoff; fixed quota/billing errors will be reported clearly instead of retrying indefinitely. If you only want to retry failed documents after quota recovers without ingesting new files:

```bash
python -m src.visualization.lightrag_graph --retry-failed-only
```

Ingesting all documents will send content to Gemini and may incur costs. Please check the provider's data policy and fees before running. The graph may vary by model and does not replace legal verification. The direct reference graph by document number (without LLM) can also be created separately with `python -m src.visualization.legal_graph`, saved at `data/legal_reference_graph.html`.
