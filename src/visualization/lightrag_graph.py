import argparse
import asyncio
import csv
import os
import re
import sys
import unicodedata
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import RateLimitError

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)

from google import genai
from google.genai import types
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.llm._error_utils import is_permanent_rate_limit_error
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

from src.visualization.legal_graph import render_html


METADATA_CSV = BASE_DIR / "data" / "metadata" / "vbpl_master.csv"
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"
WORKING_DIR = BASE_DIR / "data" / "lightrag_storage"
DEFAULT_OUTPUT = BASE_DIR / "data" / "legal_graph.html"
DEFAULT_PROVIDER = "gemini"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIMENSION = 1024
EMBEDDING_INPUT_INTERVAL_SECONDS = 0.8
LLM_MIN_INTERVAL_SECONDS = 1.0
LLM_RATE_LIMIT_MAX_RETRIES = 4
LLM_RATE_LIMIT_INITIAL_DELAY_SECONDS = 30.0
ENTITY_TYPES_GUIDANCE = (
    "Extract legal entities and relationships directly based on the document text. "
    "Prioritize types: legal document, article and clause, issuing authority, organization, "
    "subject, right, obligation, violation behavior, penalty, procedure and legal concept. "
    "Entity type name uses only letters, no slashes or commas. "
    "Entity name should keep document number or Article/Clause number when available. "
    "Describe relationship briefly in Vietnamese; do not speculate on effectiveness or relationship "
    "not stated in the source."
)
ENTITY_TYPE_LABELS = {
    "unknown": "Unclassified",
    "vănbảnphápluật": "Legal Document",
    "vănbản": "Document",
    "điềuvàkhoản": "Article and Clause",
    "điềuandkhoản": "Article and Clause",
    "điểuvàkhoản": "Point and Clause",
    "điềukhoản": "Terms",
    "cơquanbanhành": "Issuing Authority",
    "cơquannhànước": "State Agency",
    "tổchức": "Organization",
    "chủthể": "Subject",
    "quyền": "Right",
    "nghĩavụ": "Obligation",
    "hànhviphạm": "Violation Behavior",
    "hànhviviphạm": "Violation Behavior",
    "chếtài": "Penalty",
    "thủtục": "Procedure",
    "kháiniệmpháplý": "Legal Concept",
    "tàisảntrítuệ": "Intellectual Property",
}


def load_documents() -> List[dict]:
    with METADATA_CSV.open(encoding="utf-8-sig", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))

    def filename_stem(number: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "_", number).strip().casefold()

    documents_by_stem = {
        filename_stem((row.get("Số ký hiệu") or "").strip()): row
        for row in rows
        if (row.get("Số ký hiệu") or "").strip()
    }

    documents = []
    for path in sorted(RAW_TEXT_DIR.glob("*.txt")):
        metadata = documents_by_stem.get(path.stem.casefold(), {})
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        header = "\n".join(
            value
            for value in (
                f"Document number: {metadata.get('Số ký hiệu', path.stem)}",
                f"Document title: {metadata.get('Tên văn bản', '')}",
                f"Document type: {metadata.get('Loại văn bản', '')}",
                f"Issuing authority: {metadata.get('Cơ quan ban hành', '')}",
            )
            if value.split(": ", 1)[-1]
        )
        documents.append(
            {
                "id": (metadata.get("ID") or path.stem).strip(),
                "path": path,
                "text": f"{header}\n\n{content}",
            }
        )
    return documents


def create_rag(
    working_dir: Path, require_credentials: bool = True
) -> Tuple[LightRAG, Optional[genai.Client]]:
    provider = os.getenv("LIGHTRAG_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    llm_model = os.getenv("LIGHTRAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    embedding_provider = os.getenv(
        "LIGHTRAG_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER
    ).strip().lower()
    default_embedding_model = (
        "gemini-embedding-001"
        if embedding_provider == "gemini"
        else DEFAULT_EMBEDDING_MODEL
    )
    default_embedding_dimension = (
        3072 if embedding_provider == "gemini" else DEFAULT_EMBEDDING_DIMENSION
    )
    embedding_model = os.getenv(
        "LIGHTRAG_EMBEDDING_MODEL", default_embedding_model
    )
    embedding_dimension = int(
        os.getenv(
            "LIGHTRAG_EMBEDDING_DIMENSION", str(default_embedding_dimension)
        )
    )
    embedding_client = None

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if require_credentials and not api_key:
            raise RuntimeError(
                "Chưa cấu hình GEMINI_API_KEY trong file .env hoặc môi trường."
            )
        base_url = os.getenv(
            "GEMINI_OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        completion_lock = asyncio.Lock()
        next_completion_at = 0.0

        async def gemini_completion(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[dict]] = None,
            **kwargs,
        ) -> str:
            nonlocal next_completion_at
            async with completion_lock:
                loop = asyncio.get_running_loop()
                for retry_index in range(LLM_RATE_LIMIT_MAX_RETRIES + 1):
                    delay = next_completion_at - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    try:
                        result = await openai_complete_if_cache(
                            model=llm_model,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            history_messages=history_messages,
                            base_url=base_url,
                            api_key=api_key,
                            **kwargs,
                        )
                    except RateLimitError as exc:
                        retry_after = 0.0
                        response = getattr(exc, "response", None)
                        if response is not None:
                            try:
                                retry_after = float(
                                    response.headers.get("retry-after", "0")
                                )
                            except (TypeError, ValueError):
                                pass
                        backoff = max(
                            retry_after,
                            LLM_RATE_LIMIT_INITIAL_DELAY_SECONDS
                            * (2**retry_index),
                        )
                        if (
                            is_permanent_rate_limit_error(exc)
                            or retry_index == LLM_RATE_LIMIT_MAX_RETRIES
                        ):
                            next_completion_at = loop.time() + backoff
                            raise
                        logger.warning(
                            "Gemini rate limit; retrying completion in "
                            f"{backoff:.0f} seconds "
                            f"({retry_index + 1}/{LLM_RATE_LIMIT_MAX_RETRIES})."
                        )
                        next_completion_at = loop.time() + backoff
                        await asyncio.sleep(backoff)
                    else:
                        next_completion_at = loop.time() + LLM_MIN_INTERVAL_SECONDS
                        return result
                raise RuntimeError("Gemini completion retry loop exited unexpectedly.")

        llm_model_func = gemini_completion
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if require_credentials and not api_key:
            raise RuntimeError(
                "LIGHTRAG_PROVIDER=openai yêu cầu OPENAI_API_KEY trong file .env."
            )
        base_url = os.getenv("OPENAI_BASE_URL") or None

        async def openai_completion(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[dict]] = None,
            **kwargs,
        ) -> str:
            return await openai_complete_if_cache(
                model=llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        llm_model_func = openai_completion
    else:
        raise ValueError(
            f"LIGHTRAG_PROVIDER không được hỗ trợ: {provider!r}. "
            "Chọn 'gemini' hoặc 'openai'."
        )

    if embedding_provider == "local":
        local_model = None
        if require_credentials:
            os.environ.setdefault("USE_TF", "0")
            import torch
            from FlagEmbedding import BGEM3FlagModel

            use_cuda = torch.cuda.is_available()
            local_model = BGEM3FlagModel(
                embedding_model,
                use_fp16=use_cuda,
                devices="cuda" if use_cuda else "cpu",
            )

        def encode_local(texts: List[str]) -> np.ndarray:
            if local_model is None:
                raise RuntimeError(
                    "BGE-M3 chỉ được tải khi ingest; không gọi embedding trong chế độ export."
                )
            result = local_model.encode(
                texts,
                batch_size=min(16, len(texts)),
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != embedding_dimension:
                raise RuntimeError(
                    f"{embedding_model} trả vector chiều "
                    f"{vectors.shape[-1] if vectors.ndim else 0}, "
                    f"nhưng LIGHTRAG_EMBEDDING_DIMENSION={embedding_dimension}."
                )
            return vectors

        async def local_embedding(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.empty((0, embedding_dimension), dtype=np.float32)
            return await asyncio.to_thread(encode_local, texts)

        embedding_function = local_embedding
    elif embedding_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if require_credentials and not api_key:
            raise RuntimeError(
                "LIGHTRAG_EMBEDDING_PROVIDER=gemini yêu cầu GEMINI_API_KEY."
            )
        if api_key:
            embedding_client = genai.Client(api_key=api_key)
        embedding_request_lock = asyncio.Lock()
        next_embedding_request_at = 0.0

        async def gemini_embedding(texts: List[str]) -> np.ndarray:
            nonlocal next_embedding_request_at
            if embedding_client is None:
                raise RuntimeError(
                    "Cần GEMINI_API_KEY để tạo embedding. Hãy cấu hình trong .env."
                )
            async with embedding_request_lock:
                loop = asyncio.get_running_loop()
                delay = next_embedding_request_at - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                next_embedding_request_at = (
                    max(next_embedding_request_at, loop.time())
                    + len(texts) * EMBEDDING_INPUT_INTERVAL_SECONDS
                )
                response = await embedding_client.aio.models.embed_content(
                    model=embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=embedding_dimension,
                    ),
                )
            if not response.embeddings:
                raise RuntimeError("Gemini không trả về embedding cho nội dung.")
            return np.asarray(
                [embedding.values for embedding in response.embeddings],
                dtype=np.float32,
            )

        embedding_function = gemini_embedding
    elif embedding_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if require_credentials and not api_key:
            raise RuntimeError(
                "LIGHTRAG_EMBEDDING_PROVIDER=openai yêu cầu OPENAI_API_KEY."
            )
        embedding_function = partial(openai_embed, model=embedding_model)
    else:
        raise ValueError(
            f"LIGHTRAG_EMBEDDING_PROVIDER không được hỗ trợ: "
            f"{embedding_provider!r}. Chọn 'local', 'gemini' hoặc 'openai'."
        )

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_function,
            model_name=embedding_model,
        ),
        llm_model_max_async=1,
        embedding_func_max_async=1,
        addon_params={
            "language": "Vietnamese",
            "entity_types_guidance": ENTITY_TYPES_GUIDANCE,
        },
    )
    return rag, embedding_client


async def ingest_documents(rag: LightRAG, documents: List[dict], limit: int = 0) -> int:
    selected_documents = documents[:limit] if limit > 0 else documents
    for index, document in enumerate(selected_documents, start=1):
        print(f"[{index}/{len(selected_documents)}] Ingest {document['path'].name}")
        await rag.ainsert(
            document["text"],
            ids=[document["id"]],
            file_paths=[str(document["path"])],
        )
    return len(selected_documents)


async def retry_failed_documents(rag: LightRAG) -> Tuple[int, int]:
    failed_documents = await rag.doc_status.get_docs_by_statuses(
        [DocStatus.FAILED, DocStatus.PROCESSING], strict=True
    )
    retryable_documents = [
        (doc_id, status)
        for doc_id, status in failed_documents.items()
        if not (status.metadata or {}).get("duplicate_kind")
    ]

    processed_count = 0
    failed_count = 0
    for index, (doc_id, status) in enumerate(retryable_documents, start=1):
        content = await rag.full_docs.get_by_id_strict(doc_id)
        if content is None:
            raise RuntimeError(
                f"Không tìm thấy nội dung đã lưu để retry văn bản "
                f"{status.file_path!r}."
            )
        update, _, _ = rag._build_pending_reset_update(status, content)
        await rag.doc_status.upsert({doc_id: update})
        print(
            f"[Retry {index}/{len(retryable_documents)}] "
            f"{Path(status.file_path).name}"
        )
        await rag.apipeline_process_enqueue_documents()

        updated_status = await rag.doc_status.get_by_id(doc_id)
        if updated_status is None:
            raise RuntimeError(
                f"LightRAG mất trạng thái của văn bản {status.file_path!r} sau retry."
            )
        if updated_status.get("status") == DocStatus.PROCESSED:
            processed_count += 1
        else:
            failed_count += 1
            logger.error(
                f"Retry không thành công cho {Path(status.file_path).name}: "
                f"{updated_status.get('error_msg') or updated_status.get('status')}"
            )

    return processed_count, failed_count


async def export_graph(rag: LightRAG) -> Dict[str, List[dict]]:
    graph_storage = rag.chunk_entity_relation_graph
    labels = await graph_storage.get_all_labels()
    nodes_by_id = {}
    edges_by_key = {}

    for label in labels:
        graph = await graph_storage.get_knowledge_graph(
            node_label=label,
            max_depth=1,
            max_nodes=1000,
        )
        for graph_node in graph.nodes:
            properties = graph_node.properties
            node_id = graph_node.id
            entity_type = str(properties.get("entity_type", ""))
            normalized_entity_type = unicodedata.normalize(
                "NFC", entity_type.casefold()
            )
            nodes_by_id[node_id] = {
                "id": node_id,
                "number": node_id,
                "title": str(properties.get("description", "")),
                "type": ENTITY_TYPE_LABELS.get(
                    normalized_entity_type, entity_type
                ),
                "issuer": str(properties.get("source_id", "")),
                "validity": "",
                "text_available": True,
            }
        for graph_edge in graph.edges:
            properties = graph_edge.properties
            edge_key = (graph_edge.source, graph_edge.target)
            edges_by_key[edge_key] = {
                "source": graph_edge.source,
                "target": graph_edge.target,
                "mentions": properties.get("weight", 1),
                "type": str(properties.get("keywords") or graph_edge.type or ""),
                "description": str(properties.get("description", "")),
            }

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": list(edges_by_key.values()),
    }


async def run(args: argparse.Namespace) -> None:
    if not args.ingest and not any(
        args.working_dir.glob("graph_chunk_entity_relation*.graphml")
    ):
        raise RuntimeError(
            f"LightRAG index not found in {args.working_dir}. "
            "Run with --ingest to create index first."
        )
    rag, embedding_client = create_rag(
        args.working_dir,
        require_credentials=args.ingest or args.retry_failed_only,
    )
    initialized = False
    try:
        await rag.initialize_storages()
        initialized = True
        if args.retry_failed_only:
            recovered, still_failed = await retry_failed_documents(rag)
            print(
                f"Retried {recovered} documents; "
                f"{still_failed} documents still failed."
            )
        if args.ingest:
            if args.limit == 0:
                recovered, still_failed = await retry_failed_documents(rag)
                print(
                    f"Retried {recovered} documents; "
                    f"{still_failed} documents still failed."
                )
            documents = load_documents()
            if not documents:
                raise RuntimeError(f"No .txt documents found in {RAW_TEXT_DIR}")
            count = await ingest_documents(rag, documents, args.limit)
            print(f"Sent {count} documents to LightRAG.")

        graph = await export_graph(rag)
        if not graph["nodes"]:
            raise RuntimeError(
                "LightRAG knowledge graph is empty. Run command with --ingest first."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_html(graph), encoding="utf-8")
        print(
            f"Exported {args.output}: {len(graph['nodes'])} entities, "
            f"{len(graph['edges'])} LightRAG relationships."
        )
    finally:
        try:
            if initialized:
                await rag.finalize_storages()
        finally:
            if embedding_client is not None:
                await embedding_client.aio.aclose()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Ingest legal documents into LightRAG and export knowledge graph HTML."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest/update all .txt full texts in data/raw_text before exporting graph.",
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Only retry failed or processing documents; do not ingest new documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only ingest first N documents for testing; default ingests all.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=WORKING_DIR,
        help="Directory to store LightRAG graph/index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML file.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must not be negative.")
    if args.retry_failed_only and args.ingest:
        parser.error("--retry-failed-only cannot be used together with --ingest.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
