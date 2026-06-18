"""
Complete RAG Pipeline with Evaluation, Monitoring, and Observability.
Integrates: LangSmith tracing, RAGAS evaluation, custom metrics, PII handling.
Supports batch indexing of multiple PDFs with robust parsing.
"""
import os
import sys
import time
import json
import uuid
import gc
import argparse
import contextlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Core imports 
from utils.pdf_parser import parse_pdf
from utils.text_cleaner import clean_document
from utils.chunker import create_parent_child_chunks, verify_chunks
from utils.pii_sanitizer import get_sanitizer
from core.embedding import OllamaEmbeddingClient, EmbeddingConfig
from core.vector_store import get_pinecone_store
from core.reranker import get_reranker
from core.generation import OllamaGenerationClient, GenerationConfig

# Observability
from core.langsmith_tracing import get_tracer, RAGTracer
from core.evaluation import RAGEvaluator
from core.monitoring import MetricsCollector, QueryMetrics, IndexingMetrics, get_metrics_collector

# Config
from config.settings import *


class RAGPipeline:
    """
    Production RAG pipeline with full observability and multi-PDF support.
    """

    def __init__(
        self,
        embed_model: str = OLLAMA_EMBED_MODEL,
        llm_model: str = OLLAMA_LLM_MODEL,
        reranker_model: str = RERANKER_MODEL,
        pii_enabled: bool = PII_ENABLED,
        pinecone_api_key: Optional[str] = PINECONE_API_KEY,
        pinecone_index_name: Optional[str] = PINECONE_INDEX_NAME,
        pinecone_namespace: Optional[str] = PINECONE_NAMESPACE,
        langsmith_project: str = LANGCHAIN_PROJECT,
        metrics_persist_path: Optional[str] = METRIX_PATH,
    ):
        # Core Components 
        self.embed_client = OllamaEmbeddingClient(
            EmbeddingConfig(base_url=OLLAMA_BASE_URL, model=embed_model)
        )
        self.gen_client = OllamaGenerationClient(
            GenerationConfig(base_url=OLLAMA_BASE_URL, model=llm_model)
        )
        self.reranker = get_reranker(reranker_model)

        # Pinecone Vector Store 
        resolved_index_name = (
            pinecone_index_name
            or PINECONE_INDEX_NAME
            or os.getenv("PINECONE_INDEX_NAME", "doc-rag-index")
        )
        resolved_namespace = (
            pinecone_namespace
            or PINECONE_NAMESPACE
            or os.getenv("PINECONE_NAMESPACE", "default")
        )
        resolved_api_key = (
            pinecone_api_key
            or PINECONE_API_KEY
            or os.getenv("PINECONE_API_KEY", "")
        )

        logger.info(f"Pinecone Config: index='{resolved_index_name}', namespace='{resolved_namespace}'")

        self.vector_store = get_pinecone_store(
            api_key=resolved_api_key,
            index_name=resolved_index_name,
            dimension=PINECONE_DIMENSION,
            namespace=resolved_namespace,
        )

        self.sanitizer = get_sanitizer({
            "PII_ENABLED": pii_enabled,
            "PII_ENTITIES": PII_ENTITIES,
            "PII_MIN_SCORE": PII_MIN_SCORE,
        })
        self.parent_store: Dict[str, Dict[str, Any]] = {}
        self.tracker_path = Path(TRACKER_PATH)
        self.parent_store_path = self.tracker_path.with_suffix(".parents.json")
        self._load_parent_store()
        self._load_tracker()

        # Observability Components 
        self.tracer = get_tracer(project_name=langsmith_project)
        self.tracer_enabled = (
            self.tracer is not None
            and hasattr(self.tracer, 'config')
            and self.tracer.config is not None
        )
        self.evaluator = RAGEvaluator(use_ragas=True)
        self.metrics = get_metrics_collector(persist_path=metrics_persist_path)

        self._validate_tracker_config()

    # Tracer helpers 

    def _trace_run(self, name, run_type="chain", inputs=None):
        if self.tracer_enabled:
            return self.tracer.trace_run(name, run_type=run_type, inputs=inputs)
        return contextlib.nullcontext()

    def _trace_step(self, name, inputs=None):
        if self.tracer_enabled:
            return self.tracer.trace_step(name, inputs=inputs)
        return contextlib.nullcontext()

    def _log_metric(self, key, value):
        if self.tracer_enabled:
            self.tracer.log_metric(key, value)

    # TRACKER (path‑based, safe loading)

    def _load_tracker(self) -> Dict[str, Any]:
        if not self.tracker_path.exists():
            self.tracker = {}
            return {}
        try:
            with open(self.tracker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    self.tracker = {}
                else:
                    self.tracker = data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Tracker corrupt ({e}) – resetting.")
            backup = self.tracker_path.with_suffix(".json.bak")
            if self.tracker_path.exists():
                self.tracker_path.rename(backup)
                logger.info(f"Backup saved to {backup}")
            self.tracker = {}
        return self.tracker

    def _save_tracker(self) -> None:
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.tracker_path.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(self.tracker, f, indent=2, ensure_ascii=False)
        temp.replace(self.tracker_path)

    def _load_parent_store(self) -> None:
        if self.parent_store_path.exists():
            try:
                with open(self.parent_store_path, "r", encoding="utf-8") as f:
                    self.parent_store = json.load(f)
            except Exception:
                self.parent_store = {}
        else:
            self.parent_store = {}

    def _save_parent_store(self) -> None:
        with open(self.parent_store_path, "w", encoding="utf-8") as f:
            serializable = {}
            for k, v in self.parent_store.items():
                serializable[k] = {kk: vv for kk, vv in v.items() if kk != "embedding"}
            json.dump(serializable, f, indent=2)

    def _validate_tracker_config(self) -> None:
        if not self.tracker:
            return
        sample_path = next(iter(self.tracker))
        stored_index = self.tracker[sample_path].get("pinecone_index")
        stored_ns = self.tracker[sample_path].get("pinecone_namespace")
        current_index = self.vector_store.config.index_name
        current_ns = self.vector_store.config.namespace
        if stored_index and stored_index != current_index:
            logger.warning(
                f"Tracker was built with index '{stored_index}', but current is '{current_index}'. "
                "Vectors may be missing. Use --force to re‑index."
            )
        if stored_ns and stored_ns != current_ns:
            logger.warning(
                f"Tracker was built with namespace '{stored_ns}', but current is '{current_ns}'. "
                "Vectors may be missing. Use --force to re‑index."
            )

    def _is_already_indexed(self, pdf_path: str, parsed: Dict[str, Any]) -> bool:
        file_key = str(Path(pdf_path).resolve())
        record = self.tracker.get(file_key, {})
        if record.get("content_hash") != parsed["content_hash"]:
            return False

        # Verify vectors exist in Pinecone
        doc_name = parsed["document_name"]
        try:
            dummy_vec = [0.0] * self.vector_store.config.dimension
            results = self.vector_store.search(
                dummy_vec,
                top_k=1,
                filter_dict={"document_name": {"$eq": doc_name}},
            )
            if not results:
                logger.warning(f"Tracker says '{doc_name}' is indexed, but no vectors found in Pinecone. Re‑indexing.")
                return False
        except Exception as e:
            logger.warning(f"Could not verify vectors for '{doc_name}': {e}. Assuming not indexed.")
            return False

        return True

    def _update_tracker(self, pdf_path: str, parsed: Dict[str, Any]) -> None:
        file_key = str(Path(pdf_path).resolve())
        self.tracker[file_key] = {
            "content_hash": parsed["content_hash"],
            "total_pages": parsed.get("total_pages", 0),
            "tables": len(parsed.get("tables", [])),
            "last_processed": str(Path(pdf_path).stat().st_mtime),
            "pinecone_index": self.vector_store.config.index_name,
            "pinecone_namespace": self.vector_store.config.namespace,
        }
        self._save_tracker()

    # INDEXING (single + batch)

    def index_document(self, pdf_path: str, force: bool = False) -> Dict[str, Any]:
        pdf_path = str(Path(pdf_path).resolve())
        doc_name = Path(pdf_path).stem

        with self._trace_run(
            name="index_document",
            run_type="chain",
            inputs={"pdf_path": pdf_path, "document_name": doc_name, "force": force},
        ) as run:
            idx_metrics = IndexingMetrics(
                document_name=doc_name,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            overall_start = time.time()

            try:
                # Parse 
                with self._trace_step("parse_pdf", inputs={"path": pdf_path}):
                    t0 = time.time()
                    parsed = parse_pdf(pdf_path)
                    idx_metrics.parse_latency_ms = (time.time() - t0) * 1000
                    self._log_metric("parse_latency_ms", idx_metrics.parse_latency_ms)

                # Skip only if NOT forced AND already indexed with vectors present
                if not force and self._is_already_indexed(pdf_path, parsed):
                    msg = f"Document unchanged (hash: {parsed['content_hash'][:16]}...)"
                    logger.info(f"⏭️  {msg}")
                    return {"status": "skipped", "reason": msg, "document": doc_name}

                if force:
                    logger.info(f"🔄 Force‑reindexing '{doc_name}' (--force enabled)")

                # Clean 
                with self._trace_step("clean_text"):
                    t0 = time.time()
                    cleaned_text = clean_document(parsed)
                    idx_metrics.clean_latency_ms = (time.time() - t0) * 1000

                # Chunk 
                with self._trace_step("chunk_document"):
                    t0 = time.time()
                    chunks = create_parent_child_chunks(
                        document_text=cleaned_text,
                        parent_chunk_size=PARENT_CHUNK_SIZE,
                        child_chunk_size=CHILD_CHUNK_SIZE,
                        overlap_fraction=OVERLAP_FRACTION,
                        document_name=doc_name,
                    )
                    idx_metrics.chunk_latency_ms = (time.time() - t0) * 1000
                    stats = verify_chunks(chunks)
                    idx_metrics.total_chunks = stats["total_chunks"]
                    idx_metrics.parent_chunks = stats["parents"]
                    idx_metrics.child_chunks = stats["children"]

                # PII 
                with self._trace_step("sanitize_pii"):
                    t0 = time.time()
                    chunks = self.sanitizer.sanitize_chunks(chunks)
                    idx_metrics.pii_latency_ms = (time.time() - t0) * 1000
                    pii_report = self.sanitizer.get_pii_report(chunks)
                    idx_metrics.pii_instances = pii_report["total_pii_instances"]

                # Delete old (only if re‑indexing) 
                if force:
                    old_chunks = self.vector_store.get_document_chunks(doc_name)
                    if old_chunks:
                        logger.info(f"🗑️  Removing {len(old_chunks)} old chunks from Pinecone...")
                        self.vector_store.delete_by_document(doc_name)

                # Embed 
                with self._trace_step("generate_embeddings"):
                    t0 = time.time()
                    child_chunks = [c for c in chunks if c.get("chunk_type") == "child"]
                    texts = [c["text"] for c in child_chunks]
                    embeddings = self.embed_client.embed_batch(texts)
                    idx_metrics.embed_latency_ms = (time.time() - t0) * 1000

                # Upsert (batching is handled inside vector_store.add) 
                with self._trace_step("upsert_to_pinecone"):
                    t0 = time.time()
                    self.vector_store.add(child_chunks, embeddings)   # <-- batches internally
                    idx_metrics.upsert_latency_ms = (time.time() - t0) * 1000

                # Store parents 
                for chunk in chunks:
                    if chunk.get("chunk_type") == "parent":
                        self.parent_store[chunk["chunk_id"]] = chunk
                self._save_parent_store()

                # Update tracker 
                self._update_tracker(pdf_path, parsed)
                self.metrics.record_indexing(idx_metrics)

                total_time = (time.time() - overall_start) * 1000
                self._log_metric("total_indexing_ms", total_time)

                logger.info(f"✅ Indexed '{doc_name}' in {total_time:.0f}ms")
                return {
                    "status": "indexed",
                    "document": doc_name,
                    "chunks": stats,
                    "pii_report": pii_report,
                    "latency_ms": total_time,
                }

            except Exception as e:
                idx_metrics.error = str(e)
                self.metrics.record_indexing(idx_metrics)
                logger.error(f"❌ Indexing failed for {doc_name}: {e}")
                raise

    def index_documents(self, pdf_paths: List[str], force: bool = False) -> List[Dict[str, Any]]:
        results = []
        for path in pdf_paths:
            try:
                result = self.index_document(path, force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to index {path}: {e}")
                results.append({"status": "error", "path": path, "error": str(e)})
            finally:
                gc.collect()
        return results

    def index_folder(self, folder_path: str, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        folder = Path(folder_path)
        if not folder.is_dir():
            return {"status": "error", "reason": "Folder not found", "folder": folder_path}

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            return {"status": "error", "reason": "No PDF files found", "folder": folder_path}

        logger.info(f"📁 Found {len(pdf_files)} PDF(s) in {folder}")
        if dry_run:
            logger.info("🏁 DRY RUN – no embedding will be performed.")
            stats = {"total": len(pdf_files), "would_index": 0, "would_skip": 0}
            for pdf_file in pdf_files:
                parsed = parse_pdf(str(pdf_file))
                if force:
                    stats["would_index"] += 1
                elif self._is_already_indexed(str(pdf_file), parsed):
                    stats["would_skip"] += 1
                else:
                    stats["would_index"] += 1
                gc.collect()
            logger.info(f"   Would index: {stats['would_index']}, Would skip: {stats['would_skip']}")
            return stats

        results = []
        for idx, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"\n[{idx}/{len(pdf_files)}] ", extra={"end": ""})
            try:
                result = self.index_document(str(pdf_file), force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to index {pdf_file.name}: {e}")
                results.append({"status": "error", "path": str(pdf_file), "error": str(e)})
            finally:
                gc.collect()

        embedded = sum(1 for r in results if r.get("status") == "indexed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        errors = sum(1 for r in results if r.get("status") == "error")
        logger.info("\n" + "═"*60)
        logger.info("📊 BATCH SUMMARY")
        logger.info(f"   Total files   : {len(pdf_files)}")
        logger.info(f"   Indexed       : {embedded}")
        logger.info(f"   Skipped       : {skipped}")
        logger.info(f"   Errors        : {errors}")
        logger.info("═"*60)
        return {"total": len(pdf_files), "indexed": embedded, "skipped": skipped, "errors": errors}

    # QUERY
    def query(
        self,
        question: str,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank: int = TOP_K_RERANK,
        filter_by_document: Optional[str] = None,
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())[:8]
        q_metrics = QueryMetrics(
            query_id=query_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            query_text=question,
        )
        overall_start = time.time()

        with self._trace_run(
            name="rag_query",
            run_type="chain",
            inputs={"query": question, "top_k": top_k_retrieve},
        ) as run:
            try:
                # Embed query 
                with self._trace_step("embed_query"):
                    t0 = time.time()
                    query_embedding = self.embed_client.embed_single(question)
                    q_metrics.retrieval_latency_ms = (time.time() - t0) * 1000

                # Retrieve 
                with self._trace_step("vector_search"):
                    t0 = time.time()
                    filter_dict = None
                    if filter_by_document:
                        filter_dict = {"document_name": {"$eq": filter_by_document}}

                    child_results = self.vector_store.search(
                        query_embedding,
                        top_k=top_k_retrieve,
                        filter_dict=filter_dict,
                    )
                    q_metrics.chunks_retrieved = len(child_results)
                    q_metrics.retrieval_latency_ms += (time.time() - t0) * 1000

                if not child_results:
                    response = "I don't have any relevant information to answer this question."
                    q_metrics.generation_latency_ms = 0
                    q_metrics.latency_ms = (time.time() - overall_start) * 1000
                    self.metrics.record_query(q_metrics)
                    return {
                        "query": question,
                        "response": response,
                        "sources": [],
                        "query_id": query_id,
                    }

                # Rerank 
                with self._trace_step("rerank"):
                    t0 = time.time()
                    reranked = self.reranker.rerank(question, child_results, top_k=top_k_rerank)
                    q_metrics.chunks_reranked = len(reranked)
                    q_metrics.rerank_latency_ms = (time.time() - t0) * 1000

                # Parent expansion 
                with self._trace_step("parent_expansion"):
                    parent_chunks = self._expand_to_parents(reranked)
                    q_metrics.parent_chunks_used = len(parent_chunks)

                # Generate 
                with self._trace_step("generate_response"):
                    t0 = time.time()
                    gen_result = self.gen_client.generate(question, parent_chunks)
                    q_metrics.generation_latency_ms = (time.time() - t0) * 1000
                    q_metrics.prompt_tokens = gen_result.get("prompt_tokens", 0)
                    q_metrics.completion_tokens = gen_result.get("completion_tokens", 0)
                    q_metrics.tokens_used = q_metrics.prompt_tokens + q_metrics.completion_tokens

                # Finalize metrics 
                q_metrics.latency_ms = (time.time() - overall_start) * 1000
                q_metrics.document_name = parent_chunks[0].get("document_name", "") if parent_chunks else ""
                self.metrics.record_query(q_metrics)

                # Evaluate (optional) 
                eval_result = None
                if evaluate:
                    contexts = [c.get("text", "") for c in parent_chunks]
                    eval_result = self.evaluator.evaluate_single(
                        query=question,
                        response=gen_result["response"],
                        contexts=contexts,
                        latency_ms=q_metrics.latency_ms,
                    )

                # Log to LangSmith 
                self._log_metric("total_latency_ms", q_metrics.latency_ms)
                self._log_metric("tokens_used", q_metrics.tokens_used)
                self._log_metric("chunks_retrieved", q_metrics.chunks_retrieved)

                return {
                    "query": question,
                    "response": gen_result["response"],
                    "sources": gen_result["sources"],
                    "query_id": query_id,
                    "latency_ms": q_metrics.latency_ms,
                    "tokens": {
                        "prompt": q_metrics.prompt_tokens,
                        "completion": q_metrics.completion_tokens,
                        "total": q_metrics.tokens_used,
                    },
                    "retrieval": {
                        "chunks_retrieved": q_metrics.chunks_retrieved,
                        "chunks_reranked": q_metrics.chunks_reranked,
                        "parent_chunks_used": q_metrics.parent_chunks_used,
                    },
                    "evaluation": eval_result.metrics if eval_result else None,
                }

            except Exception as e:
                q_metrics.error = str(e)
                q_metrics.latency_ms = (time.time() - overall_start) * 1000
                self.metrics.record_query(q_metrics)
                logger.error(f"Query failed: {e}")
                raise

    # OBSERVABILITY API

    def get_health(self) -> Dict[str, Any]:
        return self.metrics.get_health()

    def get_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        return self.metrics.get_query_stats(window_minutes=window_minutes)

    def get_evaluation_summary(self) -> str:
        return self.evaluator.generate_report()

    def export_metrics(self, path: str) -> None:
        self.metrics.save()
        self.evaluator.export_results(path)

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "health": self.get_health(),
            "metrics": self.get_metrics(),
            "evaluation": self.evaluator.get_summary(),
            "pipeline": self.get_stats(),
        }

    # HELPERS
    def _expand_to_parents(self, child_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parents = []
        seen = set()
        for child in child_chunks:
            parent_id = child.get("parent_id")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                if parent_id in self.parent_store:
                    parent = self.parent_store[parent_id].copy()
                    parent["rerank_score"] = child.get("rerank_score", 0)
                    parents.append(parent)
                else:
                    parents.append(child)
        return parents

    def get_stats(self) -> Dict[str, Any]:
        try:
            pinecone_stats = self.vector_store.get_stats()
            pinecone_vectors = pinecone_stats.get("total_vectors", 0)
        except Exception:
            pinecone_vectors = 0
        return {
            "pinecone_vectors": pinecone_vectors,
            "total_parents_cached": len(self.parent_store),
            "embed_model": self.embed_client.config.model,
            "llm_model": self.gen_client.config.model,
            "reranker": getattr(self.reranker, "model_name", "fallback"),
            "pii_enabled": getattr(self.sanitizer, "enabled", True),
            "langsmith_project": self.tracer.config.project_name if self.tracer_enabled else "",
        }

    def list_indexed_documents(self) -> List[str]:
        return list(self.tracker.keys())


# MAIN (CLI)

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline – Index and query PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all PDFs in the default 'documents' folder
  python -m core.rag_pipeline --folder documents

  # Force re‑index all PDFs (ignore tracker)
  python -m core.rag_pipeline --folder documents --force

  # Ask multiple questions
  python -m core.rag_pipeline --query "What is AI?" --query "What is ML?"

  # Read questions from a file
  python -m core.rag_pipeline --questions-file questions.txt

  # Dry run to see which files would be indexed
  python -m core.rag_pipeline --folder documents --dry-run
        """
    )
    parser.add_argument("--folder", type=str, default="documents",
                        help="Folder containing PDFs to index (batch mode)")
    parser.add_argument("--query", action="append",
                        help="Ask a question (can be used multiple times)")
    parser.add_argument("--questions-file", type=str,
                        help="Path to a text file with one question per line")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show which PDFs would be indexed, without embedding")
    parser.add_argument("--force", action="store_true",
                        help="Force re‑indexing of all documents, ignoring the tracker")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate essential config
    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY is not set. Pinecone operations may fail.")
    if not OLLAMA_BASE_URL:
        logger.warning("OLLAMA_BASE_URL is not set. Embedding and generation may fail.")

    rag = RAGPipeline(
        pinecone_index_name=PINECONE_INDEX_NAME,
        pinecone_namespace=PINECONE_NAMESPACE,
        langsmith_project=LANGCHAIN_PROJECT,
        metrics_persist_path=r"D:\3_Machine learning\GenAI\DocRAG-Observe\metrics.json",
    )

    logger.info("\n📊 Pipeline Stats:")
    for k, v in rag.get_stats().items():
        logger.info(f"   {k}: {v}")

    # Indexing 
    if args.folder:
        result = rag.index_folder(args.folder, dry_run=args.dry_run, force=args.force)
        logger.info(f"\n📁 Folder indexing result: {result}")

    # Build question list 
    questions = []
    if args.query:
        questions.extend(args.query)
    if args.questions_file:
        try:
            with open(args.questions_file, "r", encoding="utf-8") as f:
                questions.extend([line.strip() for line in f if line.strip()])
        except Exception as e:
            logger.error(f"Failed to read questions file: {e}")

    # Query loop 
    if questions:
        if rag.vector_store.get_stats().get("total_vectors", 0) == 0:
            logger.warning("⚠️ No documents indexed – cannot query.")
        else:
            for idx, q in enumerate(questions, 1):
                try:
                    logger.info(f"\n[{idx}/{len(questions)}] ❓ {q}")
                    result = rag.query(q, evaluate=True)
                    print("\n" + "=" * 70)
                    print(f"❓ {result['query']}")
                    print(f"💬 {result['response']}")
                    print(f"\n📈 Latency: {result['latency_ms']:.0f}ms | Tokens: {result['tokens']['total']}")
                    if result.get('evaluation'):
                        print(f"📊 Evaluation: {result['evaluation']}")
                    print("=" * 70)
                except Exception as e:
                    logger.error(f"Query failed for '{q}': {e}")

    # Final reports 
    logger.info("\n" + "=" * 70)
    logger.info("📋 Health Status:")
    logger.info(json.dumps(rag.get_health(), indent=2))

    logger.info("\n📋 Query Metrics:")
    logger.info(json.dumps(rag.get_metrics(), indent=2))

    logger.info("\n📋 Evaluation Report:")
    logger.info(rag.get_evaluation_summary())

    logger.info("\n📋 Dashboard Data:")
    dashboard = rag.get_dashboard_data()
    logger.info(f"Health: {dashboard['health']['status']}")
    logger.info(f"Total Queries: {dashboard['metrics'].get('total_queries', 0)}")
    logger.info(f"Avg Latency: {dashboard['metrics'].get('avg_latency_ms', 0):.0f}ms")
    logger.info(f"Tracked Documents: {len(rag.list_indexed_documents())}")


if __name__ == "__main__":
    main()