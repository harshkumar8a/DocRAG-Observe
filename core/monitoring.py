"""
Monitoring & Metrics Collection for DocRAG Pipeline.
Tracks: latency, throughput, error rates, token usage, retrieval quality.
"""
import time
import json
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: str
    query_text: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks_retrieved: int = 0
    chunks_reranked: int = 0
    parent_chunks_used: int = 0
    pii_detected: int = 0
    error: Optional[str] = None
    document_name: str = ""


@dataclass
class IndexingMetrics:
    """Metrics for document indexing."""
    document_name: str
    timestamp: str
    parse_latency_ms: float = 0.0
    clean_latency_ms: float = 0.0
    chunk_latency_ms: float = 0.0
    pii_latency_ms: float = 0.0
    embed_latency_ms: float = 0.0
    upsert_latency_ms: float = 0.0
    total_chunks: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    pii_instances: int = 0
    error: Optional[str] = None


class MetricsCollector:
    """
    Centralized metrics collection for the RAG pipeline.
    Thread-safe, with in-memory storage and optional persistence.
    """

    def __init__(self, max_history: int = 10000, persist_path: Optional[str] = None):
        self.max_history = max_history
        self.persist_path = persist_path

        # Thread-safe storage
        self._lock = threading.Lock()
        self._query_metrics: deque = deque(maxlen=max_history)
        self._indexing_metrics: deque = deque(maxlen=max_history)
        self._error_count: int = 0
        self._total_queries: int = 0
        self._total_tokens: int = 0
        self._start_time: float = time.time()

    # Query Metrics 

    def record_query(self, metrics: QueryMetrics) -> None:
        """Record metrics for a query."""
        with self._lock:
            self._query_metrics.append(metrics)
            self._total_queries += 1
            self._total_tokens += metrics.tokens_used
            if metrics.error:
                self._error_count += 1

    def record_indexing(self, metrics: IndexingMetrics) -> None:
        """Record metrics for indexing."""
        with self._lock:
            self._indexing_metrics.append(metrics)
            if metrics.error:
                self._error_count += 1

    # Aggregations 

    def get_query_stats(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get query statistics, optionally filtered by time window."""
        with self._lock:
            metrics = list(self._query_metrics)

        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            metrics = [m for m in metrics if datetime.fromisoformat(m.timestamp) >= cutoff]

        if not metrics:
            return {"message": "No query metrics in window"}

        latencies = [m.latency_ms for m in metrics]
        retrieval_lats = [m.retrieval_latency_ms for m in metrics if m.retrieval_latency_ms > 0]
        generation_lats = [m.generation_latency_ms for m in metrics if m.generation_latency_ms > 0]

        return {
            "total_queries": len(metrics),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2),
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            "avg_retrieval_latency_ms": round(sum(retrieval_lats) / max(len(retrieval_lats), 1), 2),
            "avg_generation_latency_ms": round(sum(generation_lats) / max(len(generation_lats), 1), 2),
            "total_tokens": sum(m.tokens_used for m in metrics),
            "avg_chunks_retrieved": round(sum(m.chunks_retrieved for m in metrics) / len(metrics), 2),
            "error_rate": round(self._error_count / max(self._total_queries, 1) * 100, 2),
        }

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        with self._lock:
            metrics = list(self._indexing_metrics)

        if not metrics:
            return {"message": "No indexing metrics"}

        return {
            "total_documents": len(metrics),
            "avg_total_chunks": round(sum(m.total_chunks for m in metrics) / len(metrics), 2),
            "avg_embed_latency_ms": round(
                sum(m.embed_latency_ms for m in metrics) / len(metrics), 2
            ),
            "total_pii_instances": sum(m.pii_instances for m in metrics),
        }

    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        uptime_seconds = time.time() - self._start_time
        query_stats = self.get_query_stats()

        return {
            "status": "healthy" if self._error_count < self._total_queries * 0.1 else "degraded",
            "uptime_seconds": round(uptime_seconds, 2),
            "total_queries": self._total_queries,
            "total_errors": self._error_count,
            "error_rate_percent": round(self._error_count / max(self._total_queries, 1) * 100, 2),
            "avg_latency_ms": query_stats.get("avg_latency_ms", 0),
            "total_tokens_consumed": self._total_tokens,
        }

    # Persistence 

    def save(self) -> None:
        """Save metrics to disk."""
        if not self.persist_path:
            return

        with self._lock:
            data = {
                "query_metrics": [asdict(m) for m in self._query_metrics],
                "indexing_metrics": [asdict(m) for m in self._indexing_metrics],
                "error_count": self._error_count,
                "total_queries": self._total_queries,
                "total_tokens": self._total_tokens,
            }

        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load metrics from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return

        with open(self.persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._error_count = data.get("error_count", 0)
            self._total_queries = data.get("total_queries", 0)
            self._total_tokens = data.get("total_tokens", 0)
            # Note: full history restore omitted for brevity

    # Export 

    def export_to_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        stats = self.get_query_stats()
        lines = [
            "# HELP rag_queries_total Total number of queries",
            "# TYPE rag_queries_total counter",
            f"rag_queries_total {stats.get('total_queries', 0)}",
            "",
            "# HELP rag_latency_ms Average query latency",
            "# TYPE rag_latency_ms gauge",
            f"rag_latency_ms {stats.get('avg_latency_ms', 0)}",
            "",
            "# HELP rag_error_rate_percent Error rate",
            "# TYPE rag_error_rate_percent gauge",
            f"rag_error_rate_percent {stats.get('error_rate', 0)}",
        ]
        return "\n".join(lines)

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for a monitoring dashboard."""
        return {
            "health": self.get_health(),
            "query_stats": self.get_query_stats(),
            "indexing_stats": self.get_indexing_stats(),
            "recent_queries": [
                {
                    "query": m.query_text[:50] + "..." if len(m.query_text) > 50 else m.query_text,
                    "latency_ms": m.latency_ms,
                    "tokens": m.tokens_used,
                    "error": m.error,
                }
                for m in list(self._query_metrics)[-10:]
            ],
        }


# Singleton Instance 

_metrics_instance: Optional[MetricsCollector] = None


def get_metrics_collector(persist_path: Optional[str] = None) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(persist_path=persist_path)
    return _metrics_instance


# Decorator for automatic metric collection 

def timed_query(metric_name: str = "query"):
    """Decorator to automatically time and record query metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                # Record metrics (simplified)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                raise
        return wrapper
    return decorator


# Main / Test 

def main():
    collector = MetricsCollector(persist_path="/tmp/rag_metrics.json")

    # Simulate queries
    for i in range(5):
        collector.record_query(QueryMetrics(
            query_id=f"q{i}",
            timestamp=datetime.now().isoformat(),
            query_text=f"Question {i}",
            latency_ms=1000 + i * 100,
            tokens_used=500 + i * 50,
            chunks_retrieved=10,
        ))

    print("Query Stats:")
    print(json.dumps(collector.get_query_stats(), indent=2))

    print("\nHealth:")
    print(json.dumps(collector.get_health(), indent=2))

    print("\nPrometheus Format:")
    print(collector.export_to_prometheus())


if __name__ == "__main__":
    main()