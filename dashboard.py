"""
Real-time monitoring dashboard for DocRAG Pipeline.
Displays health, metrics, and recent queries.
"""
import json
import time
import os
from pathlib import Path

from core.rag_pipeline import RAGPipeline


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_metric(name: str, value: Any, unit: str = ""):
    unit_str = f" {unit}" if unit else ""
    print(f"  {name:.<50} {value}{unit_str}")


def display_health(rag: RAGPipeline):
    print_header("🏥 SYSTEM HEALTH")
    health = rag.get_health()
    status_emoji = "🟢" if health["status"] == "healthy" else "🟡" if health["status"] == "degraded" else "🔴"
    print_metric("Status", f"{status_emoji} {health['status']}")
    print_metric("Uptime", f"{health['uptime_seconds']:.0f}", "seconds")
    print_metric("Total Queries", health["total_queries"])
    print_metric("Total Errors", health["total_errors"])
    print_metric("Error Rate", f"{health['error_rate_percent']:.2f}", "%")
    print_metric("Total Tokens", health["total_tokens_consumed"])


def display_metrics(rag: RAGPipeline):
    print_header("📊 QUERY METRICS (Last Hour)")
    metrics = rag.get_metrics(window_minutes=60)
    if "message" in metrics:
        print(f"  {metrics['message']}")
        return
    print_metric("Total Queries", metrics["total_queries"])
    print_metric("Avg Latency", f"{metrics['avg_latency_ms']:.0f}", "ms")
    print_metric("P50 Latency", f"{metrics['p50_latency_ms']:.0f}", "ms")
    print_metric("P99 Latency", f"{metrics['p99_latency_ms']:.0f}", "ms")
    print_metric("Avg Retrieval Latency", f"{metrics['avg_retrieval_latency_ms']:.0f}", "ms")
    print_metric("Avg Generation Latency", f"{metrics['avg_generation_latency_ms']:.0f}", "ms")
    print_metric("Total Tokens", metrics["total_tokens"])
    print_metric("Avg Chunks Retrieved", f"{metrics['avg_chunks_retrieved']:.1f}")


def display_pipeline_stats(rag: RAGPipeline):
    print_header("⚙️  PIPELINE CONFIGURATION")
    stats = rag.get_stats()
    print_metric("Embed Model", stats["embed_model"])
    print_metric("LLM Model", stats["llm_model"])
    print_metric("Reranker", stats["reranker"])
    print_metric("PII Enabled", "✅ Yes" if stats["pii_enabled"] else "❌ No")
    print_metric("Pinecone Vectors", stats["pinecone_vectors"])
    print_metric("Parents Cached", stats["total_parents_cached"])
    print_metric("LangSmith Project", stats.get("langsmith_project", "N/A"))


def display_evaluation(rag: RAGPipeline):
    print_header("📋 EVALUATION SUMMARY")
    print(rag.get_evaluation_summary())


def display_recent_queries(rag: RAGPipeline):
    print_header("🕐 RECENT QUERIES")
    dashboard = rag.get_dashboard_data()
    recent = dashboard.get("recent_queries", [])
    if not recent:
        print("  No recent queries")
        return
    for i, q in enumerate(recent[-5:], 1):
        status = "✅" if not q.get("error") else "❌"
        print(f"  {status} [{q['latency_ms']:.0f}ms] {q['query'][:60]}...")


def main():
    print("\n" + "🚀" * 35)
    print("  DocRAG-Observe Monitoring Dashboard")
    print("🚀" * 35)

    rag = RAGPipeline(
        pinecone_index_name="doc-rag-observed",
        pinecone_namespace="production",
        langsmith_project="doc-rag-production",
    )

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        display_health(rag)
        display_metrics(rag)
        display_pipeline_stats(rag)
        display_evaluation(rag)
        display_recent_queries(rag)

        print("\n" + "-" * 70)
        print("  Press Ctrl+C to exit | Refreshing every 10 seconds...")
        print("-" * 70)

        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped.")
            break


if __name__ == "__main__":
    main()