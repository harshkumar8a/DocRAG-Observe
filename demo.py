"""
Demo script to test the RAG pipeline with a set of questions.
"""
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from core.rag_pipeline import RAGPipeline
from config.settings import *   # import everything


def main():
    # Initialize using the SAME config as indexing 
    rag = RAGPipeline(
        pinecone_index_name=PINECONE_INDEX_NAME,   # e.g., docrag-obs
        pinecone_namespace=PINECONE_NAMESPACE,     # e.g., docrag
        langsmith_project=LANGCHAIN_PROJECT,       # or fallback
        metrics_persist_path="metrics.json",
    )

    # Check vectors 
    vector_count = rag.vector_store.count()
    print(f"📊 Pinecone index '{PINECONE_INDEX_NAME}' has {vector_count} vectors.")

    if vector_count == 0:
        print("⚠️  No vectors found. Please run indexing first.")
        return

    # Define your test questions 
    questions = [
        "What is an AI agent?"
    ]

    # Run queries 
    print("\n" + "=" * 80)
    print("🔍 Running test queries...")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        result = rag.query(q, evaluate=False)

        print(f"\n[{i}/{len(questions)}] ❓ {q}")
        print(f"💬 Answer:\n{result['response']}\n")

        sources = result.get("sources", [])
        print(f"📚 Sources ({len(sources)}):")
        for src in sources[:5]:
            doc = src.get("document", "unknown")
            page = src.get("page", "?")
            score = src.get("rerank_score", src.get("score", 0))
            print(f"   • {doc} (page {page}) [score: {score:.3f}]")

        if sources:
            snippet = sources[0].get("text", "")[:300]
            if len(sources[0].get("text", "")) > 300:
                snippet += "..."
            print(f"\n📄 Top context snippet:\n{snippet}\n")

        print(f"⏱️  Latency: {result.get('latency_ms', 0):.0f} ms")
        tokens = result.get("tokens", {})
        print(f"🔢 Tokens: {tokens.get('total', 0)} (prompt: {tokens.get('prompt', 0)}, completion: {tokens.get('completion', 0)})")
        print("-" * 60)


if __name__ == "__main__":
    main()