"""
Batch evaluation runner for DocRAG Pipeline.
Runs test cases and generates a comprehensive evaluation report.
"""
import json
from pathlib import Path
from core.rag_pipeline import RAGPipeline


# Test Dataset 

TEST_DATASET = [
    {
        "query": "What is artificial intelligence?",
        "ground_truth": "AI is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans and animals.",
        "expected_sources": ["ai.pdf"],
    },
    {
        "query": "What are the main applications of AI?",
        "ground_truth": "AI applications include web search engines, recommendation systems, speech understanding, self-driving cars, and generative tools.",
        "expected_sources": ["ai.pdf"],
    },
    {
        "query": "How does machine learning relate to AI?",
        "ground_truth": "Machine learning is a subset of artificial intelligence that uses data and algorithms to learn patterns.",
        "expected_sources": ["ai.pdf"],
    },
]


def run_evaluation(rag: RAGPipeline, test_cases: list) -> dict:
    """Run evaluation on test dataset."""
    results = []

    for case in test_cases:
        print(f"\n📝 Evaluating: {case['query'][:50]}...")

        result = rag.query(case["query"], evaluate=True)

        results.append({
            "query": case["query"],
            "response": result["response"],
            "ground_truth": case.get("ground_truth", ""),
            "latency_ms": result["latency_ms"],
            "tokens": result["tokens"],
            "sources": result["sources"],
            "evaluation": result.get("evaluation"),
        })

    return {"results": results, "total": len(results)}


def generate_report(eval_results: dict) -> str:
    """Generate markdown evaluation report."""
    lines = [
        "# DocRAG Evaluation Report",
        "",
        f"**Total Test Cases:** {eval_results['total']}",
        "",
        "## Results",
        "",
    ]

    for i, r in enumerate(eval_results["results"], 1):
        lines.extend([
            f"### Test Case {i}",
            f"**Query:** {r['query']}",
            f"**Response:** {r['response'][:200]}...",
            f"**Ground Truth:** {r['ground_truth'][:200]}...",
            f"**Latency:** {r['latency_ms']:.0f}ms",
            f"**Tokens:** {r['tokens']['total']}",
            "",
        ])
        if r.get("evaluation"):
            lines.append("**Metrics:**")
            for k, v in r["evaluation"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("📊 DocRAG Batch Evaluation")
    print("=" * 70)

    rag = RAGPipeline(
        pinecone_index_name="doc-rag-observed",
        pinecone_namespace="production",
    )

    # Check if we have indexed documents
    if rag.vector_store.count() == 0:
        print("\n⚠️  No documents indexed. Please index documents first.")
        return

    # Run evaluation
    results = run_evaluation(rag, TEST_DATASET)

    # Generate and save report
    report = generate_report(results)
    report_path = Path(r"D:\3_Machine learning\GenAI\DocRAG-Observe\evaluation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_path}")
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()