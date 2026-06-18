"""
RAG Evaluation Module.
Uses RAGAS metrics + custom evaluators for retrieval quality, generation quality,
faithfulness, answer relevance, and context precision.
"""
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS not installed. RAG evaluation disabled.")
    print("   Install: pip install ragas datasets")


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metrics: Dict[str, float] = None
    latency_ms: float = 0.0
    timestamp: str = ""


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using RAGAS + custom metrics.

    Metrics:
        - Faithfulness: Is answer grounded in retrieved context?
        - Answer Relevancy: Does answer address the question?
        - Context Precision: Are retrieved chunks relevant?
        - Context Recall: Did we retrieve all necessary context?
        - Latency: End-to-end response time
        - Token Efficiency: Tokens used vs. answer quality
    """

    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        self.evaluation_history: List[EvaluationResult] = []

    def evaluate_single(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG query-response pair.

        Args:
            query: User question
            response: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional expected answer
            latency_ms: End-to-end latency

        Returns:
            EvaluationResult with all metrics
        """
        metrics = {}

        # ── Custom Metrics (always available) ───────────────────────────────
        metrics["context_count"] = len(contexts)
        metrics["response_length"] = len(response)
        metrics["avg_context_length"] = sum(len(c) for c in contexts) / max(len(contexts), 1)
        metrics["latency_ms"] = latency_ms

        # Keyword overlap (simple relevance proxy)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        metrics["keyword_overlap"] = overlap / max(len(query_words), 1)

        # Context coverage (how much of response is supported by context)
        context_text = " ".join(contexts).lower()
        response_words_in_context = sum(1 for w in response_words if w in context_text)
        metrics["context_coverage"] = response_words_in_context / max(len(response_words), 1)

        # ── RAGAS Metrics (if available) ────────────────────────────────────
        if self.use_ragas and contexts:
            ragas_metrics = self._run_ragas_evaluation(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            metrics.update(ragas_metrics)

        result = EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            latency_ms=latency_ms,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        self.evaluation_history.append(result)
        return result

    def _run_ragas_evaluation(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run RAGAS evaluation on a single example."""
        try:
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            metrics_to_use = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics_to_use.extend([context_recall, answer_similarity])

            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                raise_exceptions=False,
            )

            return {k: round(float(v), 4) for k, v in result.items()}
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {}

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases and compute aggregate statistics.

        Args:
            test_cases: List of dicts with keys: query, response, contexts, ground_truth

        Returns:
            Aggregate metrics and per-case results
        """
        results = []
        for case in test_cases:
            result = self.evaluate_single(
                query=case["query"],
                response=case["response"],
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
                latency_ms=case.get("latency_ms", 0),
            )
            results.append(result)

        # Compute aggregates
        all_metrics = {}
        metric_keys = set()
        for r in results:
            if r.metrics:
                metric_keys.update(r.metrics.keys())

        for key in metric_keys:
            values = [r.metrics[key] for r in results if r.metrics and key in r.metrics]
            if values:
                all_metrics[f"{key}_mean"] = round(sum(values) / len(values), 4)
                all_metrics[f"{key}_min"] = round(min(values), 4)
                all_metrics[f"{key}_max"] = round(max(values), 4)

        return {
            "total_evaluated": len(results),
            "aggregate_metrics": all_metrics,
            "per_case_results": [
                {
                    "query": r.query,
                    "metrics": r.metrics,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}

        total = len(self.evaluation_history)
        avg_latency = sum(r.latency_ms for r in self.evaluation_history) / total

        # Average of all RAGAS scores if available
        ragas_scores = {}
        for r in self.evaluation_history:
            if r.metrics:
                for k, v in r.metrics.items():
                    if k not in ragas_scores:
                        ragas_scores[k] = []
                    ragas_scores[k].append(v)

        avg_scores = {k: round(sum(v) / len(v), 4) for k, v in ragas_scores.items()}

        return {
            "total_evaluations": total,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_metrics": avg_scores,
        }

    def export_results(self, path: str) -> None:
        """Export evaluation history to JSON."""
        data = [
            {
                "query": r.query,
                "response": r.response,
                "contexts": r.contexts,
                "ground_truth": r.ground_truth,
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
                "timestamp": r.timestamp,
            }
            for r in self.evaluation_history
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_report(self) -> str:
        """Generate a human-readable evaluation report."""
        summary = self.get_summary()
        lines = [
            "=" * 70,
            "RAG EVALUATION REPORT",
            "=" * 70,
            f"Total Evaluations: {summary.get('total_evaluations', 0)}",
            f"Average Latency: {summary.get('avg_latency_ms', 0):.0f}ms",
            "",
            "Average Metrics:",
        ]
        for metric, value in summary.get("avg_metrics", {}).items():
            lines.append(f"  {metric}: {value}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ─── LLM-as-Judge Evaluator ─────────────────────────────────────────────────

class LLMJudgeEvaluator:
    """
    Uses an LLM to evaluate response quality.
    Useful when ground truth is not available.
    """

    def __init__(self, judge_fn: Optional[callable] = None):
        """
        Args:
            judge_fn: Function that takes (query, response, contexts) and returns score 0-1
        """
        self.judge_fn = judge_fn

    def evaluate(self, query: str, response: str, contexts: List[str]) -> Dict[str, Any]:
        """Evaluate using LLM judge."""
        if not self.judge_fn:
            return {"error": "No judge function provided"}

        score = self.judge_fn(query, response, contexts)
        return {
            "llm_judge_score": score,
            "passed": score >= 0.7,
        }


# Main / Test 

def main():
    evaluator = RAGEvaluator(use_ragas=False)  # Set True if RAGAS installed

    test_cases = [
        {
            "query": "What is AI?",
            "response": "AI is intelligence demonstrated by machines.",
            "contexts": [
                "Artificial intelligence is intelligence demonstrated by machines.",
                "AI research studies intelligent agents.",
            ],
            "ground_truth": "AI refers to machine intelligence.",
            "latency_ms": 1200,
        },
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of AI that uses data to learn patterns.",
            "contexts": [
                "Machine learning is a subset of artificial intelligence.",
                "It uses algorithms to learn from data.",
            ],
            "latency_ms": 1500,
        },
    ]

    results = evaluator.evaluate_batch(test_cases)
    print(evaluator.generate_report())
    print("\nAggregate Metrics:")
    for k, v in results["aggregate_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()