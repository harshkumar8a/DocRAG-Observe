"""
Cross-Encoder Reranker for RAG.
Re-ranks retrieved chunks using a cross-encoder model for better relevance.
"""
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class CrossEncoderReranker:
    """
    Cross-encoder reranker for RAG retrieval.

    Pipeline:
        1. Bi-encoder retrieves top-k candidates (fast)
        2. Cross-encoder re-ranks candidates (accurate)
        3. Return top-n after reranking
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = CrossEncoder(self.model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        score_key: str = "similarity_score",
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks based on query relevance.

        Args:
            query: User query string
            chunks: Retrieved chunks from vector store
            top_k: Number of top chunks to return after reranking
            score_key: Key for original retrieval score

        Returns:
            Chunks sorted by cross-encoder relevance score
        """
        if not chunks:
            return []

        # Prepare query-document pairs
        texts = [chunk.get("text", "") for chunk in chunks]
        pairs = [[query, text] for text in texts]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Attach rerank scores
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
            # Combined score: weighted average of retrieval and rerank
            original_score = chunk.get(score_key, 0.5)
            chunk["combined_score"] = 0.3 * original_score + 0.7 * float(score)

        # Sort by rerank score (descending)
        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]

    def rerank_with_parent_expansion(
        self,
        query: str,
        child_chunks: List[Dict[str, Any]],
        parent_lookup: Dict[str, Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank child chunks and expand to parent chunks for richer context.

        Args:
            query: User query
            child_chunks: Retrieved child chunks
            parent_lookup: Dict mapping parent_id -> parent chunk
            top_k: Final number of parent chunks to return

        Returns:
            Parent chunks sorted by best child rerank score
        """
        if not child_chunks:
            return []

        # Rerank children
        reranked_children = self.rerank(query, child_chunks, top_k=len(child_chunks))

        # Group by parent and keep best score per parent
        parent_scores: Dict[str, float] = {}
        parent_children: Dict[str, List[Dict]] = {}

        for child in reranked_children:
            parent_id = child.get("parent_id")
            if not parent_id:
                continue

            score = child["rerank_score"]
            if parent_id not in parent_scores or score > parent_scores[parent_id]:
                parent_scores[parent_id] = score

            if parent_id not in parent_children:
                parent_children[parent_id] = []
            parent_children[parent_id].append(child)

        # Get parent chunks
        results = []
        for parent_id, score in sorted(parent_scores.items(), key=lambda x: x[1], reverse=True):
            if parent_id in parent_lookup:
                parent = parent_lookup[parent_id].copy()
                parent["rerank_score"] = score
                parent["child_chunks"] = parent_children.get(parent_id, [])
                results.append(parent)

        return results[:top_k]


class FallbackReranker:
    """Simple fallback reranker when cross-encoder is unavailable.
    Uses BM25-style keyword matching."""

    def __init__(self):
        pass

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        score_key: str = "similarity_score",
    ) -> List[Dict[str, Any]]:
        """Simple keyword overlap scoring."""
        query_terms = set(query.lower().split())

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            text_terms = set(text.split())
            overlap = len(query_terms & text_terms)
            chunk["rerank_score"] = overlap / max(len(query_terms), 1)
            original = chunk.get(score_key, 0.5)
            chunk["combined_score"] = 0.5 * original + 0.5 * chunk["rerank_score"]

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def get_reranker(model_name: Optional[str] = None, device: str = "cpu"):
    """Factory function to get the best available reranker."""
    if ST_AVAILABLE:
        return CrossEncoderReranker(model_name, device)
    else:
        print("⚠️  sentence-transformers not available. Using fallback reranker.")
        return FallbackReranker()


# Main / Test 

def main():
    reranker = get_reranker()

    query = "What are the health benefits of running?"
    chunks = [
        {"text": "Running is a popular form of physical exercise enjoyed by millions worldwide."},
        {"text": "Running regularly improves cardiovascular health, boosts mental well-being, and helps with weight management."},
        {"text": "Many athletes use running as part of their training to improve endurance and performance."},
        {"text": "Email notifications can be configured in your account settings."},
    ]

    print(f"Query: {query}")
    print("\nBefore Reranking:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['text'][:60]}...")

    ranked = reranker.rerank(query, chunks, top_k=3)

    print("\nAfter Reranking:")
    for i, c in enumerate(ranked, 1):
        print(f"  {i}. [score: {c['rerank_score']:.3f}] {c['text'][:60]}...")


if __name__ == "__main__":
    main()