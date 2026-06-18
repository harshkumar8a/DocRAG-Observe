"""
Ollama Generation Client for RAG.
Handles LLM inference with context assembly from retrieved chunks.
"""
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    system_prompt: Optional[str] = None


class OllamaGenerationClient:
    """
    Client for LLM generation via Ollama API.
    Assembles RAG context and generates responses.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain the answer, say so clearly.
Always cite your sources using the format [Source: Document Name, Page X]."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context chunks.

        Args:
            query: User question
            context_chunks: Retrieved and reranked chunks
            system_prompt: Optional override for system prompt

        Returns:
            Dict with response text, sources, and metadata
        """
        # Build context string
        context = self._build_context(context_chunks)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Call Ollama
        response = self._call_generate(prompt, system_prompt)

        # Extract sources for citation
        sources = self._extract_sources(context_chunks)

        return {
            "query": query,
            "response": response.get("response", ""),
            "sources": sources,
            "context_used": context,
            "model": self.config.model,
            "total_duration_ms": response.get("total_duration", 0) / 1e6,
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
        }

    def generate_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ):
        """
        Stream generation response (generator).
        Yields partial response text chunks.
        """
        context = self._build_context(context_chunks)
        prompt = self._build_prompt(query, context)

        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt or self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": True,
        }

        response = self._session.post(url, json=payload, stream=True, timeout=self.config.timeout)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", chunk.get("page", "N/A"))
            score = chunk.get("rerank_score", chunk.get("similarity_score", 0))

            context_parts.append(
                f"[Document: {doc_name}, Page: {page}, Relevance: {score:.3f}]\n{text}\n"
            )

        return "\n---\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the final prompt for the LLM."""
        return f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question: {query}

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."

Answer:"""

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique source information from chunks."""
        sources = []
        seen = set()
        for chunk in chunks:
            doc = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", chunk.get("page", "N/A"))
            key = f"{doc}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc,
                    "page": page,
                    "score": chunk.get("rerank_score", chunk.get("similarity_score", 0)),
                })
        return sources

    def _call_generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama generate API with retry logic."""
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt or self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": False,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.config.base_url}. "
                    "Is Ollama running? Run: ollama serve"
                )
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"Ollama generation error: {e}")

    def health_check(self) -> bool:
        """Check if Ollama is running and LLM model is available."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return self.config.model in models or any(
                self.config.model in m for m in models
            )
        except Exception:
            return False


# Convenience Functions 

def get_generation_client(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2",
    temperature: float = 0.3,
) -> OllamaGenerationClient:
    """Factory function to create generation client."""
    config = GenerationConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
    return OllamaGenerationClient(config)


def main():
    client = get_generation_client()

    # Health check
    print("Checking Ollama health...")
    if not client.health_check():
        print(f"❌ Ollama not available or model not pulled.")
        print(f"   Run: ollama pull {client.config.model}")
        return

    print(f"✅ Ollama ready. Model: {client.config.model}")

    # Test generation
    query = "What is artificial intelligence?"
    context_chunks = [
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
            "document_name": "ai_overview.pdf",
            "page_number": 1,
            "rerank_score": 0.95,
        },
        {
            "text": "AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions.",
            "document_name": "ai_overview.pdf",
            "page_number": 2,
            "rerank_score": 0.88,
        },
    ]

    print(f"\nQuery: {query}")
    print("\nGenerating response...")
    result = client.generate(query, context_chunks)

    print(f"\n--- Response ---")
    print(result["response"])
    print(f"\n--- Sources ---")
    for s in result["sources"]:
        print(f"  • {s['document']} (Page {s['page']}) - Score: {s['score']:.3f}")
    print(f"\n--- Stats ---")
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Completion tokens: {result['completion_tokens']}")
    print(f"  Duration: {result['total_duration_ms']:.0f}ms")


if __name__ == "__main__":
    main()