"""
Ollama Embedding Client for RAG.
Handles batch embedding generation via Ollama's /api/embed endpoint.
"""
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config.settings import *

from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

@dataclass
class EmbeddingConfig:
    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_LLM_MODEL
    timeout: int = OLLAMA_TIMEOUT
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0


class OllamaEmbeddingClient:
    """
    Client for generating embeddings via Ollama API.
    Supports batch processing with retry logic.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self._call_api([text])
        if response and "embeddings" in response and len(response["embeddings"]) > 0:
            return response["embeddings"][0]
        raise RuntimeError(f"Failed to get embedding for: {text[:50]}...")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            response = self._call_api(batch)

            if response and "embeddings" in response:
                all_embeddings.extend(response["embeddings"])
            else:
                raise RuntimeError(f"Failed to get embeddings for batch {i//self.config.batch_size}")

        return all_embeddings

    def _call_api(self, inputs: List[str]) -> Optional[Dict[str, Any]]:
        """Call Ollama embed API with retry logic."""
        url = f"{self.config.base_url}/api/embed"
        payload = {
            "model": self.config.model,
            "input": inputs,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                )
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
            except requests.exceptions.Timeout:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"Ollama request timed out after {self.config.timeout}s")
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"Ollama API error: {e}")

    def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
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

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")


# Convenience Functions 

def get_embedding_client(
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
) -> OllamaEmbeddingClient:
    """Factory function to create embedding client."""
    config = EmbeddingConfig(base_url=base_url, model=model)
    return OllamaEmbeddingClient(config)


def main():
    client = get_embedding_client()

    # Health check
    print("Checking Ollama health...")
    if not client.health_check():
        print("❌ Ollama not available or model not pulled.")
        print(f"   Available models: {client.list_models()}")
        print(f"   Run: ollama pull {client.config.model}")
        return

    print(f"✅ Ollama ready. Model: {client.config.model}")

    # Test single embedding
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\n--- Single Embedding ---")
    print(f"Text: {text}")
    embedding = client.embed_single(text)
    print(f"Embedding dim: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
    ]
    print(f"\n--- Batch Embedding ({len(texts)} texts) ---")
    embeddings = client.embed_batch(texts)
    for i, emb in enumerate(embeddings):
        print(f"  Text {i+1}: dim={len(emb)}, first={emb[:3]}")


if __name__ == "__main__":
    main()