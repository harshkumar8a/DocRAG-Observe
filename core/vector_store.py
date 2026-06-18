"""
Pinecone Vector Store for RAG.
Cloud-native vector database with metadata filtering and namespace support.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from config.settings import *


@dataclass
class PineconeConfig:
    api_key: str = PINECONE_API_KEY
    index_name: str = PINECONE_INDEX_NAME
    dimension: int = PINECONE_DIMENSION
    metric: str = PINECONE_METRIC
    cloud: str = PINECONE_CLOUD
    region: str = PINECONE_REGION
    namespace: str = PINECONE_NAMESPACE


class PineconeVectorStore:
    def __init__(self, config: Optional[PineconeConfig] = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("pinecone not installed. Run: pip install pinecone")

        self.config = config or PineconeConfig()
        # ... (config resolution same as yours)

        print(f"🔌 Connecting to Pinecone index: '{self.config.index_name}'")
        self.pc = Pinecone(api_key=self.config.api_key)
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        if not self.pc.has_index(self.config.index_name):
            print(f"📦 Creating Pinecone index: '{self.config.index_name}'")
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(cloud=self.config.cloud, region=self.config.region),
            )
        else:
            print(f"✅ Using existing Pinecone index: '{self.config.index_name}'")
        return self.pc.index(self.config.index_name)

    def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 100) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match")

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]
            vectors = []
            for chunk, embedding in zip(batch_chunks, batch_embeds):
                vector_id = chunk.get("chunk_id", chunk.get("id", f"vec_{hash(str(chunk))}"))
                metadata = {
                    "text": chunk.get("text", "")[:4000],
                    "chunk_type": chunk.get("chunk_type", "unknown"),
                    "document_name": chunk.get("document_name", "unknown"),
                    "parent_id": chunk.get("parent_id", ""),
                    "parent_idx": chunk.get("parent_idx", -1),
                    "child_idx": chunk.get("child_idx", -1),
                    "page_number": chunk.get("page_number", chunk.get("page", -1)),
                    "token_count": chunk.get("token_count", 0),
                    "has_pii": chunk.get("has_pii", False),
                    "pii_count": chunk.get("pii_count", 0),
                }
                vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

            self.index.upsert(vectors=vectors, namespace=self.config.namespace)
            if total > batch_size:
                print(f"✅ Upserted batch {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}")

        print(f"✅ Upserted all {total} vectors into namespace '{self.config.namespace}'")

    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.config.namespace,
            include_metadata=True,
            filter=filter_dict,
        )
        chunks = []
        for match in results.matches:
            chunk = dict(match.metadata) if match.metadata else {}
            chunk["chunk_id"] = match.id
            chunk["similarity_score"] = float(match.score)
            chunk["parent_id"] = match.metadata.get("parent_id", "") if match.metadata else ""
            chunks.append(chunk)
        return chunks

    def search_by_text(self, query_text: str, embed_fn: callable, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        embedding = embed_fn(query_text)
        return self.search(embedding, top_k=top_k, filter_dict=filter_dict)

    def delete_by_document(self, document_name: str) -> int:
        self.index.delete(namespace=self.config.namespace, filter={"document_name": {"$eq": document_name}})
        stats = self.index.describe_index_stats()
        return stats.total_vector_count

    def delete_by_ids(self, ids: List[str]) -> None:
        self.index.delete(ids=ids, namespace=self.config.namespace)

    def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        zero_vector = [0.0] * self.config.dimension
        results = self.index.query(
            vector=zero_vector,
            top_k=10000,
            namespace=self.config.namespace,
            include_metadata=True,
            filter={"document_name": {"$eq": document_name}},
        )
        chunks = []
        for match in results.matches:
            chunk = dict(match.metadata) if match.metadata else {}
            chunk["chunk_id"] = match.id
            chunk["similarity_score"] = float(match.score)
            chunks.append(chunk)
        return chunks

    def count(self) -> int:
        """Total vectors in the namespace."""
        stats = self.index.describe_index_stats()
        ns_stats = stats.namespaces.get(self.config.namespace, {})
        return ns_stats.get("vector_count", 0)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": {k: v.vector_count for k, v in stats.namespaces.items()},
        }

    def list_namespaces(self) -> List[str]:
        stats = self.index.describe_index_stats()
        return list(stats.namespaces.keys())


def get_pinecone_store(api_key: Optional[str] = None, index_name: Optional[str] = None,
                       dimension: Optional[int] = None, namespace: Optional[str] = None) -> PineconeVectorStore:
    config = PineconeConfig(
        api_key=api_key or PINECONE_API_KEY or os.getenv("PINECONE_API_KEY", ""),
        index_name=index_name or PINECONE_INDEX_NAME or "doc-rag-index",
        dimension=dimension or PINECONE_DIMENSION or 768,
        namespace=namespace or PINECONE_NAMESPACE or "default",
    )
    return PineconeVectorStore(config)


if __name__ == "__main__":
    import numpy as np
    store = get_pinecone_store(index_name="test-rag-index", dimension=768)
    print(f"Index stats: {store.get_stats()}")
    # ... rest of test