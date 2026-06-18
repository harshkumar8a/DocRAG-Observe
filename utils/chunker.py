"""
Hierarchical Parent-Child Chunking for RAG.
Parents = large retrieval targets | Children = small embedding targets.
"""
import hashlib
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.constants import *


# Token-based length function 

def get_token_length(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken (OpenAI's tokenizer)."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding(model)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate (1 token ≈ 4 chars)
    return len(text) // 4


# Parent-Child Chunker 

def create_parent_child_chunks(
    document_text: str,
    parent_chunk_size: int,
    child_chunk_size: int,
    overlap_fraction: float,
    document_name: str = "unknown",
    use_tokens: bool = True,
    model_name: str = "cl100k_base",
) -> List[Dict[str, Any]]:
    """
    Hierarchical parent-child chunking for RAG.

    Strategy:
        - Parents: large, non-overlapping chunks (retrieval targets)
        - Children: small, overlapping chunks (embedding targets, linked to parent)
    """
    length_fn = get_token_length if use_tokens else len
    doc_hash = hashlib.md5(document_text.encode()).hexdigest()[:8]

    # Parent Splitting 
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    parent_texts = [
        p.strip() for p in parent_splitter.split_text(document_text)
        if p.strip()
    ]

    # Child Splitting 
    child_overlap = int(child_chunk_size * overlap_fraction)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    all_chunks: List[Dict[str, Any]] = []
    char_offset = 0

    for parent_idx, parent_text in enumerate(parent_texts):
        start_char = document_text.find(parent_text, char_offset)
        end_char = start_char + len(parent_text) if start_char != -1 else -1
        if start_char != -1:
            char_offset = start_char + 1

        # Parent Chunk 
        parent_id = f"{doc_hash}_P{parent_idx}"
        parent_chunk = {
            METADATA_CHUNK_ID: parent_id,
            METADATA_TEXT: parent_text,
            METADATA_CHUNK_TYPE: CHUNK_TYPE_PARENT,
            METADATA_PARENT_ID: parent_id,
            "document_name": document_name,
            "parent_idx": parent_idx,
            "start_char": start_char,
            "end_char": end_char,
            "token_count": length_fn(parent_text),
        }
        all_chunks.append(parent_chunk)

        # Child Chunks 
        child_texts = [
            c.strip() for c in child_splitter.split_text(parent_text)
            if c.strip()
        ]

        for child_idx, child_text in enumerate(child_texts):
            child_start_in_parent = parent_text.find(child_text)
            child_start_char = (
                start_char + child_start_in_parent if start_char != -1 else -1
            )

            all_chunks.append({
                METADATA_CHUNK_ID: f"{parent_id}_C{child_idx}",
                METADATA_TEXT: child_text,
                METADATA_CHUNK_TYPE: CHUNK_TYPE_CHILD,
                METADATA_PARENT_ID: parent_id,
                "document_name": document_name,
                "parent_idx": parent_idx,
                "child_idx": child_idx,
                "start_char": child_start_char,
                "end_char": child_start_char + len(child_text) if child_start_char != -1 else -1,
                "token_count": length_fn(child_text),
            })

    return all_chunks


# Verification Helper 

def verify_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate chunk integrity and return stats."""
    parents = [c for c in chunks if c[METADATA_CHUNK_TYPE] == CHUNK_TYPE_PARENT]
    children = [c for c in chunks if c[METADATA_CHUNK_TYPE] == CHUNK_TYPE_CHILD]

    orphan_children = [
        c for c in children 
        if c[METADATA_PARENT_ID] not in {p[METADATA_CHUNK_ID] for p in parents}
    ]

    empty_chunks = [c for c in chunks if not c[METADATA_TEXT].strip()]

    return {
        "total_chunks": len(chunks),
        "parents": len(parents),
        "children": len(children),
        "avg_parent_tokens": sum(p.get("token_count", 0) for p in parents) / max(len(parents), 1),
        "avg_child_tokens": sum(c.get("token_count", 0) for c in children) / max(len(children), 1),
        "orphan_children": len(orphan_children),
        "empty_chunks": len(empty_chunks),
    }


# Main / Test 

def main():
    sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines.

AI research has been defined as the field of study of intelligent agents.

The term "artificial intelligence" had previously been used to describe machines that mimic human cognitive skills.

AI applications include advanced web search engines, recommendation systems, and self-driving cars.

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI."""

    chunks = create_parent_child_chunks(
        document_text=sample_text,
        parent_chunk_size=300,
        child_chunk_size=100,
        overlap_fraction=0.2,
        document_name="ai_overview.txt",
        use_tokens=True,
    )

    stats = verify_chunks(chunks)
    print("Chunk Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n--- Sample Output ---")
    for chunk in chunks[:6]:
        print(f"\n[{chunk[METADATA_CHUNK_TYPE].upper()}] {chunk[METADATA_CHUNK_ID]}")
        print(f"  Parent: {chunk[METADATA_PARENT_ID]} | Tokens: {chunk['token_count']}")
        print(f"  Text: {chunk[METADATA_TEXT][:120]}...")


if __name__ == "__main__":
    main()