"""
Text cleaning utilities for RAG pipeline.
Works with the new parser output format (single markdown string).
"""
import re
from typing import Dict, Any, List


def strip_page_numbers_footers(text: str) -> str:
    """
    Remove common page number patterns and footers.
    """
    patterns = [
        r'(?m)^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$',      # "Page 1 of 10"
        r'(?m)^\s*\d+\s*$',                              # standalone number
        r'(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$',             # "- 2 -"
        r'(?m)^\s*\d+\s*/\s*\d+\s*$',                    # "1/10"
        r'(?m)^\s*Copyright\s+©.*$',                     # copyright footers
        r'(?m)^\s*Confidential.*$',                      # confidential markers
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.MULTILINE)
    return text


def normalize_line_breaks(text: str) -> str:
    """
    Replace excessive newlines, collapse multiple spaces,
    preserve paragraph breaks (double newline).
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text


def remove_excessive_whitespace(text: str) -> str:
    """Remove multiple consecutive spaces and tabs."""
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def clean_markdown(text: str) -> str:
    """
    Run all cleaning steps on a markdown string.
    Order matters: strip footers first, then normalize whitespace.
    """
    text = strip_page_numbers_footers(text)
    text = normalize_line_breaks(text)
    text = remove_excessive_whitespace(text)
    return text


def clean_document(parsed: Dict[str, Any]) -> str:
    """
    Clean the parsed document output from parse_pdf().
    
    Args:
        parsed: Output from parse_pdf() containing 'markdown', 'tables', 'document_name'
    
    Returns:
        Cleaned full-text string ready for chunking.
    """
    raw_markdown = parsed.get("markdown", "")
    
    if not raw_markdown:
        # Fallback: if markdown is empty, try concatenating tables
        tables = parsed.get("tables", [])
        raw_markdown = "\n\n".join(
            t.get("markdown", "") for t in tables
        )
    
    cleaned = clean_markdown(raw_markdown)
    return cleaned


def extract_table_text(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract table metadata separately for specialized indexing.
    """
    tables = []
    for t in parsed.get("tables", []):
        tables.append({
            "page": t.get("page"),
            "markdown": clean_markdown(t.get("markdown", "")),
        })
    return tables


def main():
    # Test with simulated parser output
    parsed = {
        "markdown": (
            "Artificial Intelligence\n\n"
            "Page 1\n\n"
            "AI is transforming industries.\n\n\n\n\n"
            "Machine learning is a subset of AI.\n\n"
            "Page 2\n\n"
            "Deep learning uses neural networks."
        ),
        "tables": [
            {"page": 1, "markdown": "| Model | Accuracy |\n|-------|----------|\n| CNN   | 95% "},
        ],
        "document_name": "ai",
    }

    cleaned = clean_document(parsed)
    print("--- CLEANED DOCUMENT ---")
    print(cleaned)
    print("\n--- TABLES ---")
    for t in extract_table_text(parsed):
        print(f"Page {t['page']}: {t['markdown'][:60]}...")


if __name__ == "__main__":
    main()