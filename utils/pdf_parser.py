import os
import sys
import gc
import hashlib
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Parsers 
import pdfplumber  # primary, safe

# Docling is optional (disabled by default)
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# ── Configuration ──
EMBEDDING_TRACKER_PATH = Path(r"D:\3_Machine learning\GenAI\DocRAG-Observe\embedding_tracker.json")
DEFAULT_PDF_FOLDER = Path(r"D:\3_Machine learning\GenAI\DocRAG-Observe\documents")

# Set this to True if you want to use Docling for small PDFs (≤ 30 pages)
USE_DOCLING_FOR_SMALL_PDFS = False   # Change to True at your own risk
DOCLING_PAGE_LIMIT = 30              # PDFs with more pages will use pdfplumber only


# HASHING & TRACKER (safe loading & atomic save)

def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_embedding_tracker() -> Dict[str, Any]:
    if EMBEDDING_TRACKER_PATH.exists():
        try:
            with open(EMBEDDING_TRACKER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    print("⚠️ Tracker file is not a dict – resetting.")
                    return {}
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Could not load tracker file ({e}) – starting fresh.")
            backup = EMBEDDING_TRACKER_PATH.with_suffix(".json.bak")
            if EMBEDDING_TRACKER_PATH.exists():
                EMBEDDING_TRACKER_PATH.rename(backup)
                print(f"   📁 Corrupt file backed up to {backup}")
            return {}
    return {}

def save_embedding_tracker(tracker: Dict[str, Any]) -> None:
    EMBEDDING_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = EMBEDDING_TRACKER_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)
    temp_path.replace(EMBEDDING_TRACKER_PATH)

def should_embed(pdf_path: str, content_hash: str, tracker: Dict[str, Any],
                 check_mode: str = "content") -> tuple[bool, Optional[str]]:
    file_path = str(Path(pdf_path).resolve())
    file_hash = compute_file_hash(pdf_path)
    record = tracker.get(file_path, {})
    prev_content_hash = record.get("content_hash")
    prev_file_hash = record.get("file_hash")
    if check_mode == "content":
        return prev_content_hash != content_hash, prev_content_hash
    elif check_mode == "file":
        return prev_file_hash != file_hash, prev_file_hash
    else:
        return (prev_content_hash != content_hash) or (prev_file_hash != file_hash), prev_content_hash

def update_tracker(pdf_path: str, content_hash: str, tracker: Dict[str, Any],
                   chunks_count: int = 0) -> None:
    file_path = str(Path(pdf_path).resolve())
    tracker[file_path] = {
        "file_hash": compute_file_hash(pdf_path),
        "content_hash": content_hash,
        "last_embedded": datetime.now().isoformat(),
        "chunks_count": chunks_count,
        "document_name": Path(pdf_path).stem,
    }
    save_embedding_tracker(tracker)


# PDF PARSERS

def get_pdf_page_count(pdf_path: str) -> int:
    """Quick page count using pdfplumber (lightweight)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except:
        return 0

def parse_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
    """Primary parser – safe, fast, no memory issues."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return {
        "markdown": text,
        "tables": [],
        "document_name": Path(pdf_path).stem,
        "content_hash": compute_content_hash(text),
    }

def parse_with_docling(pdf_path: str) -> Dict[str, Any]:
    """Optional Docling parser – only for small PDFs if enabled."""
    if not DOCLING_AVAILABLE:
        raise RuntimeError("Docling not installed")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    # pipeline_options.image_scale = 0.5  # reduce further if needed

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    markdown = doc.export_to_markdown()
    return {
        "markdown": markdown,
        "tables": [],
        "document_name": Path(pdf_path).stem,
        "content_hash": compute_content_hash(markdown),
    }

def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Choose parser based on settings and page count.
    By default, only pdfplumber is used to avoid std::bad_alloc.
    """
    page_count = get_pdf_page_count(pdf_path)
    use_docling = USE_DOCLING_FOR_SMALL_PDFS and page_count <= DOCLING_PAGE_LIMIT

    if use_docling and DOCLING_AVAILABLE:
        try:
            return parse_with_docling(pdf_path)
        except Exception as e:
            print(f"  ⚠️ Docling failed for {Path(pdf_path).name}: {e}")
            print("  ➜ Falling back to pdfplumber.")
            return parse_with_pdfplumber(pdf_path)
    else:
        return parse_with_pdfplumber(pdf_path)


# EMBEDDING STUBS (replace with your actual vector DB logic)

def embed_document(parsed: Dict[str, Any]) -> int:
    print(f"  🔄 Embedding '{parsed['document_name']}' into vector DB...")
    chunks_count = len(parsed["markdown"]) // 500
    print(f"  ✅ Embedded {chunks_count} chunks")
    return chunks_count

def delete_old_embeddings(document_name: str) -> None:
    print(f"  🗑️  Deleting old embeddings for '{document_name}'...")



# PROCESS SINGLE PDF

def process_pdf(pdf_path: str, check_mode: str = "content") -> Dict[str, Any]:
    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.exists(pdf_path):
        return {"status": "error", "reason": "File not found", "path": pdf_path}

    print(f"\n{'─'*60}")
    print(f"📄 Processing: {Path(pdf_path).name}")
    print(f"{'─'*60}")

    print("  🔍 Parsing PDF...")
    try:
        parsed = parse_pdf(pdf_path)
    except Exception as e:
        print(f"  ❌ Parsing failed: {e}")
        return {"status": "error", "reason": f"Parsing failed: {str(e)}",
                "document_name": Path(pdf_path).stem}

    content_hash = parsed["content_hash"]
    print(f"  📊 Content hash: {content_hash[:16]}...")

    tracker = load_embedding_tracker()
    needs_embed, prev_hash = should_embed(pdf_path, content_hash, tracker, check_mode)

    if not needs_embed:
        print(f"  ⏭️  SKIPPED – unchanged (hash: {content_hash[:16]}...)")
        return {"status": "skipped", "reason": "Content unchanged",
                "document_name": parsed["document_name"], "content_hash": content_hash}

    if prev_hash:
        print(f"  📝 Changed! Previous: {prev_hash[:16]}...")
        delete_old_embeddings(parsed["document_name"])
    else:
        print("  🆕 New document")

    chunks_count = embed_document(parsed)
    update_tracker(pdf_path, content_hash, tracker, chunks_count)
    print("  💾 Tracker updated")

    return {"status": "embedded", "document_name": parsed["document_name"],
            "content_hash": content_hash, "chunks_count": chunks_count}


# BATCH PROCESSING

def process_folder(folder_path: Path, check_mode: str = "content", dry_run: bool = False) -> None:
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in {folder_path}")
        return

    print(f"\n📁 Found {len(pdf_files)} PDF(s) in {folder_path}")
    if dry_run:
        print("🏁 DRY RUN – no embedding will be performed.")

    stats = {"total": len(pdf_files), "embedded": 0, "skipped": 0, "errors": 0}

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] ", end="")

        if dry_run:
            tracker = load_embedding_tracker()
            parsed = parse_pdf(str(pdf_file))
            needs_embed, _ = should_embed(str(pdf_file), parsed["content_hash"], tracker, check_mode)
            status = "needs embedding" if needs_embed else "up to date"
            print(f"📄 {pdf_file.name}: {status}")
            continue

        result = process_pdf(str(pdf_file), check_mode)
        status = result.get("status", "unknown")
        if status == "embedded":
            stats["embedded"] += 1
        elif status == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"] += 1
            print(f"  ❌ Error: {result.get('reason', 'Unknown')}")

        gc.collect()  # free memory after each file

    print("\n" + "═"*60)
    print("📊 BATCH SUMMARY")
    print(f"   Total files   : {stats['total']}")
    print(f"   Embedded      : {stats['embedded']}")
    print(f"   Skipped       : {stats['skipped']}")
    print(f"   Errors        : {stats['errors']}")
    print("═"*60)


# MAIN

def main():
    parser = argparse.ArgumentParser(description="PDF embedding pipeline")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_PDF_FOLDER),
                        help="Folder containing PDF files")
    parser.add_argument("--mode", choices=["content", "file", "both"], default="content",
                        help="Change detection mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show which files need processing, do not embed")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    process_folder(folder, check_mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()