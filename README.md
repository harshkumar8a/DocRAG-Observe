# DocRAG-Observe

Production‑ready RAG (Retrieval‑Augmented Generation) pipeline with full observability, evaluation, and multi‑PDF support.
Built with Ollama, Pinecone, and LangSmith – designed for real‑world document Q&A.

## 🚀 Features

1. 📄 Robust PDF parsing – uses pdfplumber by default (memory‑safe); optional Docling for small PDFs.

2. ✂️ Parent‑child chunking – hierarchical chunks for richer context during retrieval.

3. 🔒 PII sanitization – automatically detects and redacts sensitive information.

4. 🧠 Embedding & Generation – powered by Ollama (local models) – configurable.

5. ☁️ Pinecone vector store – cloud‑native, scalable, with metadata filtering.

6. ⚖️ Cross‑encoder reranking – improves retrieval quality.

7. 🔍 Full observability – LangSmith tracing, custom metrics (latency, tokens), health monitoring.

8. 📊 RAGAS evaluation – assess answer quality (faithfulness, relevance, etc.).

9. 📁 Batch indexing – process entire folders of PDFs, skip unchanged documents.

10. 🧪 CLI interface – index, query, dry‑run, force re‑index, batch questions from file.


## 🛠️ Tech Stack

1. Language: Python 3.10+

2. Document parsing: pdfplumber (primary), docling (optional)

3. Embedding & LLM: Ollama (nomic-embed-text, llama3.2)

4. Vector DB: Pinecone

5. Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

6. Tracing: LangSmith

7. Evaluation: RAGAS

8. Monitoring: Custom metrics collector (in‑memory + JSON persistence)


## 📦 Prerequisites

* Python 3.10 or higher

* Ollama installed and running locally (or accessible via network)

* Pinecone account and API key (free tier available)

* (Optional) LangSmith API key for tracing

* (Optional) Hugging Face token for RAGAS models


## 🔧 Installation

    # Clone the repository
    git clone https://github.com/yourusername/DocRAG-Observe.git
    cd DocRAG-Observe

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    # or .venv\Scripts\activate  # Windows

    uv add docling pdfplumber pinecone langsmith ragas datasets ollama pypdf python-dotenv

## ⚙️ Configuration

    # Ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_EMBED_MODEL=nomic-embed-text
    OLLAMA_LLM_MODEL=llama3.2:3b

    # Pinecone
    PINECONE_API_KEY=your-pinecone-api-key
    PINECONE_INDEX_NAME=docrag-obs
    PINECONE_NAMESPACE=docrag
    PINECONE_DIMENSION=768

    # LangSmith (optional)
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your-langsmith-key
    LANGCHAIN_PROJECT=doc-rag-production

    # Paths
    TRACKER_PATH=./embedding_tracker.json
    METRIX_PATH=./metrics.json

## 📂 Project Structure

    DocRAG-Observe/
    ├── core/
    │   ├── rag_pipeline.py          # Main RAGPipeline class
    │   ├── embedding.py             # Ollama embedding client
    │   ├── generation.py            # Ollama generation client
    │   ├── vector_store.py          # Pinecone wrapper with batching
    │   ├── reranker.py              # Cross‑encoder reranker
    │   ├── langsmith_tracing.py     # LangSmith tracer wrapper
    │   ├── evaluation.py            # RAGAS evaluator
    │   └── monitoring.py            # Metrics collector (QueryMetrics, IndexingMetrics)
    ├── utils/
    │   ├── pdf_parser.py            # pdfplumber/Docling parser (memory‑safe)
    │   ├── text_cleaner.py          # Clean extracted text
    │   ├── chunker.py               # Parent‑child chunking logic
    │   └── pii_sanitizer.py         # PII detection and redaction
    ├── config/
    │   └── settings.py              # Configuration constants
    ├── documents/                   # Place your PDFs here
    ├── .env                         # Environment variables
    ├── pyproject.toml
    |__ uv.lock
    └── README.md

## 🚀 Usage

### 1. Start Ollama
Ensure Ollama is running and the required models are pulled:

    ollama pull nomic-embed-text
    ollama pull llama3.2:3b

### 2. Index PDFs
Index all PDFs in the default documents folder:

    python -m core.rag_pipeline --folder documents


### 3. Query
Ask a single question (after indexing):

    python -m core.rag_pipeline --query "What is artificial intelligence?"

### 4. Full pipeline (index + query)

    python -m core.rag_pipeline --folder documents --query "What is deep learning?"



## 🧪 Observability & Monitoring

**LangSmith**: All indexing and query operations are traced. Log in to your LangSmith dashboard to inspect runs, latency, and token usage.

**Metrics**: The pipeline collects:

*   Query latency, tokens, retrieval/reranking stats.

*   Indexing latency per document, chunk counts, PII instances.

*   Health status (error rates, uptime, total queries).

Export metrics:

    rag.export_metrics("metrics.json")


## 📊 Evaluation (RAGAS)

If ragas is installed, the pipeline automatically evaluates each query on:

*   Faithfulness (answer grounded in context)

*   Answer relevancy

*   Context relevancy

*   Context recall

Run queries with evaluate=True (default) to include evaluation metrics in the response.


    python -m core.rag_pipeline --query "Explain neural networks"

Evaluation results appear in the CLI output and are stored in the evaluation report.


## 🐛 Troubleshooting
1. **std::bad_alloc during parsing**

    The pipeline uses pdfplumber by default – this error should not occur. If you enabled Docling, disable it in utils/pdf_parser.py or set USE_DOCLING_FOR_SMALL_PDFS = False.

2. **Pinecone payload too large**

    The add() method now batches upserts (default 100 vectors per request). Adjust batch_size in the call if needed.

3. **'NoneType' object has no attribute 'end'**

    LangSmith tracer is not properly configured. Set LANGCHAIN_TRACING_V2=false or provide a valid API key. The pipeline also has conditional tracing to prevent this error.

4. **No vectors found despite indexing**

    Check that the PINECONE_INDEX_NAME and PINECONE_NAMESPACE in your .env match the ones used during indexing. Use --force to re‑index if needed.

