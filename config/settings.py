"""
Configuration settings for the DocRAG pipeline.
All paths and model settings centralized here.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(r"D:\3_Machine learning\GenAI\DocRAG-Observe")
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "output"
TRACKER_PATH = BASE_DIR / "embedding_tracker.json"

# CONFIGURATION
EMBEDDING_TRACKER_PATH = Path(r"D:\3_Machine learning\GenAI\DocRAG-Observe\embedding_tracker.json")

# Ollama Settings 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")      # Embedding model (768-dim)
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_CHAT_MODEL")                # Generation model
OLLAMA_TIMEOUT = 1200  # seconds

# Pinecone Settings 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") 
PINECONE_NAMESPACE = "docrag"
PINECONE_DIMENSION = 768                       # Must match embedding model
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT") 

# Chunking Settings 
PARENT_CHUNK_SIZE = 512       # tokens
CHILD_CHUNK_SIZE = 128        # tokens
OVERLAP_FRACTION = 0.2        # 20% overlap between children

# Retrieval Settings 
TOP_K_RETRIEVE = 10           # Initial retrieval count from Pinecone
TOP_K_RERANK = 5              # After cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# PII Settings 
PII_ENABLED = True
PII_MIN_SCORE = 0.7           # Confidence threshold for PII detection
PII_ENTITIES = [
  "PERSON",
  "EMAIL_ADDRESS",
  "PHONE_NUMBER",
  "LOCATION",
  "ADDRESS",
  "IP_ADDRESS",
  "SOCIAL_SECURITY_NUMBER",
  "PASSPORT_NUMBER",
  "DRIVER_LICENSE_NUMBER",
  "CREDIT_CARD_NUMBER",
  "BANK_ROUTING_NUMBER",
  "BANK_ACCOUNT_NUMBER",
  "TAX_PAYER_IDENTIFICATION_NUMBER",
  "CRYPTO_WALLET_ADDRESS",
  "MEDICAL_RECORD_NUMBER",
  "HEALTH_PLAN_BENEFICIARY_NUMBER",
  "DATE_OF_BIRTH"
]

# LangSmith Settings 
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "doc-rag-observe")


# Generation Settings 
MAX_CONTEXT_TOKENS = 4096
TEMPERATURE = 0.2
MAX_TOKENS = 1024

# Monitoring Settings 
METRICS_MAX_HISTORY = 10000   # Max query metrics to keep in memory
METRICS_WINDOW_MINUTES = 60    # Default time window for metrics