"""
LangSmith Tracing Integration for DocRAG Pipeline.
Provides automatic trace collection for all RAG operations.
"""
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from config.settings import *
try:
    from langsmith import Client
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("⚠️LangSmith not installed. Tracing disabled.")
    print("Install: pip install langsmith")


@dataclass
class TraceConfig:
    project_name: str = LANGCHAIN_PROJECT
    api_key: str = LANGCHAIN_API_KEY
    endpoint: str = LANGCHAIN_ENDPOINT
    tracing_enabled: bool = LANGCHAIN_TRACING_V2
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGTracer:
    """
    LangSmith tracer for RAG pipeline observability.
    Captures: retrieval, reranking, generation, PII, latency, tokens.
    """

    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()
        self.config.api_key = self.config.api_key or os.getenv("LANGCHAIN_API_KEY", "")
        self.config.tracing_enabled = self.config.tracing_enabled and LANGSMITH_AVAILABLE

        if self.config.tracing_enabled and not self.config.api_key:
            print("⚠️LANGCHAIN_API_KEY not set. Tracing disabled.")
            self.config.tracing_enabled = False

        self.client = Client(
            api_key=self.config.api_key,
            api_url=self.config.endpoint,
        ) if self.config.tracing_enabled else None

        self._current_run: Optional[RunTree] = None

    @contextmanager
    def trace_run(self, name: str, run_type: str = "chain", inputs: Dict[str, Any] = None):
        """Context manager for tracing a RAG run."""
        if not self.config.tracing_enabled or not self.client:
            yield None
            return

        run_id = str(uuid.uuid4())
        start_time = time.time()

        run = self.client.create_run(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            project_name=self.config.project_name,
            tags=self.config.tags,
            extra=self.config.metadata,
        )

        self._current_run = run

        try:
            yield run
            # On success
            run.end(outputs={"status": "success"}, end_time=datetime.now())
        except Exception as e:
            # On error
            run.end(
                outputs={"status": "error", "error": str(e)},
                end_time=datetime.now(),
                error=str(e),
            )
            raise
        finally:
            self._current_run = None
            latency_ms = (time.time() - start_time) * 1000
            if run:
                run.update(extra={"latency_ms": latency_ms})

    @contextmanager
    def trace_step(self, name: str, run_type: str = "chain", inputs: Dict[str, Any] = None):
        """Trace a sub-step (retrieval, rerank, generation, etc.)."""
        if not self.config.tracing_enabled or not self.client or not self._current_run:
            yield None
            return

        start_time = time.time()
        step_run = self.client.create_run(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            parent_run_id=self._current_run.id,
            project_name=self.config.project_name,
        )

        try:
            yield step_run
            step_run.end(outputs={"status": "success"}, end_time=datetime.now())
        except Exception as e:
            step_run.end(
                outputs={"status": "error", "error": str(e)},
                end_time=datetime.now(),
                error=str(e),
            )
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            if step_run:
                step_run.update(extra={"latency_ms": latency_ms})

    def log_metric(self, key: str, value: float, step_run=None) -> None:
        """Log a custom metric to the current trace."""
        if not self.config.tracing_enabled:
            return
        target = step_run or self._current_run
        if target:
            target.update(extra={key: value})

    def log_feedback(self, run_id: str, key: str, score: float, comment: str = "") -> None:
        """Log human/LLM feedback for a run."""
        if not self.config.tracing_enabled or not self.client:
            return
        self.client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )

    def create_dataset(self, name: str, description: str = "") -> Any:
        """Create a dataset for evaluation."""
        if not self.config.tracing_enabled or not self.client:
            return None
        return self.client.create_dataset(
            dataset_name=name,
            description=description,
        )

    def add_example(self, dataset_id: str, inputs: Dict, outputs: Dict, metadata: Dict = None) -> None:
        """Add an example to a dataset."""
        if not self.config.tracing_enabled or not self.client:
            return
        self.client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_id,
            metadata=metadata or {},
        )


def get_tracer(project_name: str = "doc-rag-observe") -> RAGTracer:
    """Factory function to get configured tracer."""
    return RAGTracer(TraceConfig(
        project_name=project_name,
        api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        tracing_enabled=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
    ))