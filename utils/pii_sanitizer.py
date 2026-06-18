"""
PII (Personally Identifiable Information) Detection & Sanitization Module.
Uses Microsoft Presidio for detection and anonymization.
"""
import re
from typing import List, Dict, Any, Optional

# Try to import Presidio 
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("⚠️Presidio not installed. PII detection disabled.")
    print("Install: pip install presidio-analyzer presidio-anonymizer")


class PIISanitizer:
    """
    Detects and sanitizes PII from text before embedding and generation.
    Supports: redaction, replacement with fake data, encryption, hashing.
    """

    def __init__(
        self,
        enabled: bool = True,
        entities: Optional[List[str]] = None,
        min_score: float = 0.7,
        operation: str = "redact",  # "redact" | "replace" | "hash" | "encrypt"
        encryption_key: Optional[str] = None,
    ):
        self.enabled = enabled and PRESIDIO_AVAILABLE
        self.entities = entities or [
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
            "CREDIT_CARD", "US_SSN", "IP_ADDRESS",
            "LOCATION", "DATE_TIME", "NRP",
        ]
        self.min_score = min_score
        self.operation = operation
        self.encryption_key = encryption_key or "default-key-16b"

        if self.enabled:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in text. Returns list of findings."""
        if not self.enabled or not text:
            return []

        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en",
        )

        findings = []
        for result in results:
            if result.score >= self.min_score:
                findings.append({
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": round(result.score, 3),
                    "text": text[result.start:result.end],
                })
        return findings

    def sanitize(self, text: str) -> Dict[str, Any]:
        """
        Sanitize text by removing/redacting PII.
        Returns dict with sanitized text and metadata.
        """
        if not self.enabled or not text:
            return {
                "sanitized_text": text,
                "original_text": text,
                "findings": [],
                "pii_count": 0,
            }

        # Analyze
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en",
        )

        # Filter by score
        analyzer_results = [
            r for r in analyzer_results if r.score >= self.min_score
        ]

        if not analyzer_results:
            return {
                "sanitized_text": text,
                "original_text": text,
                "findings": [],
                "pii_count": 0,
            }

        # Configure anonymization operators
        operators = {}
        for entity in self.entities:
            if self.operation == "redact":
                operators[entity] = OperatorConfig("replace", {"new_value": f"[{entity}]"})
            elif self.operation == "replace":
                operators[entity] = OperatorConfig("replace", {"new_value": f"<REDACTED_{entity}>"})
            elif self.operation == "hash":
                operators[entity] = OperatorConfig("hash", {})
            elif self.operation == "encrypt":
                operators[entity] = OperatorConfig(
                    "encrypt", {"key": self.encryption_key}
                )

        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )

        findings = [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "original": text[r.start:r.end],
            }
            for r in analyzer_results
        ]

        return {
            "sanitized_text": anonymized.text,
            "original_text": text,
            "findings": findings,
            "pii_count": len(findings),
        }

    def sanitize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize a list of chunks, preserving metadata."""
        sanitized_chunks = []
        for chunk in chunks:
            result = self.sanitize(chunk.get("text", ""))
            new_chunk = chunk.copy()
            new_chunk["text"] = result["sanitized_text"]
            new_chunk["pii_findings"] = result["findings"]
            new_chunk["pii_count"] = result["pii_count"]
            new_chunk["has_pii"] = result["pii_count"] > 0
            sanitized_chunks.append(new_chunk)
        return sanitized_chunks

    def get_pii_report(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a PII summary report for a document."""
        total_pii = sum(c.get("pii_count", 0) for c in chunks)
        chunks_with_pii = sum(1 for c in chunks if c.get("has_pii", False))

        entity_types = {}
        for chunk in chunks:
            for finding in chunk.get("pii_findings", []):
                etype = finding["type"]
                entity_types[etype] = entity_types.get(etype, 0) + 1

        return {
            "total_chunks": len(chunks),
            "chunks_with_pii": chunks_with_pii,
            "total_pii_instances": total_pii,
            "entity_type_breakdown": entity_types,
            "sanitization_enabled": self.enabled,
            "operation": self.operation,
        }


# Fallback: Regex-based PII detection (no Presidio) 

class RegexPIISanitizer:
    """Lightweight regex-based PII detection (fallback when Presidio unavailable)."""

    PATTERNS = {
        "EMAIL_ADDRESS": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        "PHONE_NUMBER": re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
        "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        "US_SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def sanitize(self, text: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"sanitized_text": text, "findings": [], "pii_count": 0}

        findings = []
        sanitized = text
        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,
                    "original": match.group(),
                })
                sanitized = sanitized.replace(match.group(), f"[{entity_type}]")

        return {
            "sanitized_text": sanitized,
            "original_text": text,
            "findings": findings,
            "pii_count": len(findings),
        }


def get_sanitizer(config: Dict[str, Any] = None) -> PIISanitizer:
    """Factory function to get the appropriate sanitizer."""
    if PRESIDIO_AVAILABLE:
        return PIISanitizer(
            enabled=config.get("PII_ENABLED", True) if config else True,
            entities=config.get("PII_ENTITIES") if config else None,
            min_score=config.get("PII_MIN_SCORE", 0.7) if config else 0.7,
        )
    else:
        return RegexPIISanitizer()