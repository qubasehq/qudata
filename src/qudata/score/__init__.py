"""
Quality scoring module for data quality assessment.

Provides a minimal, dependency-free scorer used by the pipeline. It estimates
an overall quality score based on simple heuristics (length, language tag,
and basic content checks). This can be extended later with advanced models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QualityScoreResult:
    overall_score: float
    length_score: float = 0.0
    language_score: float = 0.0
    cleanliness_score: float = 0.0


class QualityScorer:
    """Simple heuristic-based quality scorer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Accept a dict from PipelineConfig.model_dump() (or None)
        self.config = config or {}
        # Tunables with safe defaults
        self.target_length = int(self.config.get("target_length", 800))  # tokens/chars proxy
        self.min_quality = float(self.config.get("min_quality_score", 0.0))

    def score_document(self, document) -> QualityScoreResult:
        """Return a QualityScoreResult with an overall_score attribute.

        Scoring components:
        - length_score: grows with content length up to target_length
        - language_score: 1.0 if language is known and not 'unknown', else 0.5
        - cleanliness_score: penalize excessive whitespace/non-printables
        """
        content = getattr(document, "content", "") or ""
        meta = getattr(document, "metadata", None)

        # Length score (0..1), saturates at target_length
        length = len(content)
        length_score = max(0.0, min(1.0, length / max(1, self.target_length)))

        # Language score
        language = getattr(meta, "language", "unknown") if meta else "unknown"
        language_score = 1.0 if (language and language.lower() != "unknown") else 0.5

        # Cleanliness score: penalize if too many non-printables or control chars
        non_printables = sum(1 for ch in content if ord(ch) < 9 or (13 < ord(ch) < 32))
        ratio = non_printables / max(1, length)
        cleanliness_score = max(0.0, 1.0 - min(1.0, ratio * 10))  # harsh penalty if lots of control chars

        # Overall: simple average
        overall = (length_score + language_score + cleanliness_score) / 3.0

        # Enforce minimum quality if configured
        if overall < self.min_quality:
            overall = self.min_quality

        return QualityScoreResult(
            overall_score=round(overall, 4),
            length_score=round(length_score, 4),
            language_score=round(language_score, 4),
            cleanliness_score=round(cleanliness_score, 4),
        )


__all__ = ["QualityScorer", "QualityScoreResult"]