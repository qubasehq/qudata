"""
Quality analysis module for multi-dimensional quality scoring.

This module provides comprehensive quality analysis including content quality scoring,
format validation, readability assessment, and quality reporting.
"""

import re
import math
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from ..models import Document

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Multi-dimensional quality score for content."""
    overall_score: float  # 0.0 to 1.0
    length_score: float
    language_score: float
    coherence_score: float
    uniqueness_score: float
    readability_score: float
    structure_score: float
    completeness_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "length_score": self.length_score,
            "language_score": self.language_score,
            "coherence_score": self.coherence_score,
            "uniqueness_score": self.uniqueness_score,
            "readability_score": self.readability_score,
            "structure_score": self.structure_score,
            "completeness_score": self.completeness_score
        }


@dataclass
class QualityIssue:
    """Represents a quality issue found in content."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion
        }


@dataclass
class DocumentQuality:
    """Quality analysis result for a document."""
    document_id: str
    quality_score: QualityScore
    issues: List[QualityIssue] = field(default_factory=list)
    quality_grade: str = "C"  # A, B, C, D, F
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "quality_score": self.quality_score.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "quality_grade": self.quality_grade
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for a document collection."""
    document_qualities: List[DocumentQuality]
    overall_quality_score: float
    quality_distribution: Dict[str, int]  # Grade distribution
    common_issues: List[Tuple[str, int]]  # (issue_type, count)
    quality_statistics: Dict[str, float]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_qualities": [dq.to_dict() for dq in self.document_qualities],
            "overall_quality_score": self.overall_quality_score,
            "quality_distribution": self.quality_distribution,
            "common_issues": self.common_issues,
            "quality_statistics": self.quality_statistics,
            "recommendations": self.recommendations
        }


class QualityAnalyzer:
    """
    Multi-dimensional quality analyzer for content assessment.
    
    Provides comprehensive quality scoring including length, language,
    coherence, uniqueness, readability, structure, and completeness analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quality analyzer.
        
        Args:
            config: Configuration dictionary with quality parameters
        """
        self.config = config or {}
        
        # Quality thresholds
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 50000)
        self.optimal_length_min = self.config.get("optimal_length_min", 200)
        self.optimal_length_max = self.config.get("optimal_length_max", 5000)
        
        # Scoring weights
        self.weights = {
            'length': self.config.get("length_weight", 0.15),
            'language': self.config.get("language_weight", 0.15),
            'coherence': self.config.get("coherence_weight", 0.20),
            'uniqueness': self.config.get("uniqueness_weight", 0.15),
            'readability': self.config.get("readability_weight", 0.15),
            'structure': self.config.get("structure_weight", 0.10),
            'completeness': self.config.get("completeness_weight", 0.10)
        }
        
        # Quality patterns and indicators
        self.quality_patterns = self._load_quality_patterns()
        self.spam_indicators = self._load_spam_indicators()
        self.completeness_indicators = self._load_completeness_indicators()
        
        # Language detection (simplified)
        self.language_confidence_threshold = self.config.get("language_confidence_threshold", 0.7)
    
    def _load_quality_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate quality issues."""
        return {
            'repetitive': [
                r'(.{10,})\1{3,}',  # Repeated sequences
                r'\b(\w+)\s+\1\s+\1\b',  # Repeated words
                r'(.{50,})\n\1',  # Repeated lines
            ],
            'low_quality': [
                r'\b(click here|buy now|free|!!+|amazing|incredible)\b',
                r'[A-Z]{5,}',  # Excessive caps
                r'[!]{3,}|[?]{3,}',  # Excessive punctuation
                r'\$\d+|\d+%\s*off',  # Price/discount indicators
            ],
            'incomplete': [
                r'\.\.\.$',  # Trailing ellipsis
                r'\[.*\]$',  # Placeholder brackets
                r'TODO|FIXME|XXX',  # Development placeholders
                r'Lorem ipsum',  # Placeholder text
            ],
            'formatting_issues': [
                r'\s{3,}',  # Excessive whitespace
                r'\n{4,}',  # Excessive line breaks
                r'[^\w\s]{5,}',  # Excessive special characters
            ]
        }
    
    def _load_spam_indicators(self) -> List[str]:
        """Load spam and low-quality content indicators."""
        return [
            'click here', 'buy now', 'limited time', 'act now', 'free trial',
            'no obligation', 'risk free', 'money back', 'guarantee',
            'winner', 'congratulations', 'selected', 'urgent', 'immediate',
            'call now', 'don\'t wait', 'order now', 'special offer',
            'exclusive deal', 'save money', 'earn money', 'make money'
        ]
    
    def _load_completeness_indicators(self) -> Dict[str, List[str]]:
        """Load indicators of content completeness."""
        return {
            'introduction': ['introduction', 'overview', 'summary', 'abstract'],
            'conclusion': ['conclusion', 'summary', 'in conclusion', 'to summarize'],
            'structure': ['first', 'second', 'third', 'finally', 'next', 'then'],
            'references': ['reference', 'source', 'citation', 'bibliography'],
            'examples': ['example', 'for instance', 'such as', 'including']
        }
    
    def analyze_quality(self, documents: List[Document]) -> QualityReport:
        """
        Analyze quality for a collection of documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            QualityReport with comprehensive quality analysis
        """
        if not documents:
            return self._empty_report()
        
        document_qualities = []
        all_scores = []
        issue_counts = defaultdict(int)
        grade_counts = defaultdict(int)
        
        # Analyze each document
        for doc in documents:
            doc_quality = self.analyze_document_quality(doc, documents)
            document_qualities.append(doc_quality)
            
            all_scores.append(doc_quality.quality_score.overall_score)
            grade_counts[doc_quality.quality_grade] += 1
            
            # Count issues
            for issue in doc_quality.issues:
                issue_counts[issue.issue_type] += 1
        
        # Calculate overall statistics
        overall_score = sum(all_scores) / len(all_scores)
        
        quality_statistics = {
            'mean_score': overall_score,
            'min_score': min(all_scores),
            'max_score': max(all_scores),
            'std_score': self._calculate_std(all_scores),
            'high_quality_percentage': (grade_counts['A'] + grade_counts['B']) / len(documents) * 100,
            'low_quality_percentage': (grade_counts['D'] + grade_counts['F']) / len(documents) * 100
        }
        
        # Get common issues
        common_issues = Counter(issue_counts).most_common(10)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(document_qualities, common_issues)
        
        return QualityReport(
            document_qualities=document_qualities,
            overall_quality_score=overall_score,
            quality_distribution=dict(grade_counts),
            common_issues=common_issues,
            quality_statistics=quality_statistics,
            recommendations=recommendations
        )
    
    def analyze_document_quality(self, document: Document, corpus: List[Document] = None) -> DocumentQuality:
        """
        Analyze quality for a single document.
        
        Args:
            document: Document to analyze
            corpus: Full document corpus for uniqueness analysis
            
        Returns:
            DocumentQuality with detailed quality assessment
        """
        content = document.content
        issues = []
        
        # Calculate individual quality scores
        length_score = self._calculate_length_score(content, issues)
        language_score = self._calculate_language_score(content, issues)
        coherence_score = self._calculate_coherence_score(content, issues)
        uniqueness_score = self._calculate_uniqueness_score(document, corpus or [document], issues)
        readability_score = self._calculate_readability_score(content, issues)
        structure_score = self._calculate_structure_score(content, document.structure, issues)
        completeness_score = self._calculate_completeness_score(content, issues)
        
        # Calculate overall score using weights
        overall_score = (
            self.weights['length'] * length_score +
            self.weights['language'] * language_score +
            self.weights['coherence'] * coherence_score +
            self.weights['uniqueness'] * uniqueness_score +
            self.weights['readability'] * readability_score +
            self.weights['structure'] * structure_score +
            self.weights['completeness'] * completeness_score
        )
        
        quality_score = QualityScore(
            overall_score=overall_score,
            length_score=length_score,
            language_score=language_score,
            coherence_score=coherence_score,
            uniqueness_score=uniqueness_score,
            readability_score=readability_score,
            structure_score=structure_score,
            completeness_score=completeness_score
        )
        
        # Assign quality grade
        quality_grade = self._assign_quality_grade(overall_score)
        
        return DocumentQuality(
            document_id=document.id,
            quality_score=quality_score,
            issues=issues,
            quality_grade=quality_grade
        )
    
    def _calculate_length_score(self, content: str, issues: List[QualityIssue]) -> float:
        """Calculate length-based quality score."""
        length = len(content)
        word_count = len(content.split())
        
        # Check minimum length
        if length < self.min_length:
            issues.append(QualityIssue(
                issue_type="length",
                severity="high",
                description=f"Content too short ({length} characters, minimum {self.min_length})",
                suggestion="Add more detailed content"
            ))
            return 0.2
        
        # Check maximum length
        if length > self.max_length:
            issues.append(QualityIssue(
                issue_type="length",
                severity="medium",
                description=f"Content very long ({length} characters, maximum {self.max_length})",
                suggestion="Consider breaking into smaller sections"
            ))
            return 0.7
        
        # Optimal length scoring
        if self.optimal_length_min <= length <= self.optimal_length_max:
            return 1.0
        elif length < self.optimal_length_min:
            # Gradual decrease below optimal
            ratio = length / self.optimal_length_min
            return max(0.5, ratio)
        else:
            # Gradual decrease above optimal
            excess_ratio = (length - self.optimal_length_max) / self.optimal_length_max
            return max(0.6, 1.0 - excess_ratio * 0.4)
    
    def _calculate_language_score(self, content: str, issues: List[QualityIssue]) -> float:
        """Calculate language quality score."""
        score = 1.0
        
        # Check for mixed languages (simplified)
        non_ascii_ratio = sum(1 for c in content if ord(c) > 127) / len(content) if content else 0
        
        if non_ascii_ratio > 0.3:
            issues.append(QualityIssue(
                issue_type="language",
                severity="medium",
                description="High ratio of non-ASCII characters detected",
                suggestion="Ensure consistent language usage"
            ))
            score *= 0.8
        
        # Check for encoding issues
        if '�' in content or '\ufffd' in content:
            issues.append(QualityIssue(
                issue_type="language",
                severity="high",
                description="Encoding issues detected (replacement characters found)",
                suggestion="Fix text encoding"
            ))
            score *= 0.5
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content) if content else 0
        
        if special_char_ratio > 0.2:
            issues.append(QualityIssue(
                issue_type="language",
                severity="low",
                description="High ratio of special characters",
                suggestion="Review and clean special characters"
            ))
            score *= 0.9
        
        return max(0.0, score)
    
    def _calculate_coherence_score(self, content: str, issues: List[QualityIssue]) -> float:
        """Calculate coherence and flow quality score."""
        score = 1.0
        
        # Check for repetitive patterns
        for pattern_type, patterns in self.quality_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    if pattern_type == 'repetitive':
                        issues.append(QualityIssue(
                            issue_type="coherence",
                            severity="medium",
                            description=f"Repetitive content detected ({len(matches)} instances)",
                            suggestion="Remove or rephrase repetitive sections"
                        ))
                        score *= 0.7
                    elif pattern_type == 'low_quality':
                        issues.append(QualityIssue(
                            issue_type="coherence",
                            severity="high",
                            description="Low-quality content patterns detected",
                            suggestion="Remove promotional or spam-like content"
                        ))
                        score *= 0.5
        
        # Check sentence variety
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            
            # Low variance indicates poor sentence variety
            if length_variance < 5:
                issues.append(QualityIssue(
                    issue_type="coherence",
                    severity="low",
                    description="Low sentence length variety",
                    suggestion="Vary sentence lengths for better readability"
                ))
                score *= 0.9
        
        return max(0.0, score)
    
    def _calculate_uniqueness_score(self, document: Document, corpus: List[Document], issues: List[QualityIssue]) -> float:
        """Calculate content uniqueness score."""
        if len(corpus) <= 1:
            return 1.0
        
        content = document.content.lower()
        content_words = set(content.split())
        
        max_similarity = 0.0
        similar_docs = 0
        
        for other_doc in corpus:
            if other_doc.id == document.id:
                continue
                
            other_content = other_doc.content.lower()
            other_words = set(other_content.split())
            
            # Calculate Jaccard similarity
            if content_words and other_words:
                intersection = len(content_words & other_words)
                union = len(content_words | other_words)
                similarity = intersection / union if union > 0 else 0.0
                
                max_similarity = max(max_similarity, similarity)
                
                if similarity > 0.7:
                    similar_docs += 1
        
        # Score based on uniqueness
        uniqueness_score = 1.0 - max_similarity
        
        if max_similarity > 0.8:
            issues.append(QualityIssue(
                issue_type="uniqueness",
                severity="high",
                description=f"Very similar content detected (similarity: {max_similarity:.2f})",
                suggestion="Ensure content originality"
            ))
        elif max_similarity > 0.6:
            issues.append(QualityIssue(
                issue_type="uniqueness",
                severity="medium",
                description=f"Similar content detected (similarity: {max_similarity:.2f})",
                suggestion="Review for duplicate content"
            ))
        
        if similar_docs > 0:
            issues.append(QualityIssue(
                issue_type="uniqueness",
                severity="medium",
                description=f"Content similar to {similar_docs} other documents",
                suggestion="Increase content diversity"
            ))
        
        return max(0.0, uniqueness_score)
    
    def _calculate_readability_score(self, content: str, issues: List[QualityIssue]) -> float:
        """Calculate readability quality score."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        words = content.split()
        if not words:
            return 0.0
        
        # Calculate basic readability metrics
        avg_sentence_length = len(words) / len(sentences)
        
        # Estimate syllables (simple heuristic)
        total_syllables = 0
        for word in words:
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
            total_syllables += syllables
        
        avg_syllables_per_word = total_syllables / len(words)
        
        # Simplified Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        normalized_score = max(0.0, min(1.0, flesch_score / 100))
        
        # Check for readability issues
        if avg_sentence_length > 25:
            issues.append(QualityIssue(
                issue_type="readability",
                severity="medium",
                description=f"Long average sentence length ({avg_sentence_length:.1f} words)",
                suggestion="Use shorter sentences for better readability"
            ))
        
        if avg_syllables_per_word > 2.0:
            issues.append(QualityIssue(
                issue_type="readability",
                severity="low",
                description="Complex vocabulary detected",
                suggestion="Consider using simpler words where appropriate"
            ))
        
        return normalized_score
    
    def _calculate_structure_score(self, content: str, structure, issues: List[QualityIssue]) -> float:
        """Calculate structural quality score."""
        score = 1.0
        
        # Check for basic structure elements
        has_paragraphs = '\n\n' in content or structure.paragraphs > 1
        has_headings = len(structure.headings) > 0 if structure else False
        
        if not has_paragraphs and len(content) > 500:
            issues.append(QualityIssue(
                issue_type="structure",
                severity="medium",
                description="No paragraph breaks in long content",
                suggestion="Add paragraph breaks for better structure"
            ))
            score *= 0.8
        
        # Check for formatting issues
        for pattern_type, patterns in self.quality_patterns.items():
            if pattern_type == 'formatting_issues':
                for pattern in patterns:
                    if re.search(pattern, content):
                        issues.append(QualityIssue(
                            issue_type="structure",
                            severity="low",
                            description="Formatting issues detected",
                            suggestion="Clean up spacing and formatting"
                        ))
                        score *= 0.9
                        break
        
        # Bonus for good structure
        if has_headings:
            score = min(1.0, score * 1.1)
        
        return max(0.0, score)
    
    def _calculate_completeness_score(self, content: str, issues: List[QualityIssue]) -> float:
        """Calculate content completeness score."""
        score = 1.0
        content_lower = content.lower()
        
        # Check for incomplete content indicators
        for pattern_type, patterns in self.quality_patterns.items():
            if pattern_type == 'incomplete':
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(QualityIssue(
                            issue_type="completeness",
                            severity="high",
                            description="Incomplete content indicators found",
                            suggestion="Complete the content and remove placeholders"
                        ))
                        score *= 0.6
                        break
        
        # Check for completeness indicators
        completeness_indicators_found = 0
        for indicator_type, indicators in self.completeness_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    completeness_indicators_found += 1
                    break
        
        # Bonus for completeness indicators
        completeness_bonus = min(0.2, completeness_indicators_found * 0.05)
        score = min(1.0, score + completeness_bonus)
        
        return max(0.0, score)
    
    def _assign_quality_grade(self, overall_score: float) -> str:
        """Assign letter grade based on overall score."""
        if overall_score >= 0.9:
            return 'A'
        elif overall_score >= 0.8:
            return 'B'
        elif overall_score >= 0.7:
            return 'C'
        elif overall_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _generate_recommendations(self, document_qualities: List[DocumentQuality], 
                                common_issues: List[Tuple[str, int]]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Overall quality recommendations
        low_quality_count = sum(1 for dq in document_qualities if dq.quality_score.overall_score < 0.6)
        if low_quality_count > len(document_qualities) * 0.2:
            recommendations.append("Consider implementing stricter quality filters during data collection")
        
        # Issue-specific recommendations
        for issue_type, count in common_issues[:5]:
            if issue_type == "length" and count > len(document_qualities) * 0.3:
                recommendations.append("Review content length requirements and filtering criteria")
            elif issue_type == "coherence" and count > len(document_qualities) * 0.2:
                recommendations.append("Implement better content deduplication and spam filtering")
            elif issue_type == "uniqueness" and count > len(document_qualities) * 0.15:
                recommendations.append("Strengthen duplicate detection and content diversity measures")
            elif issue_type == "readability" and count > len(document_qualities) * 0.25:
                recommendations.append("Consider readability preprocessing to improve text clarity")
            elif issue_type == "structure" and count > len(document_qualities) * 0.3:
                recommendations.append("Implement better text formatting and structure normalization")
        
        # Grade distribution recommendations
        grade_counts = Counter(dq.quality_grade for dq in document_qualities)
        if grade_counts.get('F', 0) + grade_counts.get('D', 0) > len(document_qualities) * 0.3:
            recommendations.append("Consider raising quality thresholds to exclude low-quality content")
        
        if not recommendations:
            recommendations.append("Overall quality is good. Continue current quality practices.")
        
        return recommendations
    
    def _empty_report(self) -> QualityReport:
        """Return empty report for edge cases."""
        return QualityReport(
            document_qualities=[],
            overall_quality_score=0.0,
            quality_distribution={},
            common_issues=[],
            quality_statistics={},
            recommendations=["No documents to analyze"]
        )
    
    def get_quality_summary(self, report: QualityReport) -> Dict[str, Any]:
        """
        Generate a summary of quality analysis results.
        
        Args:
            report: QualityReport to summarize
            
        Returns:
            Dictionary with quality analysis summary
        """
        if not report.document_qualities:
            return {"message": "No quality data available"}
        
        total_docs = len(report.document_qualities)
        
        # Calculate quality level
        if report.overall_quality_score >= 0.8:
            quality_level = "High"
        elif report.overall_quality_score >= 0.6:
            quality_level = "Medium"
        else:
            quality_level = "Low"
        
        return {
            "total_documents": total_docs,
            "overall_quality_score": round(report.overall_quality_score, 3),
            "quality_level": quality_level,
            "grade_distribution": report.quality_distribution,
            "quality_statistics": {
                k: round(v, 3) for k, v in report.quality_statistics.items()
            },
            "top_issues": report.common_issues[:5],
            "recommendations_count": len(report.recommendations),
            "high_quality_percentage": report.quality_statistics.get('high_quality_percentage', 0),
            "low_quality_percentage": report.quality_statistics.get('low_quality_percentage', 0)
        }