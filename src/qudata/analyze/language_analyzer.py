"""
Language analysis module for language detection and distribution statistics.

This module provides comprehensive language analysis including language detection,
distribution statistics, and multilingual content analysis.
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from ..models import Document

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """Result of language detection for text."""
    language: str
    confidence: float
    iso_code: str
    script: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "confidence": self.confidence,
            "iso_code": self.iso_code,
            "script": self.script
        }


@dataclass
class DocumentLanguage:
    """Language analysis result for a document."""
    document_id: str
    primary_language: LanguageDetection
    detected_languages: List[LanguageDetection] = field(default_factory=list)
    is_multilingual: bool = False
    language_segments: List[Tuple[str, str, float]] = field(default_factory=list)  # (text, lang, confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "primary_language": self.primary_language.to_dict(),
            "detected_languages": [lang.to_dict() for lang in self.detected_languages],
            "is_multilingual": self.is_multilingual,
            "language_segments": [
                {"text": text[:100], "language": lang, "confidence": conf}
                for text, lang, conf in self.language_segments
            ]
        }


@dataclass
class LanguageDistribution:
    """Language distribution statistics for a document collection."""
    total_documents: int
    language_counts: Dict[str, int]
    language_percentages: Dict[str, float]
    multilingual_count: int
    multilingual_percentage: float
    confidence_stats: Dict[str, float]  # min, max, avg confidence
    script_distribution: Dict[str, int]
    dominant_language: str
    language_diversity_index: float  # Shannon diversity index
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_documents": self.total_documents,
            "language_counts": self.language_counts,
            "language_percentages": self.language_percentages,
            "multilingual_count": self.multilingual_count,
            "multilingual_percentage": self.multilingual_percentage,
            "confidence_stats": self.confidence_stats,
            "script_distribution": self.script_distribution,
            "dominant_language": self.dominant_language,
            "language_diversity_index": self.language_diversity_index
        }


class LanguageAnalyzer:
    """
    Language analyzer for detection and distribution statistics.
    
    Provides methods for detecting languages in text, analyzing language
    distributions, and handling multilingual content.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize language analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.min_text_length = self.config.get("min_text_length", 20)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.segment_analysis = self.config.get("segment_analysis", True)
        
        # Language detection libraries availability
        self.langdetect_available = self._check_langdetect()
        self.polyglot_available = self._check_polyglot()
        
        # Load language mappings and patterns
        self.language_names = self._load_language_names()
        self.script_patterns = self._load_script_patterns()
        self.language_patterns = self._load_language_patterns()
    
    def _check_langdetect(self) -> bool:
        """Check if langdetect library is available."""
        try:
            import langdetect
            return True
        except ImportError:
            logger.info("langdetect not available. Using pattern-based language detection.")
            return False
    
    def _check_polyglot(self) -> bool:
        """Check if polyglot library is available."""
        try:
            import polyglot
            return True
        except ImportError:
            logger.info("polyglot not available. Advanced language features will be limited.")
            return False
    
    def _load_language_names(self) -> Dict[str, str]:
        """Load language code to name mappings."""
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'fa': 'Persian',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'hu': 'Hungarian',
            'cs': 'Czech',
            'sk': 'Slovak',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'el': 'Greek',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Filipino',
            'sw': 'Swahili',
            'am': 'Amharic',
            'unknown': 'Unknown'
        }
    
    def _load_script_patterns(self) -> Dict[str, str]:
        """Load script detection patterns."""
        return {
            'latin': r'[a-zA-ZÀ-ÿĀ-žА-я]',
            'cyrillic': r'[А-я]',
            'arabic': r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]',
            'chinese': r'[\u4e00-\u9fff]',
            'japanese': r'[\u3040-\u309F\u30A0-\u30FF\u4e00-\u9fff]',
            'korean': r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]',
            'devanagari': r'[\u0900-\u097F]',
            'thai': r'[\u0E00-\u0E7F]',
            'hebrew': r'[\u0590-\u05FF]',
            'greek': r'[\u0370-\u03FF]'
        }
    
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        """Load language-specific word patterns for basic detection."""
        return {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im'],
            'it': ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'del', 'da', 'a', 'al', 'le', 'si'],
            'pt': ['o', 'de', 'a', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no'],
            'ru': ['в', 'и', 'не', 'на', 'я', 'быть', 'то', 'он', 'с', 'а', 'как', 'по', 'это', 'она', 'к'],
            'zh': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上', '个'],
            'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も'],
            'ar': ['في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'التي', 'التي', 'كان', 'كانت', 'يكون', 'تكون', 'أن', 'أو', 'لا']
        }
    
    def analyze_languages(self, documents: List[Document]) -> LanguageDistribution:
        """
        Analyze language distribution for a collection of documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            LanguageDistribution with comprehensive language statistics
        """
        if not documents:
            return self._empty_distribution()
        
        document_languages = []
        all_confidences = []
        multilingual_count = 0
        script_counts = defaultdict(int)
        
        for doc in documents:
            doc_lang = self.analyze_document_language(doc)
            document_languages.append(doc_lang)
            
            all_confidences.append(doc_lang.primary_language.confidence)
            
            if doc_lang.is_multilingual:
                multilingual_count += 1
            
            # Count scripts
            if doc_lang.primary_language.script:
                script_counts[doc_lang.primary_language.script] += 1
        
        # Calculate language distribution
        language_counts = Counter(doc_lang.primary_language.language for doc_lang in document_languages)
        total_docs = len(documents)
        
        language_percentages = {
            lang: (count / total_docs) * 100
            for lang, count in language_counts.items()
        }
        
        # Find dominant language
        dominant_language = language_counts.most_common(1)[0][0] if language_counts else 'unknown'
        
        # Calculate language diversity (Shannon diversity index)
        diversity_index = self._calculate_diversity_index(language_counts, total_docs)
        
        # Confidence statistics
        confidence_stats = {
            'min': min(all_confidences) if all_confidences else 0.0,
            'max': max(all_confidences) if all_confidences else 0.0,
            'avg': sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        }
        
        return LanguageDistribution(
            total_documents=total_docs,
            language_counts=dict(language_counts),
            language_percentages=language_percentages,
            multilingual_count=multilingual_count,
            multilingual_percentage=(multilingual_count / total_docs) * 100,
            confidence_stats=confidence_stats,
            script_distribution=dict(script_counts),
            dominant_language=dominant_language,
            language_diversity_index=diversity_index
        )
    
    def analyze_document_language(self, document: Document) -> DocumentLanguage:
        """
        Analyze language for a single document.
        
        Args:
            document: Document to analyze
            
        Returns:
            DocumentLanguage with language detection results
        """
        text = document.content
        
        if len(text) < self.min_text_length:
            # Text too short for reliable detection
            return DocumentLanguage(
                document_id=document.id,
                primary_language=LanguageDetection(
                    language='unknown',
                    confidence=0.0,
                    iso_code='unknown'
                )
            )
        
        # Detect primary language
        primary_language = self._detect_language(text)
        
        # Detect all languages if segment analysis is enabled
        detected_languages = [primary_language]
        language_segments = []
        is_multilingual = False
        
        if self.segment_analysis:
            segments = self._segment_text(text)
            segment_languages = []
            
            for segment in segments:
                if len(segment) >= self.min_text_length:
                    seg_lang = self._detect_language(segment)
                    segment_languages.append(seg_lang)
                    language_segments.append((segment, seg_lang.language, seg_lang.confidence))
            
            # Check for multilingual content
            unique_languages = set(lang.language for lang in segment_languages 
                                 if lang.confidence >= self.confidence_threshold)
            
            if len(unique_languages) > 1:
                is_multilingual = True
                detected_languages = list({lang.language: lang for lang in segment_languages}.values())
        
        return DocumentLanguage(
            document_id=document.id,
            primary_language=primary_language,
            detected_languages=detected_languages,
            is_multilingual=is_multilingual,
            language_segments=language_segments
        )
    
    def _detect_language(self, text: str) -> LanguageDetection:
        """
        Detect language of text using available methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetection result
        """
        # Try langdetect first if available
        if self.langdetect_available:
            return self._detect_with_langdetect(text)
        else:
            return self._detect_with_patterns(text)
    
    def _detect_with_langdetect(self, text: str) -> LanguageDetection:
        """Detect language using langdetect library."""
        try:
            from langdetect import detect, detect_langs, LangDetectException
            
            # Get primary language
            primary_lang = detect(text)
            
            # Get confidence scores
            lang_probs = detect_langs(text)
            confidence = 0.0
            
            for lang_prob in lang_probs:
                if lang_prob.lang == primary_lang:
                    confidence = lang_prob.prob
                    break
            
            # Get language name and script
            language_name = self.language_names.get(primary_lang, primary_lang)
            script = self._detect_script(text)
            
            return LanguageDetection(
                language=language_name,
                confidence=confidence,
                iso_code=primary_lang,
                script=script
            )
            
        except (LangDetectException, Exception) as e:
            logger.warning(f"langdetect failed: {e}")
            return self._detect_with_patterns(text)
    
    def _detect_with_patterns(self, text: str) -> LanguageDetection:
        """Detect language using pattern-based approach."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return LanguageDetection(
                language='unknown',
                confidence=0.0,
                iso_code='unknown'
            )
        
        # Score languages based on common word patterns
        language_scores = defaultdict(int)
        
        for lang_code, common_words in self.language_patterns.items():
            for word in words:
                if word in common_words:
                    language_scores[lang_code] += 1
        
        # Detect script
        script = self._detect_script(text)
        
        # Adjust scores based on script
        if script == 'cyrillic':
            language_scores['ru'] += len(words) * 0.1
        elif script == 'arabic':
            language_scores['ar'] += len(words) * 0.1
        elif script == 'chinese':
            language_scores['zh'] += len(words) * 0.1
        elif script == 'japanese':
            language_scores['ja'] += len(words) * 0.1
        elif script == 'korean':
            language_scores['ko'] += len(words) * 0.1
        
        # Find best match
        if language_scores:
            best_lang = max(language_scores.items(), key=lambda x: x[1])
            lang_code = best_lang[0]
            score = best_lang[1]
            confidence = min(1.0, score / len(words))
        else:
            # Default to English for Latin script, unknown otherwise
            if script == 'latin':
                lang_code = 'en'
                confidence = 0.3
            else:
                lang_code = 'unknown'
                confidence = 0.0
        
        language_name = self.language_names.get(lang_code, lang_code)
        
        return LanguageDetection(
            language=language_name,
            confidence=confidence,
            iso_code=lang_code,
            script=script
        )
    
    def _detect_script(self, text: str) -> str:
        """Detect the primary script used in text."""
        script_counts = defaultdict(int)
        
        for script_name, pattern in self.script_patterns.items():
            matches = re.findall(pattern, text)
            script_counts[script_name] = len(matches)
        
        if script_counts:
            return max(script_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    def _segment_text(self, text: str) -> List[str]:
        """Segment text for multilingual analysis."""
        # Simple sentence-based segmentation
        sentences = re.split(r'[.!?]+', text)
        
        # Group sentences into segments
        segments = []
        current_segment = []
        segment_length = 0
        target_length = self.config.get("segment_length", 200)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_segment.append(sentence)
            segment_length += len(sentence)
            
            if segment_length >= target_length:
                segments.append(' '.join(current_segment))
                current_segment = []
                segment_length = 0
        
        # Add remaining segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def _calculate_diversity_index(self, language_counts: Counter, total_docs: int) -> float:
        """Calculate Shannon diversity index for language distribution."""
        if total_docs == 0:
            return 0.0
        
        import math
        diversity = 0.0
        for count in language_counts.values():
            if count > 0:
                proportion = count / total_docs
                diversity -= proportion * math.log2(proportion) if proportion > 0 else 0
        
        return diversity
    
    def _empty_distribution(self) -> LanguageDistribution:
        """Return empty distribution for edge cases."""
        return LanguageDistribution(
            total_documents=0,
            language_counts={},
            language_percentages={},
            multilingual_count=0,
            multilingual_percentage=0.0,
            confidence_stats={'min': 0.0, 'max': 0.0, 'avg': 0.0},
            script_distribution={},
            dominant_language='unknown',
            language_diversity_index=0.0
        )
    
    def get_language_summary(self, distribution: LanguageDistribution) -> Dict[str, Any]:
        """
        Generate a summary of language analysis results.
        
        Args:
            distribution: LanguageDistribution to summarize
            
        Returns:
            Dictionary with language analysis summary
        """
        if distribution.total_documents == 0:
            return {"message": "No language data available"}
        
        # Sort languages by count
        sorted_languages = sorted(
            distribution.language_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top languages
        top_languages = sorted_languages[:5]
        
        # Diversity interpretation
        diversity_level = "Low"
        if distribution.language_diversity_index > 1.5:
            diversity_level = "High"
        elif distribution.language_diversity_index > 0.8:
            diversity_level = "Medium"
        
        return {
            "total_documents": distribution.total_documents,
            "dominant_language": distribution.dominant_language,
            "language_count": len(distribution.language_counts),
            "top_languages": [
                {
                    "language": lang,
                    "count": count,
                    "percentage": round(distribution.language_percentages[lang], 1)
                }
                for lang, count in top_languages
            ],
            "multilingual_documents": {
                "count": distribution.multilingual_count,
                "percentage": round(distribution.multilingual_percentage, 1)
            },
            "language_diversity": {
                "index": round(distribution.language_diversity_index, 3),
                "level": diversity_level
            },
            "confidence_stats": {
                k: round(v, 3) for k, v in distribution.confidence_stats.items()
            },
            "script_distribution": distribution.script_distribution
        }