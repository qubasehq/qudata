"""
Language detection and content filtering module.

This module provides functionality for detecting document languages, filtering content
based on language confidence scores, and normalizing multi-language content.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from ..models import ProcessingError, ErrorSeverity


# Set seed for consistent results
DetectorFactory.seed = 0


@dataclass
class LanguageResult:
    """Result of language detection."""
    language: str
    confidence: float
    is_reliable: bool
    detected_languages: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "confidence": self.confidence,
            "is_reliable": self.is_reliable,
            "detected_languages": self.detected_languages
        }


@dataclass
class FilterResult:
    """Result of language-based content filtering."""
    should_keep: bool
    reason: str
    language_result: LanguageResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_keep": self.should_keep,
            "reason": self.reason,
            "language_result": self.language_result.to_dict()
        }


class LanguageDetector:
    """
    Language detection and content filtering using langdetect library.
    
    Provides functionality to detect document languages, filter content based on
    language confidence scores, and handle multi-language content normalization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize language detector.
        
        Args:
            config: Configuration dictionary with the following options:
                - min_confidence: Minimum confidence threshold (default: 0.7)
                - allowed_languages: List of allowed language codes (default: None = all)
                - min_text_length: Minimum text length for detection (default: 50)
                - max_text_length: Maximum text length to analyze (default: 10000)
                - fallback_language: Language to use when detection fails (default: 'unknown')
        """
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.allowed_languages = self.config.get("allowed_languages")
        self.min_text_length = self.config.get("min_text_length", 50)
        self.max_text_length = self.config.get("max_text_length", 10000)
        self.fallback_language = self.config.get("fallback_language", "unknown")
        
        # Language name mappings for better readability
        self.language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Filipino',
            'he': 'Hebrew',
            'fa': 'Persian',
            'ur': 'Urdu',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'gu': 'Gujarati',
            'pa': 'Punjabi',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'si': 'Sinhala',
            'my': 'Myanmar',
            'km': 'Khmer',
            'lo': 'Lao',
            'ka': 'Georgian',
            'am': 'Amharic',
            'sw': 'Swahili',
            'zu': 'Zulu',
            'af': 'Afrikaans',
            'sq': 'Albanian',
            'az': 'Azerbaijani',
            'be': 'Belarusian',
            'bg': 'Bulgarian',
            'ca': 'Catalan',
            'hr': 'Croatian',
            'cs': 'Czech',
            'et': 'Estonian',
            'eu': 'Basque',
            'gl': 'Galician',
            'hu': 'Hungarian',
            'is': 'Icelandic',
            'ga': 'Irish',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'mk': 'Macedonian',
            'mt': 'Maltese',
            'ro': 'Romanian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'sr': 'Serbian',
            'uk': 'Ukrainian',
            'cy': 'Welsh'
        }
    
    def detect_language(self, text: str) -> LanguageResult:
        """
        Detect the primary language of the given text.
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            LanguageResult containing detected language and confidence information
        """
        if not text or len(text.strip()) < self.min_text_length:
            return LanguageResult(
                language=self.fallback_language,
                confidence=0.0,
                is_reliable=False,
                detected_languages=[]
            )
        
        # Clean and prepare text for detection
        clean_text = self._prepare_text_for_detection(text)
        
        if len(clean_text) < self.min_text_length:
            return LanguageResult(
                language=self.fallback_language,
                confidence=0.0,
                is_reliable=False,
                detected_languages=[]
            )
        
        try:
            # Detect all possible languages with probabilities
            lang_probs = detect_langs(clean_text)
            detected_languages = [(lang.lang, lang.prob) for lang in lang_probs]
            
            if not detected_languages:
                return LanguageResult(
                    language=self.fallback_language,
                    confidence=0.0,
                    is_reliable=False,
                    detected_languages=[]
                )
            
            # Get the most likely language
            primary_lang = detected_languages[0][0]
            primary_confidence = detected_languages[0][1]
            
            # Determine if the detection is reliable
            is_reliable = (
                primary_confidence >= self.min_confidence and
                len(clean_text) >= self.min_text_length
            )
            
            return LanguageResult(
                language=primary_lang,
                confidence=primary_confidence,
                is_reliable=is_reliable,
                detected_languages=detected_languages
            )
            
        except LangDetectException as e:
            # Language detection failed
            return LanguageResult(
                language=self.fallback_language,
                confidence=0.0,
                is_reliable=False,
                detected_languages=[]
            )
    
    def filter_by_language(self, text: str, target_languages: List[str] = None) -> FilterResult:
        """
        Filter content based on language detection and confidence scores.
        
        Args:
            text: Text to analyze and potentially filter
            target_languages: List of allowed language codes (overrides config)
            
        Returns:
            FilterResult indicating whether content should be kept and why
        """
        # Use provided target languages or fall back to config
        allowed_langs = target_languages or self.allowed_languages
        
        # Detect language
        lang_result = self.detect_language(text)
        
        # If no language restrictions, only filter by confidence
        if not allowed_langs:
            if lang_result.is_reliable:
                return FilterResult(
                    should_keep=True,
                    reason=f"Language detected as {lang_result.language} with confidence {lang_result.confidence:.2f}",
                    language_result=lang_result
                )
            else:
                return FilterResult(
                    should_keep=False,
                    reason=f"Low confidence language detection: {lang_result.language} ({lang_result.confidence:.2f})",
                    language_result=lang_result
                )
        
        # Check if detected language is in allowed list
        if lang_result.language in allowed_langs:
            if lang_result.is_reliable:
                return FilterResult(
                    should_keep=True,
                    reason=f"Allowed language {lang_result.language} detected with confidence {lang_result.confidence:.2f}",
                    language_result=lang_result
                )
            else:
                return FilterResult(
                    should_keep=False,
                    reason=f"Allowed language {lang_result.language} but low confidence ({lang_result.confidence:.2f})",
                    language_result=lang_result
                )
        else:
            return FilterResult(
                should_keep=False,
                reason=f"Language {lang_result.language} not in allowed list: {allowed_langs}",
                language_result=lang_result
            )
    
    def normalize_multilingual_content(self, text: str) -> Dict[str, Any]:
        """
        Normalize multi-language content by detecting and segmenting different languages.
        
        Args:
            text: Text that may contain multiple languages
            
        Returns:
            Dictionary containing normalized content and language information
        """
        # Split text into paragraphs for individual analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {
                "normalized_content": text,
                "primary_language": self.fallback_language,
                "language_segments": [],
                "language_distribution": {},
                "is_multilingual": False
            }
        
        language_segments = []
        language_counts = {}
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) >= self.min_text_length:
                lang_result = self.detect_language(paragraph)
                
                language_segments.append({
                    "segment_index": i,
                    "text": paragraph,
                    "language": lang_result.language,
                    "confidence": lang_result.confidence,
                    "is_reliable": lang_result.is_reliable
                })
                
                # Count language occurrences
                if lang_result.is_reliable:
                    language_counts[lang_result.language] = language_counts.get(lang_result.language, 0) + 1
            else:
                # Short paragraph, assign to unknown
                language_segments.append({
                    "segment_index": i,
                    "text": paragraph,
                    "language": "unknown",
                    "confidence": 0.0,
                    "is_reliable": False
                })
        
        # Determine primary language
        if language_counts:
            primary_language = max(language_counts.keys(), key=lambda k: language_counts[k])
        else:
            primary_language = self.fallback_language
        
        # Calculate language distribution
        total_segments = len([seg for seg in language_segments if seg["is_reliable"]])
        language_distribution = {}
        if total_segments > 0:
            for lang, count in language_counts.items():
                language_distribution[lang] = count / total_segments
        
        # Determine if content is multilingual
        is_multilingual = len(language_counts) > 1
        
        # Create normalized content (for now, just return original)
        # In future versions, this could include translation or language-specific processing
        normalized_content = text
        
        return {
            "normalized_content": normalized_content,
            "primary_language": primary_language,
            "language_segments": language_segments,
            "language_distribution": language_distribution,
            "is_multilingual": is_multilingual,
            "detected_languages": list(language_counts.keys()),
            "total_segments": len(language_segments),
            "reliable_segments": total_segments
        }
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get human-readable language name from language code.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            Human-readable language name
        """
        return self.language_names.get(language_code, language_code.upper())
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of supported ISO 639-1 language codes
        """
        return list(self.language_names.keys())
    
    def _prepare_text_for_detection(self, text: str) -> str:
        """
        Prepare text for language detection by cleaning and truncating.
        
        Args:
            text: Raw text to prepare
            
        Returns:
            Cleaned text suitable for language detection
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)
        
        # Remove email addresses
        clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', clean_text)
        
        # Remove excessive punctuation
        clean_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', clean_text)
        
        # Remove numbers-only lines
        clean_text = re.sub(r'\b\d+\b', '', clean_text)
        
        # Collapse multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Truncate if too long
        if len(clean_text) > self.max_text_length:
            clean_text = clean_text[:self.max_text_length]
        
        return clean_text
    
    def batch_detect_languages(self, texts: List[str]) -> List[LanguageResult]:
        """
        Detect languages for multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of LanguageResult objects
        """
        results = []
        for text in texts:
            results.append(self.detect_language(text))
        return results
    
    def get_language_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get language distribution statistics for a collection of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with language statistics
        """
        results = self.batch_detect_languages(texts)
        
        language_counts = {}
        reliable_detections = 0
        total_texts = len(texts)
        
        for result in results:
            if result.is_reliable:
                reliable_detections += 1
                language_counts[result.language] = language_counts.get(result.language, 0) + 1
        
        # Calculate percentages
        language_percentages = {}
        if reliable_detections > 0:
            for lang, count in language_counts.items():
                language_percentages[lang] = (count / reliable_detections) * 100
        
        return {
            "total_texts": total_texts,
            "reliable_detections": reliable_detections,
            "reliability_rate": (reliable_detections / total_texts) * 100 if total_texts > 0 else 0,
            "language_counts": language_counts,
            "language_percentages": language_percentages,
            "detected_languages": list(language_counts.keys()),
            "most_common_language": max(language_counts.keys(), key=lambda k: language_counts[k]) if language_counts else None
        }
