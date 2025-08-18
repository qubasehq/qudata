"""
Text normalization and cleaning module.

This module provides text normalization capabilities including Unicode normalization,
whitespace cleanup, OCR error correction, and encoding detection/conversion.
"""

import re
import unicodedata
import chardet
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..models import ProcessingError, ErrorSeverity


@dataclass
class NormalizationResult:
    """Result of text normalization operations."""
    normalized_text: str
    original_encoding: Optional[str] = None
    detected_language: Optional[str] = None
    corrections_applied: List[str] = None
    quality_score: float = 1.0
    
    def __post_init__(self):
        if self.corrections_applied is None:
            self.corrections_applied = []


class TextNormalizer:
    """
    Text normalizer for Unicode normalization and whitespace cleanup.
    
    Handles Unicode normalization, whitespace cleanup, quote normalization,
    and basic text standardization operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TextNormalizer.
        
        Args:
            config: Configuration dictionary with normalization settings
        """
        self.config = config or {}
        
        # Normalization settings
        self.unicode_form = self.config.get('unicode_form', 'NFKC')
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.normalize_quotes = self.config.get('normalize_quotes', True)
        self.normalize_punctuation = self.config.get('normalize_punctuation', False)
        self.remove_control_chars = self.config.get('remove_control_chars', True)
        self.preserve_line_breaks = self.config.get('preserve_line_breaks', True)
        
        # Quote normalization mappings
        self.quote_mappings = {
            # Smart quotes to straight quotes
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201e': '"',  # Double low-9 quotation mark
            '\u201a': "'",  # Single low-9 quotation mark
            '\u00ab': '"',  # Left-pointing double angle quotation mark
            '\u00bb': '"',  # Right-pointing double angle quotation mark
            '\u2039': "'",  # Single left-pointing angle quotation mark
            '\u203a': "'",  # Single right-pointing angle quotation mark
            '\u300c': '"',  # Left corner bracket
            '\u300d': '"',  # Right corner bracket
            '\u300e': '"',  # Left white corner bracket
            '\u300f': '"',  # Right white corner bracket
        }
        
        # Punctuation normalization mappings
        self.punctuation_mappings = {
            '\u2026': '...',  # Horizontal ellipsis
            '\u2013': '-',    # En dash
            '\u2014': '-',    # Em dash
            '\u2212': '-',    # Minus sign
            '\u2022': '*',    # Bullet
            '\u00b7': '*',    # Middle dot
            '\u2027': '*',    # Hyphenation point
            '\u2219': '*',    # Bullet operator
            '\u2032': "'",    # Prime
            '\u2033': "''",   # Double prime (two single quotes)
            '\u2034': "'''",  # Triple prime
        }
        
        # Control characters to remove (except common whitespace)
        self.control_chars_pattern = re.compile(
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
        )
        
        # Multiple whitespace patterns
        self.whitespace_patterns = [
            (re.compile(r'\r\n'), '\n'),  # Windows line endings
            (re.compile(r'\r'), '\n'),    # Mac line endings
            (re.compile(r'\n{3,}'), '\n\n'),  # Multiple newlines
            (re.compile(r'[ \t]+'), ' '),  # Multiple spaces/tabs
            (re.compile(r'[ \t]*\n[ \t]*'), '\n'),  # Whitespace around newlines
        ]
    
    def normalize_text(self, text: str) -> NormalizationResult:
        """
        Normalize text with all configured normalizations.
        
        Args:
            text: Input text to normalize
            
        Returns:
            NormalizationResult with normalized text and metadata
        """
        if not text or not isinstance(text, str):
            return NormalizationResult(normalized_text="")
        
        result = NormalizationResult(normalized_text=text)
        original_text = text
        
        try:
            # Unicode normalization
            if self.unicode_form:
                text = self._normalize_unicode(text)
                if text != original_text:
                    result.corrections_applied.append("unicode_normalization")
            
            # Remove control characters
            if self.remove_control_chars:
                text_before = text
                text = self._remove_control_characters(text)
                if text != text_before:
                    result.corrections_applied.append("control_char_removal")
            
            # Quote normalization
            if self.normalize_quotes:
                text_before = text
                text = self._normalize_quotes(text)
                if text != text_before:
                    result.corrections_applied.append("quote_normalization")
            
            # Punctuation normalization
            if self.normalize_punctuation:
                text_before = text
                text = self._normalize_punctuation(text)
                if text != text_before:
                    result.corrections_applied.append("punctuation_normalization")
            
            # Whitespace normalization
            if self.normalize_whitespace:
                text_before = text
                text = self._normalize_whitespace(text)
                if text != text_before:
                    result.corrections_applied.append("whitespace_normalization")
            
            result.normalized_text = text
            result.quality_score = self._calculate_quality_score(original_text, text)
            
        except Exception as e:
            raise ProcessingError(
                stage="normalization",
                error_type="NormalizationError",
                message=f"Failed to normalize text: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return result
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters using specified form."""
        try:
            return unicodedata.normalize(self.unicode_form, text)
        except Exception:
            # Fallback to NFKC if specified form fails
            return unicodedata.normalize('NFKC', text)
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters while preserving important whitespace."""
        if self.preserve_line_breaks:
            # Preserve newlines and tabs
            return self.control_chars_pattern.sub('', text)
        else:
            # Remove all control characters
            return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote characters to standard ASCII quotes."""
        for smart_quote, standard_quote in self.quote_mappings.items():
            text = text.replace(smart_quote, standard_quote)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize various punctuation characters to standard ASCII."""
        for special_punct, standard_punct in self.punctuation_mappings.items():
            text = text.replace(special_punct, standard_punct)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters and patterns."""
        for pattern, replacement in self.whitespace_patterns:
            text = pattern.sub(replacement, text)
        
        # Final cleanup - strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _calculate_quality_score(self, original: str, normalized: str) -> float:
        """Calculate quality score based on normalization changes."""
        if not original:
            return 1.0
        
        # Calculate ratio of changes
        changes = len(original) - len(normalized)
        change_ratio = abs(changes) / len(original)
        
        # Score decreases with more changes, but caps at reasonable levels
        quality_score = max(0.5, 1.0 - (change_ratio * 0.5))
        
        return round(quality_score, 3)


class OCRCorrector:
    """
    OCR error correction for common OCR mistakes and patterns.
    
    Handles common OCR errors like character substitutions, spacing issues,
    and formatting problems that occur during optical character recognition.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OCRCorrector.
        
        Args:
            config: Configuration dictionary with OCR correction settings
        """
        self.config = config or {}
        
        # Enable/disable correction types
        self.correct_character_substitutions = self.config.get('correct_character_substitutions', True)
        self.correct_spacing_issues = self.config.get('correct_spacing_issues', True)
        self.correct_line_breaks = self.config.get('correct_line_breaks', True)
        self.correct_word_splitting = self.config.get('correct_word_splitting', True)
        
        # Common OCR character substitution patterns
        self.character_corrections = {
            # Number/letter confusion
            '0': ['O', 'o'],  # Zero to letter O
            'O': ['0'],       # Letter O to zero (context dependent)
            '1': ['l', 'I', '|'],  # One to lowercase l, uppercase I, pipe
            'l': ['1', 'I'],  # Lowercase l to one, uppercase I
            'I': ['1', 'l'],  # Uppercase I to one, lowercase l
            '5': ['S', 's'],  # Five to letter S
            'S': ['5'],       # Letter S to five (context dependent)
            '6': ['G', 'b'],  # Six to letter G or b
            '8': ['B'],       # Eight to letter B
            'B': ['8'],       # Letter B to eight (context dependent)
            
            # Common character confusions
            'rn': ['m'],      # rn combination to m
            'm': ['rn'],      # m to rn (context dependent)
            'cl': ['d'],      # cl combination to d
            'ii': ['n'],      # ii combination to n
            'vv': ['w'],      # vv combination to w
            'w': ['vv'],      # w to vv (context dependent)
            
            # Punctuation issues
            ',': ['.', ';'],  # Comma confusion
            '.': [','],       # Period confusion
            ':': [';'],       # Colon confusion
            ';': [':'],       # Semicolon confusion
        }
        
        # Word boundary patterns for context-aware corrections
        self.word_boundary_patterns = [
            # Fix spacing around punctuation
            (re.compile(r'\s+([,.;:!?])'), r'\1'),  # Remove space before punctuation
            (re.compile(r'([,.;:!?])([A-Za-z])'), r'\1 \2'),  # Add space after punctuation
            
            # Fix hyphenated words split across lines
            (re.compile(r'-\s*\n\s*'), ''),  # Remove hyphen and line break
            
            # Fix words split by spaces
            (re.compile(r'\b([A-Z][a-z]+)\s+([a-z]+)\b'), self._check_word_split),
        ]
        
        # Common OCR word corrections (most frequent mistakes)
        self.word_corrections = {
            'tlie': 'the', 'tiie': 'the', 'tne': 'the', 'tha': 'the',
            'arid': 'and', 'ancl': 'and', 'anci': 'and',
            'witli': 'with', 'widi': 'with', 'witii': 'with',
            'tliat': 'that', 'tiiat': 'that', 'tnat': 'that',
            'tliis': 'this', 'tiiis': 'this', 'tnis': 'this',
            'wliich': 'which', 'wiiich': 'which', 'wnich': 'which',
            'wlien': 'when', 'wiien': 'when', 'wnen': 'when',
            'wliere': 'where', 'wiiere': 'where', 'wnere': 'where',
            'tliey': 'they', 'tiiey': 'they', 'tney': 'they',
            'tliere': 'there', 'tiiere': 'there', 'tnere': 'there',
            'tlirough': 'through', 'tiirough': 'through', 'tnrough': 'through',
        }
    
    def correct_ocr_errors(self, text: str) -> NormalizationResult:
        """
        Correct common OCR errors in text.
        
        Args:
            text: Input text with potential OCR errors
            
        Returns:
            NormalizationResult with corrected text and applied corrections
        """
        if not text or not isinstance(text, str):
            return NormalizationResult(normalized_text="")
        
        result = NormalizationResult(normalized_text=text)
        original_text = text
        
        try:
            # Word-level corrections
            if self.correct_character_substitutions:
                text_before = text
                text = self._correct_word_substitutions(text)
                if text != text_before:
                    result.corrections_applied.append("word_substitutions")
            
            # Spacing corrections
            if self.correct_spacing_issues:
                text_before = text
                text = self._correct_spacing_issues(text)
                if text != text_before:
                    result.corrections_applied.append("spacing_corrections")
            
            # Line break corrections
            if self.correct_line_breaks:
                text_before = text
                text = self._correct_line_breaks(text)
                if text != text_before:
                    result.corrections_applied.append("line_break_corrections")
            
            # Word splitting corrections
            if self.correct_word_splitting:
                text_before = text
                text = self._correct_word_splitting(text)
                if text != text_before:
                    result.corrections_applied.append("word_splitting_corrections")
            
            result.normalized_text = text
            result.quality_score = self._calculate_correction_quality(original_text, text)
            
        except Exception as e:
            raise ProcessingError(
                stage="ocr_correction",
                error_type="OCRCorrectionError",
                message=f"Failed to correct OCR errors: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return result
    
    def _correct_word_substitutions(self, text: str) -> str:
        """Correct common OCR word substitution errors."""
        # Split on whitespace but preserve the original structure
        words = re.split(r'(\s+)', text)
        corrected_words = []
        
        for word in words:
            # Skip whitespace tokens
            if word.isspace():
                corrected_words.append(word)
                continue
                
            # Remove punctuation for matching, but preserve it
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.word_corrections:
                # Replace the clean part while preserving punctuation
                corrected = self.word_corrections[clean_word]
                # Preserve original case pattern
                if word and word[0].isupper():
                    corrected = corrected.capitalize()
                # Restore punctuation
                punctuation = re.findall(r'[^\w]', word)
                if punctuation:
                    corrected += ''.join(punctuation)
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ''.join(corrected_words)
    
    def _correct_spacing_issues(self, text: str) -> str:
        """Correct spacing issues around punctuation and words."""
        for pattern, replacement in self.word_boundary_patterns:
            if callable(replacement):
                # Handle complex replacements
                text = pattern.sub(replacement, text)
            else:
                text = pattern.sub(replacement, text)
        
        return text
    
    def _correct_line_breaks(self, text: str) -> str:
        """Correct line break issues common in OCR."""
        # Fix words broken across lines without hyphens (more conservative)
        # Only join if the line ends with lowercase and next starts with lowercase
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)
        
        # Fix sentences broken across lines
        text = re.sub(r'([a-z,])\n([A-Z])', r'\1 \2', text)
        
        # Convert remaining single line breaks to spaces (but preserve paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        return text
    
    def _correct_word_splitting(self, text: str) -> str:
        """Correct words that have been incorrectly split."""
        # This is a simplified approach - in practice, you'd want a dictionary
        # to check if rejoined words are valid
        
        # Fix common patterns like "ex ample" -> "example"
        # Look for short word fragments that might be split
        def rejoin_callback(match):
            return self._try_rejoin_words(match.group(1), match.group(2))
        
        text = re.sub(r'\b([a-z]{2,3})\s+([a-z]{3,})\b', rejoin_callback, text)
        
        return text
    
    def _try_rejoin_words(self, word1: str, word2: str) -> str:
        """Try to rejoin two words if they form a common word."""
        # Simple heuristic - rejoin if the combined word looks reasonable
        combined = word1 + word2
        
        # Don't rejoin if combined word is too long
        if len(combined) > 15:
            return f"{word1} {word2}"
        
        # Check for common split patterns
        vowels = set('aeiouAEIOU')
        
        # If first part has no vowels and second part does, likely split
        if not any(c in vowels for c in word1) and any(c in vowels for c in word2):
            return combined
        
        # If it's a common prefix pattern (ex + ample = example)
        common_prefixes = {'ex', 'pre', 'pro', 'con', 'dis', 'mis', 'un', 're'}
        if word1.lower() in common_prefixes:
            return combined
        
        # Default to keeping separate
        return f"{word1} {word2}"
    
    def _check_word_split(self, match) -> str:
        """Check if a word split should be rejoined."""
        word1, word2 = match.groups()
        return self._try_rejoin_words(word1, word2)
    
    def _calculate_correction_quality(self, original: str, corrected: str) -> float:
        """Calculate quality score based on OCR corrections."""
        if not original:
            return 1.0
        
        # Count the number of corrections made
        corrections = len([c for c in [original, corrected] if c != original])
        
        # More corrections might indicate lower original quality
        # but successful correction improves the score
        if len(corrected) == 0:
            return 0.0
        
        # Simple heuristic based on length changes and correction count
        length_ratio = len(corrected) / len(original)
        quality_score = min(1.0, length_ratio * 0.9 + 0.1)
        
        return round(quality_score, 3)


class EncodingDetector:
    """
    Encoding detection and UTF-8 conversion utility.
    
    Detects text encoding and converts to UTF-8 for consistent processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize EncodingDetector.
        
        Args:
            config: Configuration dictionary with encoding settings
        """
        self.config = config or {}
        
        # Confidence threshold for encoding detection
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Fallback encodings to try if detection fails
        self.fallback_encodings = self.config.get('fallback_encodings', [
            'utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii'
        ])
        
        # Whether to be strict about encoding errors
        self.strict_encoding = self.config.get('strict_encoding', False)
    
    def detect_and_convert(self, data: bytes) -> NormalizationResult:
        """
        Detect encoding of byte data and convert to UTF-8 string.
        
        Args:
            data: Raw byte data to detect encoding for
            
        Returns:
            NormalizationResult with UTF-8 text and detected encoding info
        """
        if not data:
            return NormalizationResult(normalized_text="")
        
        result = NormalizationResult(normalized_text="")
        
        try:
            # Try to detect encoding
            detection = chardet.detect(data)
            detected_encoding = detection.get('encoding')
            confidence = detection.get('confidence', 0.0)
            
            result.original_encoding = detected_encoding
            
            # If confidence is too low, try fallback encodings
            if not detected_encoding or confidence < self.confidence_threshold:
                text, encoding = self._try_fallback_encodings(data)
                result.original_encoding = encoding
                result.corrections_applied.append("fallback_encoding_used")
            else:
                # Try the detected encoding
                try:
                    text = data.decode(detected_encoding)
                    result.corrections_applied.append("encoding_detected")
                except (UnicodeDecodeError, LookupError):
                    # Fall back if detected encoding fails
                    text, encoding = self._try_fallback_encodings(data)
                    result.original_encoding = encoding
                    result.corrections_applied.append("fallback_encoding_used")
            
            # Ensure the result is valid UTF-8
            if isinstance(text, str):
                # Re-encode to UTF-8 to ensure consistency
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                result.normalized_text = text
                result.quality_score = confidence if confidence > 0 else 0.8
            else:
                raise ProcessingError(
                    stage="encoding_detection",
                    error_type="EncodingError",
                    message="Failed to decode text to string",
                    severity=ErrorSeverity.HIGH
                )
                
        except Exception as e:
            if self.strict_encoding:
                raise ProcessingError(
                    stage="encoding_detection",
                    error_type="EncodingError",
                    message=f"Failed to detect or convert encoding: {str(e)}",
                    severity=ErrorSeverity.HIGH
                )
            else:
                # Last resort - try to decode as UTF-8 with error replacement
                try:
                    text = data.decode('utf-8', errors='replace')
                    result.normalized_text = text
                    result.original_encoding = 'utf-8'
                    result.corrections_applied.append("utf8_with_replacement")
                    result.quality_score = 0.5  # Lower quality due to replacements
                except Exception:
                    # Final fallback - convert to string representation
                    result.normalized_text = str(data, errors='replace')
                    result.original_encoding = 'unknown'
                    result.corrections_applied.append("string_conversion_fallback")
                    result.quality_score = 0.3
        
        return result
    
    def _try_fallback_encodings(self, data: bytes) -> Tuple[str, str]:
        """
        Try fallback encodings when detection fails.
        
        Args:
            data: Raw byte data to decode
            
        Returns:
            Tuple of (decoded_text, encoding_used)
        """
        for encoding in self.fallback_encodings:
            try:
                text = data.decode(encoding)
                return text, encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all encodings fail, use UTF-8 with replacement
        text = data.decode('utf-8', errors='replace')
        return text, 'utf-8-with-replacement'
    
    def convert_to_utf8(self, text: str, source_encoding: str = None) -> NormalizationResult:
        """
        Convert text from source encoding to UTF-8.
        
        Args:
            text: Input text string or bytes
            source_encoding: Known source encoding (optional)
            
        Returns:
            NormalizationResult with UTF-8 text
        """
        if not text:
            return NormalizationResult(normalized_text="")
        
        result = NormalizationResult(normalized_text="")
        
        try:
            if source_encoding and isinstance(text, str):
                # Text is already decoded, just ensure it's valid UTF-8
                utf8_text = text.encode('utf-8', errors='replace').decode('utf-8')
                result.normalized_text = utf8_text
                result.original_encoding = source_encoding
                if utf8_text != text:
                    result.corrections_applied.append("utf8_cleanup")
                else:
                    result.corrections_applied.append("encoding_conversion")
            elif isinstance(text, bytes):
                # Convert bytes to string
                if source_encoding:
                    try:
                        utf8_text = text.decode(source_encoding).encode('utf-8', errors='replace').decode('utf-8')
                        result.normalized_text = utf8_text
                        result.original_encoding = source_encoding
                        result.corrections_applied.append("encoding_conversion")
                    except (UnicodeDecodeError, LookupError):
                        # Fall back to detection
                        detection_result = self.detect_and_convert(text)
                        return detection_result
                else:
                    # Use detection
                    detection_result = self.detect_and_convert(text)
                    return detection_result
            else:
                # Text is already a string, just ensure UTF-8 compliance
                utf8_text = text.encode('utf-8', errors='replace').decode('utf-8')
                result.normalized_text = utf8_text
                if utf8_text != text:
                    result.corrections_applied.append("utf8_cleanup")
            
            result.quality_score = 1.0 if not result.corrections_applied else 0.9
            
        except Exception as e:
            raise ProcessingError(
                stage="encoding_conversion",
                error_type="EncodingConversionError",
                message=f"Failed to convert to UTF-8: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return result


def normalize_text_pipeline(text: str, config: Dict[str, Any] = None) -> NormalizationResult:
    """
    Complete text normalization pipeline combining all normalization steps.
    
    Args:
        text: Input text to normalize
        config: Configuration for normalization steps
        
    Returns:
        NormalizationResult with fully normalized text
    """
    if not text:
        return NormalizationResult(normalized_text="")
    
    config = config or {}
    
    # Initialize components
    normalizer = TextNormalizer(config.get('normalizer', {}))
    ocr_corrector = OCRCorrector(config.get('ocr_corrector', {}))
    
    # Apply normalization steps
    result = normalizer.normalize_text(text)
    
    # Apply OCR corrections if enabled
    if config.get('apply_ocr_correction', True):
        ocr_result = ocr_corrector.correct_ocr_errors(result.normalized_text)
        result.normalized_text = ocr_result.normalized_text
        result.corrections_applied.extend(ocr_result.corrections_applied)
        # Average the quality scores
        result.quality_score = (result.quality_score + ocr_result.quality_score) / 2
    
    return result
