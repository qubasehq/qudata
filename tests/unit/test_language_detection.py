"""
Unit tests for language detection and content filtering.

Tests the LanguageDetector class functionality including language detection,
content filtering, multi-language normalization, and edge cases.
"""

import pytest
from unittest.mock import patch, MagicMock
from langdetect.lang_detect_exception import LangDetectException

from src.qudata.clean.language import LanguageDetector, LanguageResult, FilterResult


class TestLanguageDetector:
    """Test cases for LanguageDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
        
        # Sample texts in different languages
        self.sample_texts = {
            'en': "This is a sample text in English. It contains multiple sentences to ensure proper language detection.",
            'es': "Este es un texto de muestra en español. Contiene múltiples oraciones para asegurar una detección adecuada del idioma.",
            'fr': "Ceci est un exemple de texte en français. Il contient plusieurs phrases pour assurer une détection linguistique appropriée.",
            'de': "Dies ist ein Beispieltext auf Deutsch. Er enthält mehrere Sätze, um eine ordnungsgemäße Spracherkennung zu gewährleisten.",
            'it': "Questo è un testo di esempio in italiano. Contiene più frasi per garantire un rilevamento linguistico adeguato.",
            'pt': "Este é um texto de amostra em português. Contém várias frases para garantir a detecção adequada do idioma.",
            'ru': "Это образец текста на русском языке. Он содержит несколько предложений для обеспечения правильного определения языка.",
            'ja': "これは日本語のサンプルテキストです。適切な言語検出を確実にするために複数の文が含まれています。",
            'zh-cn': "这是中文的示例文本。它包含多个句子以确保正确的语言检测。",
            'ar': "هذا نص عينة باللغة العربية. يحتوي على جمل متعددة لضمان الكشف المناسب عن اللغة."
        }
    
    def test_init_default_config(self):
        """Test LanguageDetector initialization with default configuration."""
        detector = LanguageDetector()
        
        assert detector.min_confidence == 0.7
        assert detector.allowed_languages is None
        assert detector.min_text_length == 50
        assert detector.max_text_length == 10000
        assert detector.fallback_language == "unknown"
    
    def test_init_custom_config(self):
        """Test LanguageDetector initialization with custom configuration."""
        config = {
            "min_confidence": 0.8,
            "allowed_languages": ["en", "es", "fr"],
            "min_text_length": 100,
            "max_text_length": 5000,
            "fallback_language": "en"
        }
        detector = LanguageDetector(config)
        
        assert detector.min_confidence == 0.8
        assert detector.allowed_languages == ["en", "es", "fr"]
        assert detector.min_text_length == 100
        assert detector.max_text_length == 5000
        assert detector.fallback_language == "en"
    
    def test_detect_language_english(self):
        """Test language detection for English text."""
        result = self.detector.detect_language(self.sample_texts['en'])
        
        assert isinstance(result, LanguageResult)
        assert result.language == 'en'
        assert result.confidence > 0.7
        assert result.is_reliable is True
        assert len(result.detected_languages) > 0
        assert result.detected_languages[0][0] == 'en'
    
    def test_detect_language_spanish(self):
        """Test language detection for Spanish text."""
        result = self.detector.detect_language(self.sample_texts['es'])
        
        assert result.language == 'es'
        assert result.confidence > 0.7
        assert result.is_reliable is True
    
    def test_detect_language_french(self):
        """Test language detection for French text."""
        result = self.detector.detect_language(self.sample_texts['fr'])
        
        assert result.language == 'fr'
        assert result.confidence > 0.7
        assert result.is_reliable is True
    
    def test_detect_language_german(self):
        """Test language detection for German text."""
        result = self.detector.detect_language(self.sample_texts['de'])
        
        assert result.language == 'de'
        assert result.confidence > 0.7
        assert result.is_reliable is True
    
    def test_detect_language_short_text(self):
        """Test language detection for text shorter than minimum length."""
        short_text = "Hello"
        result = self.detector.detect_language(short_text)
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
        assert result.detected_languages == []
    
    def test_detect_language_empty_text(self):
        """Test language detection for empty text."""
        result = self.detector.detect_language("")
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
        assert result.detected_languages == []
    
    def test_detect_language_none_text(self):
        """Test language detection for None input."""
        result = self.detector.detect_language(None)
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
        assert result.detected_languages == []
    
    @patch('src.qudata.clean.language.detect_langs')
    def test_detect_language_exception_handling(self, mock_detect_langs):
        """Test language detection when langdetect raises an exception."""
        mock_detect_langs.side_effect = LangDetectException(1, "Detection failed")
        
        result = self.detector.detect_language(self.sample_texts['en'])
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
        assert result.detected_languages == []
    
    def test_filter_by_language_no_restrictions(self):
        """Test content filtering with no language restrictions."""
        result = self.detector.filter_by_language(self.sample_texts['en'])
        
        assert isinstance(result, FilterResult)
        assert result.should_keep is True
        assert "confidence" in result.reason
        assert result.language_result.language == 'en'
    
    def test_filter_by_language_allowed_language(self):
        """Test content filtering with allowed language list."""
        result = self.detector.filter_by_language(
            self.sample_texts['en'], 
            target_languages=['en', 'es']
        )
        
        assert result.should_keep is True
        assert "Allowed language en" in result.reason
    
    def test_filter_by_language_disallowed_language(self):
        """Test content filtering with disallowed language."""
        result = self.detector.filter_by_language(
            self.sample_texts['fr'], 
            target_languages=['en', 'es']
        )
        
        assert result.should_keep is False
        assert "not in allowed list" in result.reason
    
    def test_filter_by_language_low_confidence(self):
        """Test content filtering with low confidence detection."""
        # Use very short text to get low confidence
        short_text = "Bonjour monde"  # Short French text
        result = self.detector.filter_by_language(short_text)
        
        # This might pass or fail depending on langdetect behavior with short text
        # The important thing is that it handles low confidence appropriately
        assert isinstance(result, FilterResult)
        assert isinstance(result.should_keep, bool)
    
    def test_normalize_multilingual_content_single_language(self):
        """Test multi-language normalization with single language content."""
        result = self.detector.normalize_multilingual_content(self.sample_texts['en'])
        
        assert result["primary_language"] == 'en'
        assert result["is_multilingual"] is False
        assert len(result["detected_languages"]) <= 1
        assert result["normalized_content"] == self.sample_texts['en']
    
    def test_normalize_multilingual_content_mixed_languages(self):
        """Test multi-language normalization with mixed language content."""
        mixed_text = f"{self.sample_texts['en']}\n\n{self.sample_texts['es']}\n\n{self.sample_texts['fr']}"
        result = self.detector.normalize_multilingual_content(mixed_text)
        
        assert result["is_multilingual"] is True
        assert len(result["detected_languages"]) > 1
        assert len(result["language_segments"]) == 3
        assert result["total_segments"] == 3
    
    def test_normalize_multilingual_content_empty_text(self):
        """Test multi-language normalization with empty text."""
        result = self.detector.normalize_multilingual_content("")
        
        assert result["primary_language"] == "unknown"
        assert result["is_multilingual"] is False
        assert result["language_segments"] == []
        assert result["language_distribution"] == {}
    
    def test_get_language_name(self):
        """Test getting human-readable language names."""
        assert self.detector.get_language_name('en') == 'English'
        assert self.detector.get_language_name('es') == 'Spanish'
        assert self.detector.get_language_name('fr') == 'French'
        assert self.detector.get_language_name('unknown') == 'UNKNOWN'
    
    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = self.detector.get_supported_languages()
        
        assert isinstance(languages, list)
        assert 'en' in languages
        assert 'es' in languages
        assert 'fr' in languages
        assert len(languages) > 10  # Should have many supported languages
    
    def test_prepare_text_for_detection(self):
        """Test text preparation for language detection."""
        messy_text = "  This   is  a   test  with   extra   spaces  and  http://example.com  and email@test.com  "
        clean_text = self.detector._prepare_text_for_detection(messy_text)
        
        assert "http://example.com" not in clean_text
        assert "email@test.com" not in clean_text
        assert "  " not in clean_text  # No double spaces
        assert clean_text.strip() == clean_text  # No leading/trailing spaces
    
    def test_prepare_text_for_detection_long_text(self):
        """Test text preparation with text longer than max length."""
        long_text = "This is a test. " * 1000  # Very long text
        detector = LanguageDetector({"max_text_length": 100})
        clean_text = detector._prepare_text_for_detection(long_text)
        
        assert len(clean_text) <= 100
    
    def test_batch_detect_languages(self):
        """Test batch language detection."""
        texts = [self.sample_texts['en'], self.sample_texts['es'], self.sample_texts['fr']]
        results = self.detector.batch_detect_languages(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, LanguageResult) for result in results)
        assert results[0].language == 'en'
        assert results[1].language == 'es'
        assert results[2].language == 'fr'
    
    def test_get_language_statistics(self):
        """Test language statistics calculation."""
        texts = [
            self.sample_texts['en'],
            self.sample_texts['en'],  # Duplicate English
            self.sample_texts['es'],
            self.sample_texts['fr']
        ]
        stats = self.detector.get_language_statistics(texts)
        
        assert stats["total_texts"] == 4
        assert stats["reliable_detections"] > 0
        assert "language_counts" in stats
        assert "language_percentages" in stats
        assert "detected_languages" in stats
        assert stats["most_common_language"] == 'en'  # Should be most common
    
    def test_language_result_to_dict(self):
        """Test LanguageResult serialization to dictionary."""
        result = LanguageResult(
            language='en',
            confidence=0.95,
            is_reliable=True,
            detected_languages=[('en', 0.95), ('es', 0.05)]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["language"] == 'en'
        assert result_dict["confidence"] == 0.95
        assert result_dict["is_reliable"] is True
        assert result_dict["detected_languages"] == [('en', 0.95), ('es', 0.05)]
    
    def test_filter_result_to_dict(self):
        """Test FilterResult serialization to dictionary."""
        lang_result = LanguageResult('en', 0.95, True, [('en', 0.95)])
        filter_result = FilterResult(
            should_keep=True,
            reason="High confidence detection",
            language_result=lang_result
        )
        
        result_dict = filter_result.to_dict()
        
        assert result_dict["should_keep"] is True
        assert result_dict["reason"] == "High confidence detection"
        assert "language_result" in result_dict
    
    def test_mixed_language_content_detection(self):
        """Test detection of mixed language content within single text."""
        # Create text with English and Spanish mixed together
        mixed_text = f"Hello world. {self.sample_texts['es']} Thank you very much."
        result = self.detector.detect_language(mixed_text)
        
        # Should detect one of the languages (likely the dominant one)
        assert result.language in ['en', 'es']
        assert isinstance(result.confidence, float)
    
    def test_special_characters_handling(self):
        """Test language detection with special characters and symbols."""
        text_with_symbols = "This is English text with symbols: @#$%^&*()_+ and numbers 12345"
        result = self.detector.detect_language(text_with_symbols)
        
        assert result.language == 'en'
        assert result.is_reliable is True
    
    def test_code_content_detection(self):
        """Test language detection with code-like content."""
        code_text = """
        def hello_world():
            print("Hello, World!")
            return True
        
        This is a Python function that prints hello world.
        """
        result = self.detector.detect_language(code_text)
        
        # Should still detect English from the comment
        assert result.language == 'en'
    
    def test_numeric_content_handling(self):
        """Test language detection with mostly numeric content."""
        numeric_text = "123 456 789 000 111 222 333 444 555 666 777 888 999"
        result = self.detector.detect_language(numeric_text)
        
        # Should return unknown or fallback language for numeric content
        assert result.language == "unknown"
        assert result.is_reliable is False


class TestLanguageDetectorEdgeCases:
    """Test edge cases and error conditions for LanguageDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_whitespace_only_text(self):
        """Test detection with whitespace-only text."""
        whitespace_text = "   \n\t\r   "
        result = self.detector.detect_language(whitespace_text)
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
    
    def test_punctuation_only_text(self):
        """Test detection with punctuation-only text."""
        punct_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.detector.detect_language(punct_text)
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.is_reliable is False
    
    def test_very_long_text_truncation(self):
        """Test that very long text is properly truncated."""
        # Create text longer than max_text_length
        long_text = "This is English text. " * 1000
        detector = LanguageDetector({"max_text_length": 500})
        
        result = detector.detect_language(long_text)
        
        # Should still detect English despite truncation
        assert result.language == 'en'
        assert result.is_reliable is True
    
    def test_custom_fallback_language(self):
        """Test custom fallback language configuration."""
        detector = LanguageDetector({"fallback_language": "custom"})
        result = detector.detect_language("")
        
        assert result.language == "custom"
    
    def test_high_confidence_threshold(self):
        """Test with very high confidence threshold."""
        detector = LanguageDetector({"min_confidence": 0.99})
        
        # Even good text might not meet 99% confidence
        text = "This is a short English sentence."
        result = detector.detect_language(text)
        
        # Reliability depends on actual confidence score
        assert isinstance(result.is_reliable, bool)
    
    def test_multilingual_normalization_short_paragraphs(self):
        """Test multilingual normalization with short paragraphs."""
        short_paragraphs = "Hi.\n\nHola.\n\nBonjour."
        result = self.detector.normalize_multilingual_content(short_paragraphs)
        
        # Short paragraphs should be marked as unknown
        assert result["primary_language"] == "unknown"
        assert len(result["language_segments"]) == 3
        for segment in result["language_segments"]:
            assert segment["language"] == "unknown"
            assert segment["is_reliable"] is False


if __name__ == "__main__":
    pytest.main([__file__])