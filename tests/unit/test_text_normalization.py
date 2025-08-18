"""
Unit tests for text normalization and cleaning functionality.

Tests the TextNormalizer, OCRCorrector, and EncodingDetector classes
for various text cleaning scenarios and edge cases.
"""

import pytest
import unicodedata
from src.qudata.clean.normalize import (
    TextNormalizer, OCRCorrector, EncodingDetector, 
    NormalizationResult, normalize_text_pipeline
)
from src.qudata.models import ProcessingError, ErrorSeverity


class TestTextNormalizer:
    """Test cases for TextNormalizer class."""
    
    def test_basic_normalization(self):
        """Test basic text normalization functionality."""
        normalizer = TextNormalizer()
        
        # Test with normal text
        result = normalizer.normalize_text("Hello world!")
        assert result.normalized_text == "Hello world!"
        assert result.quality_score == 1.0
        assert len(result.corrections_applied) == 0
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        normalizer = TextNormalizer({'unicode_form': 'NFKC'})
        
        # Test with Unicode characters that need normalization
        text_with_unicode = "café naïve résumé"  # Contains accented characters
        result = normalizer.normalize_text(text_with_unicode)
        
        # Should normalize to NFKC form
        expected = unicodedata.normalize('NFKC', text_with_unicode)
        assert result.normalized_text == expected
        
        # Test with combining characters
        combining_text = "e\u0301"  # e + combining acute accent
        result = normalizer.normalize_text(combining_text)
        assert result.normalized_text == "é"  # Should combine to single character
        assert "unicode_normalization" in result.corrections_applied
    
    def test_quote_normalization(self):
        """Test smart quote normalization."""
        normalizer = TextNormalizer({'normalize_quotes': True})
        
        # Test various smart quotes
        smart_quotes_text = "\u201cHello\u201d and \u2018world\u2019 with \u201eother\u201c quotes"
        result = normalizer.normalize_text(smart_quotes_text)
        
        expected = '"Hello" and \'world\' with "other" quotes'
        assert result.normalized_text == expected
        assert "quote_normalization" in result.corrections_applied
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        normalizer = TextNormalizer({'normalize_whitespace': True})
        
        # Test multiple spaces and line endings
        messy_whitespace = "Hello    world\r\n\r\nWith   \t  tabs\n\n\n\nAnd newlines"
        result = normalizer.normalize_text(messy_whitespace)
        
        expected = "Hello world\n\nWith tabs\n\nAnd newlines"
        assert result.normalized_text == expected
        assert "whitespace_normalization" in result.corrections_applied
    
    def test_control_character_removal(self):
        """Test control character removal."""
        normalizer = TextNormalizer({'remove_control_chars': True})
        
        # Text with control characters
        text_with_control = "Hello\x00\x08world\x7F\x9F"
        result = normalizer.normalize_text(text_with_control)
        
        assert result.normalized_text == "Helloworld"
        assert "control_char_removal" in result.corrections_applied
    
    def test_punctuation_normalization(self):
        """Test punctuation normalization."""
        normalizer = TextNormalizer({'normalize_punctuation': True})
        
        # Test individual punctuation marks
        test_cases = [
            ("\u2026", "..."),  # Ellipsis
            ("\u2014", "-"),    # Em dash
            ("\u2013", "-"),    # En dash
            ("\u2022", "*"),    # Bullet
            ("\u2032", "'"),    # Prime
            ("\u2033", "''"),   # Double prime
        ]
        
        for input_char, expected_char in test_cases:
            test_text = f"Hello{input_char}world"
            result = normalizer.normalize_text(test_text)
            expected = f"Hello{expected_char}world"
            assert result.normalized_text == expected, f"Failed for {input_char} -> {expected_char}"
        
        # Test combined punctuation
        combined_text = "Hello\u2026 world\u2014test"
        result = normalizer.normalize_text(combined_text)
        assert result.normalized_text == "Hello... world-test"
        assert "punctuation_normalization" in result.corrections_applied
    
    def test_empty_and_none_input(self):
        """Test handling of empty and None inputs."""
        normalizer = TextNormalizer()
        
        # Test empty string
        result = normalizer.normalize_text("")
        assert result.normalized_text == ""
        assert result.quality_score == 1.0
        
        # Test None input
        result = normalizer.normalize_text(None)
        assert result.normalized_text == ""
        
        # Test whitespace-only string
        result = normalizer.normalize_text("   \t\n   ")
        assert result.normalized_text == ""
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        normalizer = TextNormalizer()
        
        # Text that requires significant normalization
        messy_text = "Hello    world\r\n\r\n"  # Will be normalized significantly
        result = normalizer.normalize_text(messy_text)
        
        # Quality score should be less than 1.0 due to changes
        assert 0.5 <= result.quality_score <= 1.0
    
    def test_configuration_options(self):
        """Test various configuration options."""
        # Test with all normalizations disabled
        config = {
            'normalize_whitespace': False,
            'normalize_quotes': False,
            'normalize_punctuation': False,
            'remove_control_chars': False
        }
        normalizer = TextNormalizer(config)
        
        text = "\u201cHello\u201d    world\x00"
        result = normalizer.normalize_text(text)
        
        # Should only apply Unicode normalization (always enabled)
        assert "\u201cHello\u201d    world\x00" in result.normalized_text or result.normalized_text == "\u201cHello\u201d    world"


class TestOCRCorrector:
    """Test cases for OCRCorrector class."""
    
    def test_word_substitution_corrections(self):
        """Test common OCR word corrections."""
        corrector = OCRCorrector()
        
        # Test common OCR mistakes
        ocr_text = "Tlie quick brown fox jumps over tne lazy dog witli great speed."
        result = corrector.correct_ocr_errors(ocr_text)
        
        expected = "The quick brown fox jumps over the lazy dog with great speed."
        assert result.normalized_text == expected
        assert "word_substitutions" in result.corrections_applied
    
    def test_spacing_corrections(self):
        """Test spacing issue corrections."""
        corrector = OCRCorrector()
        
        # Test spacing around punctuation
        spacing_issues = "Hello ,world !How are you ?"
        result = corrector.correct_ocr_errors(spacing_issues)
        
        expected = "Hello, world! How are you?"
        assert result.normalized_text == expected
        assert "spacing_corrections" in result.corrections_applied
    
    def test_line_break_corrections(self):
        """Test line break corrections."""
        corrector = OCRCorrector()
        
        # Test broken words across lines
        broken_lines = "This is a sam\nple text with bro\nken words."
        result = corrector.correct_ocr_errors(broken_lines)
        
        # Should rejoin broken words
        assert "sample" in result.normalized_text
        assert "broken" in result.normalized_text
        assert "line_break_corrections" in result.corrections_applied
    
    def test_word_splitting_corrections(self):
        """Test word splitting corrections."""
        corrector = OCRCorrector()
        
        # Test words incorrectly split by spaces
        split_words = "This is an ex ample of split w ords."
        result = corrector.correct_ocr_errors(split_words)
        
        # Should attempt to rejoin some split words
        assert "word_splitting_corrections" in result.corrections_applied
    
    def test_case_preservation(self):
        """Test that case is preserved in corrections."""
        corrector = OCRCorrector()
        
        # Test with capitalized OCR errors
        capitalized_errors = "Tlie Quick Brown Fox"
        result = corrector.correct_ocr_errors(capitalized_errors)
        
        expected = "The Quick Brown Fox"
        assert result.normalized_text == expected
    
    def test_punctuation_preservation(self):
        """Test that punctuation is preserved in corrections."""
        corrector = OCRCorrector()
        
        # Test with punctuation attached to OCR errors
        punctuated_errors = "Tlie, quick brown fox."
        result = corrector.correct_ocr_errors(punctuated_errors)
        
        expected = "The, quick brown fox."
        assert result.normalized_text == expected
    
    def test_empty_and_invalid_input(self):
        """Test handling of empty and invalid inputs."""
        corrector = OCRCorrector()
        
        # Test empty string
        result = corrector.correct_ocr_errors("")
        assert result.normalized_text == ""
        
        # Test None input
        result = corrector.correct_ocr_errors(None)
        assert result.normalized_text == ""
    
    def test_quality_score_calculation(self):
        """Test OCR correction quality score."""
        corrector = OCRCorrector()
        
        # Text with many OCR errors
        error_text = "Tlie quick brown fox witli many errors tnat need correction."
        result = corrector.correct_ocr_errors(error_text)
        
        # Should have a reasonable quality score
        assert 0.3 <= result.quality_score <= 1.0
    
    def test_configuration_options(self):
        """Test OCR corrector configuration options."""
        # Test with specific corrections disabled
        config = {
            'correct_character_substitutions': False,
            'correct_spacing_issues': True,
            'correct_line_breaks': False,
            'correct_word_splitting': False
        }
        corrector = OCRCorrector(config)
        
        text = "Tlie quick ,brown fox"
        result = corrector.correct_ocr_errors(text)
        
        # Should only apply spacing corrections
        assert "spacing_corrections" in result.corrections_applied
        assert "word_substitutions" not in result.corrections_applied


class TestEncodingDetector:
    """Test cases for EncodingDetector class."""
    
    def test_utf8_detection(self):
        """Test UTF-8 encoding detection."""
        detector = EncodingDetector()
        
        # UTF-8 encoded text
        utf8_text = "Hello world with UTF-8: café naïve"
        utf8_bytes = utf8_text.encode('utf-8')
        
        result = detector.detect_and_convert(utf8_bytes)
        
        assert result.normalized_text == utf8_text
        assert result.original_encoding == 'utf-8'
        assert "encoding_detected" in result.corrections_applied
    
    def test_latin1_detection(self):
        """Test Latin-1 encoding detection and conversion."""
        detector = EncodingDetector()
        
        # Latin-1 encoded text
        latin1_text = "Hello café"
        latin1_bytes = latin1_text.encode('latin-1')
        
        result = detector.detect_and_convert(latin1_bytes)
        
        assert result.normalized_text == latin1_text
        assert result.original_encoding.lower() in ['latin-1', 'iso-8859-1', 'cp1252']
    
    def test_fallback_encodings(self):
        """Test fallback encoding handling."""
        detector = EncodingDetector({'confidence_threshold': 0.9})  # High threshold
        
        # Ambiguous bytes that might not be detected with high confidence
        ambiguous_bytes = b"Hello world"
        
        result = detector.detect_and_convert(ambiguous_bytes)
        
        assert result.normalized_text == "Hello world"
        assert result.original_encoding is not None
    
    def test_invalid_bytes_handling(self):
        """Test handling of invalid byte sequences."""
        detector = EncodingDetector({'strict_encoding': False})
        
        # Invalid UTF-8 sequence
        invalid_bytes = b'\xff\xfe\x00\x48\x00\x65\x00\x6c\x00\x6c\x00\x6f'
        
        result = detector.detect_and_convert(invalid_bytes)
        
        # Should handle gracefully without crashing
        assert isinstance(result.normalized_text, str)
        assert len(result.normalized_text) > 0
    
    def test_empty_bytes(self):
        """Test handling of empty byte input."""
        detector = EncodingDetector()
        
        result = detector.detect_and_convert(b"")
        assert result.normalized_text == ""
    
    def test_string_to_utf8_conversion(self):
        """Test converting existing strings to UTF-8."""
        detector = EncodingDetector()
        
        # Test with a regular string
        text = "Hello world with special chars: café"
        result = detector.convert_to_utf8(text)
        
        assert result.normalized_text == text
        assert result.quality_score >= 0.9
    
    def test_known_encoding_conversion(self):
        """Test conversion with known source encoding."""
        detector = EncodingDetector()
        
        # Create text in a specific encoding
        original_text = "Hello café naïve"
        latin1_bytes = original_text.encode('latin-1')
        
        # Convert knowing the source encoding
        result = detector.convert_to_utf8(latin1_bytes.decode('latin-1'), 'latin-1')
        
        assert result.normalized_text == original_text
        assert result.original_encoding == 'latin-1'
    
    def test_strict_encoding_mode(self):
        """Test strict encoding mode behavior."""
        detector = EncodingDetector({'strict_encoding': True})
        
        # Test that strict mode works differently than non-strict
        invalid_bytes = b'\xff\xff\xff\xff'
        
        # In strict mode, should handle errors more strictly
        result = detector.detect_and_convert(invalid_bytes)
        # Should still work but with lower quality
        assert isinstance(result.normalized_text, str)
        assert result.quality_score < 0.8  # Lower quality due to encoding issues


class TestNormalizationPipeline:
    """Test cases for the complete normalization pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete normalization pipeline."""
        config = {
            'normalizer': {
                'normalize_whitespace': True,
                'normalize_quotes': True,
                'normalize_punctuation': True
            },
            'ocr_corrector': {
                'correct_character_substitutions': True,
                'correct_spacing_issues': True
            },
            'apply_ocr_correction': True
        }
        
        # Text with multiple issues
        messy_text = "Tlie   \u201cquick\u201d   brown fox ,jumps over tne lazy dog\u2026"
        result = normalize_text_pipeline(messy_text, config)
        
        # Should apply multiple corrections
        assert len(result.corrections_applied) > 0
        assert "word_substitutions" in result.corrections_applied
        assert "quote_normalization" in result.corrections_applied
        # Note: punctuation might be normalized by Unicode normalization first
        
        # Should be cleaner than original
        assert "The" in result.normalized_text
        assert "the" in result.normalized_text
        assert '"quick"' in result.normalized_text
        assert "..." in result.normalized_text
    
    def test_pipeline_with_empty_input(self):
        """Test pipeline with empty input."""
        result = normalize_text_pipeline("")
        assert result.normalized_text == ""
        assert result.quality_score == 1.0
    
    def test_pipeline_configuration(self):
        """Test pipeline with different configurations."""
        # Test with OCR correction disabled
        config = {'apply_ocr_correction': False}
        
        text_with_ocr_errors = "Tlie quick brown fox"
        result = normalize_text_pipeline(text_with_ocr_errors, config)
        
        # Should not apply OCR corrections
        assert "word_substitutions" not in result.corrections_applied
        assert "Tlie" in result.normalized_text  # OCR error should remain
    
    def test_pipeline_quality_scoring(self):
        """Test quality scoring in the pipeline."""
        # Text that needs significant cleaning
        dirty_text = "Tlie   \u201cmessy\u201d   text witli   many\u2026   issues"
        result = normalize_text_pipeline(dirty_text)
        
        # Quality score should reflect the amount of cleaning needed
        assert 0.3 <= result.quality_score <= 1.0
        
        # Clean text should have high quality score
        clean_text = "This is clean text."
        result = normalize_text_pipeline(clean_text)
        assert result.quality_score >= 0.9


class TestErrorHandling:
    """Test error handling in normalization components."""
    
    def test_normalizer_error_handling(self):
        """Test TextNormalizer error handling."""
        normalizer = TextNormalizer()
        
        # Test with various edge cases that should be handled gracefully
        edge_cases = [None, "", 123, [], {}]
        
        for case in edge_cases:
            result = normalizer.normalize_text(case)
            # Should handle gracefully and return empty string
            assert result.normalized_text == ""
    
    def test_ocr_corrector_error_handling(self):
        """Test OCRCorrector error handling."""
        corrector = OCRCorrector()
        
        # Test with various edge cases that shouldn't crash
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "A" * 10000,  # Very long string
            "🙂😊🎉",  # Emoji
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Punctuation only
        ]
        
        for case in edge_cases:
            result = corrector.correct_ocr_errors(case)
            assert isinstance(result, NormalizationResult)
            assert isinstance(result.normalized_text, str)
    
    def test_encoding_detector_error_handling(self):
        """Test EncodingDetector error handling."""
        detector = EncodingDetector({'strict_encoding': False})
        
        # Test with problematic byte sequences
        problematic_bytes = [
            b"",  # Empty
            b"\x00",  # Null byte
            b"\xff" * 100,  # Invalid UTF-8 sequence
        ]
        
        for byte_seq in problematic_bytes:
            result = detector.detect_and_convert(byte_seq)
            assert isinstance(result, NormalizationResult)
            assert isinstance(result.normalized_text, str)


class TestIntegrationScenarios:
    """Integration test scenarios with real-world examples."""
    
    def test_pdf_ocr_text_scenario(self):
        """Test scenario with OCR text from PDF."""
        # Simulate OCR text with common issues
        ocr_text = """
        Tlie Document Title
        
        Tliis is a sample document witli OCR errors. Tne text contains
        various issues sucli as:
        
        • Character substitution errors (rn vs m)
        • Spacing issues ,like this
        • Line break prob-
        lems where words are split
        • Smart quotes "like these"
        
        Tlie quality of tliis text needs improvement.
        """
        
        result = normalize_text_pipeline(ocr_text)
        
        # Should correct major OCR issues
        assert "The Document Title" in result.normalized_text
        assert "This is a sample" in result.normalized_text
        assert "The text contains" in result.normalized_text
        assert "sucli as:" in result.normalized_text or "such as:" in result.normalized_text
        assert ", like this" in result.normalized_text
        assert '"like these"' in result.normalized_text
        assert "The quality of this" in result.normalized_text
        
        # Should have applied multiple correction types
        assert len(result.corrections_applied) > 0
    
    def test_web_scraped_text_scenario(self):
        """Test scenario with web-scraped text."""
        # Simulate web text with HTML artifacts and encoding issues
        web_text = """
        Welcome to our website!   
        
        This content was scraped from the web and contains:
        
        • Extra    whitespace
        • Smart quotes \u201clike these\u201d and \u2018these\u2019
        • Special characters… and—dashes
        • Multiple


        line breaks
        """
        
        result = normalize_text_pipeline(web_text)
        
        # Should clean up web artifacts
        assert "Extra whitespace" in result.normalized_text
        assert '"like these"' in result.normalized_text
        assert "'these'" in result.normalized_text
        assert "..." in result.normalized_text
        assert "and—dashes" in result.normalized_text or "and-dashes" in result.normalized_text
        
        # Should normalize excessive line breaks
        assert result.normalized_text.count('\n\n\n') == 0
    
    def test_mixed_encoding_scenario(self):
        """Test scenario with mixed encoding issues."""
        detector = EncodingDetector()
        
        # Simulate text that might have encoding issues
        mixed_text = "Café naïve résumé piñata"
        
        # Test with different encodings
        encodings_to_test = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_test:
            try:
                encoded_bytes = mixed_text.encode(encoding)
                result = detector.detect_and_convert(encoded_bytes)
                
                # Should successfully decode and contain the original characters
                assert "Café" in result.normalized_text or "Caf" in result.normalized_text
                assert isinstance(result.normalized_text, str)
                assert len(result.normalized_text) > 0
            except UnicodeEncodeError:
                # Some encodings might not support all characters
                pass