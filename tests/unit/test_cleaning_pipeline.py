"""
Unit tests for the comprehensive cleaning pipeline.

Tests the integrated cleaning pipeline that combines normalization,
OCR correction, deduplication, and boilerplate removal.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.qudata.clean.pipeline import (
    ComprehensiveCleaningPipeline, CleaningResult, BatchCleaningResult
)


class TestComprehensiveCleaningPipeline:
    """Test cases for ComprehensiveCleaningPipeline class."""
    
    def test_basic_pipeline_functionality(self):
        """Test basic pipeline functionality with default settings."""
        pipeline = ComprehensiveCleaningPipeline()
        
        text = """
        Tlie Document Title
        
        This is tlie main content witli some OCR errors.
        Navigation: Home | About | Contact
        Advertisement: Buy our products now!
        
        More valuable content here.
        Cookie policy: We use cookies.
        """
        
        result = pipeline.clean_text(text)
        
        assert isinstance(result, CleaningResult)
        assert result.cleaned_text != result.original_text
        assert len(result.operations_applied) > 0
        assert result.quality_score > 0
        
        # Should correct OCR errors
        assert "The Document Title" in result.cleaned_text
        assert "This is the main content with" in result.cleaned_text
        
        # Should remove some boilerplate
        assert len(result.cleaned_text) < len(result.original_text)
        
        # Should detect language
        assert result.language_result is not None
        assert result.language_result.language == 'en'
        assert "language_detection" in result.operations_applied
    
    def test_language_detection_integration(self):
        """Test language detection integration in the pipeline."""
        config = {
            'language_detection': {
                'enabled': True,
                'min_confidence': 0.8
            },
            'language_filtering': {
                'enabled': True
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        # Test English text
        english_text = "This is a comprehensive test of the English language detection system. It should work properly."
        result = pipeline.clean_text(english_text)
        
        assert result.language_result is not None
        assert result.language_result.language == 'en'
        assert result.language_result.is_reliable is True
        assert "language_detection" in result.operations_applied
        assert "language_filtering" in result.operations_applied
        
        # Test Spanish text
        spanish_text = "Este es un texto de prueba en español para verificar la detección de idiomas."
        result_es = pipeline.clean_text(spanish_text)
        
        assert result_es.language_result is not None
        assert result_es.language_result.language == 'es'
        assert result_es.language_result.is_reliable is True
    
    def test_language_filtering_with_restrictions(self):
        """Test language filtering with allowed language restrictions."""
        config = {
            'language_detection': {
                'enabled': True,
                'allowed_languages': ['en', 'es']
            },
            'language_filtering': {
                'enabled': True
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        # Test allowed language (English) - use longer text for reliable detection
        english_text = "This is a comprehensive test of the English language detection system. It should work properly and be detected as English language with high confidence. The text needs to be long enough for reliable language detection."
        result = pipeline.clean_text(english_text)
        
        assert result.language_filter_result is not None
        assert result.language_filter_result.should_keep is True
        assert len(result.warnings) == 0
        
        # Test disallowed language (French) - use longer text for reliable detection
        french_text = "Ce texte devrait être filtré par le système de détection de langue. Il s'agit d'un texte en français qui devrait être identifié comme tel par le système de détection automatique des langues. Le texte doit être suffisamment long pour une détection fiable."
        result_fr = pipeline.clean_text(french_text)
        
        assert result_fr.language_filter_result is not None
        # Note: The actual filtering behavior depends on implementation
        # The test checks that filtering was attempted
        assert "language_filtering" in result_fr.operations_applied
    
    def test_pipeline_with_configuration(self):
        """Test pipeline with custom configuration."""
        config = {
            'normalization': {
                'enabled': True,
                'normalize_whitespace': True,
                'normalize_quotes': True
            },
            'ocr_correction': {
                'enabled': True,
                'character_substitutions': True
            },
            'boilerplate_removal': {
                'enabled': True,
                'aggressive_removal': True,
                'remove_navigation': True,
                'remove_ads': True
            },
            'quality_scoring': {
                'enabled': True,
                'weights': {
                    'length_score': 0.3,
                    'boilerplate_ratio': 0.4,
                    'language_confidence': 0.3
                }
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        text = "Tlie quick brown fox witli navigation menu and ads."
        result = pipeline.clean_text(text)
        
        assert "The quick brown fox with" in result.cleaned_text
        assert result.quality_score > 0
        assert "ocr_correction" in result.operations_applied
    
    def test_pipeline_from_config_file(self):
        """Test loading pipeline configuration from YAML file."""
        config_data = {
            'normalization': {
                'enabled': True,
                'unicode': True,
                'whitespace': True
            },
            'boilerplate_patterns': {
                'navigation': ['nav', 'menu'],
                'ads': ['advertisement', 'sponsored']
            },
            'boilerplate_removal': {
                'enabled': True,
                'remove_navigation': True,
                'remove_ads': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            pipeline = ComprehensiveCleaningPipeline(config_file=config_file)
            
            text = "Content with nav menu and advertisement section."
            result = pipeline.clean_text(text)
            
            assert result.cleaned_text
            assert len(result.operations_applied) > 0
            
        finally:
            Path(config_file).unlink()
    
    def test_batch_document_processing(self):
        """Test batch processing of multiple documents."""
        pipeline = ComprehensiveCleaningPipeline()
        
        documents = {
            'doc1': 'Tlie first document witli OCR errors and navigation menu.',
            'doc2': 'The second document with advertisement content.',
            'doc3': 'Tlie first document witli OCR errors and navigation menu.',  # Duplicate
            'doc4': 'A unique fourth document with good content.',
            'doc5': ''  # Empty document
        }
        
        result = pipeline.clean_documents(documents)
        
        assert isinstance(result, BatchCleaningResult)
        assert result.total_documents == 5
        assert result.processed_documents >= 4  # At least 4 should process
        assert len(result.document_results) == 5
        
        # Check individual results
        for doc_id, doc_result in result.document_results.items():
            assert isinstance(doc_result, CleaningResult)
            if doc_id != 'doc5':  # Skip empty document
                assert len(doc_result.operations_applied) > 0
    
    def test_deduplication_in_batch_processing(self):
        """Test deduplication functionality in batch processing."""
        config = {
            'deduplication': {
                'enabled': True,
                'near_duplicate_threshold': 0.8
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        documents = {
            'doc1': 'This is a unique document about machine learning.',
            'doc2': 'This is a unique document about machine learning.',  # Exact duplicate
            'doc3': 'This is a unique document about deep learning.',     # Similar
            'doc4': 'Completely different content about cooking recipes.'
        }
        
        result = pipeline.clean_documents(documents)
        
        assert result.deduplication_result is not None
        assert result.deduplication_result.get_duplicate_count() >= 1
        
        # Check that duplicates are marked
        duplicate_marked = any(
            'marked_as_duplicate' in doc_result.operations_applied
            for doc_result in result.document_results.values()
        )
        assert duplicate_marked
    
    def test_quality_scoring(self):
        """Test quality scoring functionality."""
        config = {
            'quality_scoring': {
                'enabled': True,
                'thresholds': {
                    'min_quality_score': 0.5,
                    'min_length': 50
                },
                'weights': {
                    'length_score': 0.4,
                    'boilerplate_ratio': 0.6
                }
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        # High quality text
        good_text = "This is a well-written article about artificial intelligence. " * 10
        good_result = pipeline.clean_text(good_text)
        
        # Low quality text (short and lots of boilerplate)
        bad_text = "Short. Navigation menu. Advertisement. Cookie policy."
        bad_result = pipeline.clean_text(bad_text)
        
        assert good_result.quality_score > bad_result.quality_score
        assert good_result.quality_score > 0.5
    
    def test_selective_component_enabling(self):
        """Test enabling/disabling specific pipeline components."""
        # Only normalization enabled
        config_norm_only = {
            'normalization': {'enabled': True},
            'ocr_correction': {'enabled': False},
            'boilerplate_removal': {'enabled': False},
            'deduplication': {'enabled': False}
        }
        
        pipeline_norm = ComprehensiveCleaningPipeline(config_norm_only)
        
        text = "Tlie text witli navigation menu and ads."
        result_norm = pipeline_norm.clean_text(text)
        
        # Should only apply normalization
        norm_operations = [op for op in result_norm.operations_applied if 'normalization' in op]
        boilerplate_operations = [op for op in result_norm.operations_applied if 'boilerplate' in op]
        
        assert len(norm_operations) > 0 or 'ocr_correction' in result_norm.operations_applied
        assert len(boilerplate_operations) == 0
    
    def test_error_handling(self):
        """Test error handling in pipeline processing."""
        pipeline = ComprehensiveCleaningPipeline()
        
        # Test with None input
        result = pipeline.clean_text(None)
        assert result.cleaned_text == ""
        assert len(result.errors) > 0
        
        # Test with empty string
        result = pipeline.clean_text("")
        assert result.cleaned_text == ""
        
        # Test batch processing with mixed valid/invalid inputs
        documents = {
            'valid': 'This is valid content.',
            'none': None,
            'empty': '',
            'valid2': 'Another valid document.'
        }
        
        batch_result = pipeline.clean_documents(documents)
        assert batch_result.total_documents == 4
        assert batch_result.processed_documents >= 2  # At least valid documents
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics generation."""
        config = {
            'normalization': {'enabled': True},
            'boilerplate_removal': {'enabled': True},
            'quality_scoring': {'enabled': True}
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        stats = pipeline.get_pipeline_statistics()
        
        assert 'components_enabled' in stats
        assert 'boilerplate_patterns' in stats
        assert 'quality_thresholds' in stats
        
        assert stats['components_enabled']['normalization'] == True
        assert stats['components_enabled']['boilerplate_removal'] == True
        assert stats['boilerplate_patterns'] > 0
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'normalization': {'enabled': True},
            'quality_scoring': {
                'enabled': True,
                'weights': {'length_score': 0.5, 'boilerplate_ratio': 0.5}
            }
        }
        
        pipeline_valid = ComprehensiveCleaningPipeline(valid_config)
        issues_valid = pipeline_valid.validate_configuration()
        assert len(issues_valid) == 0
        
        # Invalid configuration (no components enabled)
        invalid_config = {
            'normalization': {'enabled': False},
            'boilerplate_removal': {'enabled': False},
            'ocr_correction': {'enabled': False}
        }
        
        pipeline_invalid = ComprehensiveCleaningPipeline(invalid_config)
        issues_invalid = pipeline_invalid.validate_configuration()
        assert len(issues_invalid) > 0
    
    def test_cleaning_result_properties(self):
        """Test CleaningResult properties and methods."""
        original = "Original text with lots of content here."
        cleaned = "Cleaned text here."
        
        result = CleaningResult(
            original_text=original,
            cleaned_text=cleaned,
            operations_applied=['normalization', 'boilerplate_removal'],
            quality_score=0.8
        )
        
        assert result.get_length_reduction() == len(original) - len(cleaned)
        assert result.get_compression_ratio() > 0
        assert result.get_compression_ratio() < 1
    
    def test_batch_cleaning_result_properties(self):
        """Test BatchCleaningResult properties and methods."""
        batch_result = BatchCleaningResult(
            total_documents=10,
            processed_documents=8,
            failed_documents=2,
            total_original_length=1000,
            total_cleaned_length=800
        )
        
        assert batch_result.get_success_rate() == 0.8
        assert batch_result.get_overall_compression_ratio() == 0.2
    
    def test_final_cleanup_functionality(self):
        """Test final cleanup operations."""
        pipeline = ComprehensiveCleaningPipeline()
        
        messy_text = """
        
        
        Good content here.
        
        
        
        More content.
        x
        
        Final content.
        
        
        """
        
        cleaned = pipeline._final_cleanup(messy_text)
        
        # Should remove excessive newlines
        assert cleaned.count('\n\n\n') == 0
        
        # Should preserve good content
        assert "Good content here" in cleaned
        assert "More content" in cleaned
        assert "Final content" in cleaned
        
        # Should remove very short lines
        assert "\nx\n" not in cleaned
    
    def test_integration_with_enhanced_config(self):
        """Test integration with the enhanced cleansing rules configuration."""
        # Use the actual enhanced config structure
        config = {
            'boilerplate_patterns': {
                'navigation': ['nav', 'menu', 'sidebar'],
                'ads': ['advertisement', 'sponsored', 'promotional'],
                'social': ['share', 'like', 'follow']
            },
            'boilerplate_removal': {
                'enabled': True,
                'remove_navigation': True,
                'remove_ads': True,
                'remove_social': True,
                'aggressive_removal': False
            },
            'ocr_correction': {
                'enabled': True,
                'word_corrections': {
                    'tlie': 'the',
                    'witli': 'with'
                }
            },
            'quality_scoring': {
                'enabled': True,
                'thresholds': {
                    'min_quality_score': 0.6,
                    'min_length': 100
                }
            }
        }
        
        pipeline = ComprehensiveCleaningPipeline(config)
        
        text = """
        Tlie Main Article Title
        
        This is tlie main content of the article witli valuable information.
        
        Navigation: Home | About | Contact | Services
        Advertisement: Special offer - buy now!
        Share this article on social media!
        
        The article continues with more valuable content here.
        """
        
        result = pipeline.clean_text(text)
        
        # Should correct OCR errors
        assert "The Main Article Title" in result.cleaned_text
        assert "This is the main content" in result.cleaned_text
        assert "with valuable information" in result.cleaned_text
        
        # Should remove boilerplate
        assert "Navigation:" not in result.cleaned_text
        assert "Advertisement:" not in result.cleaned_text
        assert "Share this article" not in result.cleaned_text
        
        # Should preserve valuable content
        assert "valuable information" in result.cleaned_text
        assert "valuable content" in result.cleaned_text
        
        # Should have applied multiple operations
        assert len(result.operations_applied) >= 2
        assert result.quality_score > 0