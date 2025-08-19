"""
Unit tests for comprehensive preprocessing functionality.

Tests all subtasks of task 28: comprehensive preprocessing for high-quality LLM training datasets.
"""

import pytest
import json
import yaml
import csv
import io
from pathlib import Path
from unittest.mock import Mock, patch

from src.qudata.clean.comprehensive_preprocessor import (
    ComprehensivePreprocessor,
    PreprocessingResult,
    BatchPreprocessingResult,
    preprocess_for_llm_training,
    validate_dataset_cleanliness
)


class TestComprehensivePreprocessor:
    """Test comprehensive preprocessing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'quality': {
                'min_content_length': 20,
                'max_content_length': 10000,
                'min_quality_score': 0.5
            },
            'language': {
                'target_languages': ['en'],
                'min_confidence': 0.7
            },
            'deduplication': {
                'similarity_threshold': 0.85,
                'cross_format': True
            }
        }
        self.preprocessor = ComprehensivePreprocessor(self.config)
    
    def test_text_document_preprocessing(self):
        """Test text document preprocessing (PDF, DOCX, ODT, RTF, TXT, MD)."""
        # Test content with headers, footers, watermarks
        content = """Page 1 of 10
        
        © 2023 Company Name
        Confidential Draft
        Downloaded from XYZ
        
        This is the main content of the document.
        This paragraph contains useful information.
        
        This is the main content of the document.  # Duplicate paragraph
        
        Generated on 2023-01-01
        Last updated: 2023-01-01
        """
        
        result = self.preprocessor.preprocess_content(content, 'pdf')
        
        assert result.validation_passed
        assert 'header_footer_removal' in result.stages_applied
        assert 'paragraph_deduplication' in result.stages_applied
        assert 'whitespace_normalization' in result.stages_applied
        
        # Check that boilerplate was removed
        assert '© 2023 Company Name' not in result.processed_content
        assert 'Confidential Draft' not in result.processed_content
        assert 'Downloaded from XYZ' not in result.processed_content
        assert 'Page 1 of 10' not in result.processed_content
        
        # Check that main content is preserved
        assert 'This is the main content of the document.' in result.processed_content
        assert 'This paragraph contains useful information.' in result.processed_content
        
        # Check that duplicates were removed
        content_lines = result.processed_content.split('\n')
        main_content_count = sum(1 for line in content_lines if 'main content of the document' in line)
        assert main_content_count == 1
    
    def test_web_content_preprocessing(self):
        """Test web content preprocessing (HTML, XML)."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Skip to main content</nav>
            <div class="sidebar">Advertisement</div>
            <main>
                <h1>Main Article Title</h1>
                <p>This is the main article content.</p>
                <p>Published on January 1, 2023</p>
                <div class="related">You might also like</div>
                <footer>© 2023 Website Name</footer>
            </main>
            <script>analytics code</script>
        </body>
        </html>
        """
        
        result = self.preprocessor.preprocess_content(html_content, 'html')
        
        assert result.validation_passed
        assert 'html_cleaning' in result.stages_applied
        assert 'web_noise_removal' in result.stages_applied
        
        # Check that HTML tags were removed
        assert '<html>' not in result.processed_content
        assert '<script>' not in result.processed_content
        
        # Check that navigation and ads were removed
        assert 'Skip to main content' not in result.processed_content
        assert 'Advertisement' not in result.processed_content
        assert 'You might also like' not in result.processed_content
        
        # Check that main content is preserved
        assert 'Main Article Title' in result.processed_content
        assert 'This is the main article content.' in result.processed_content
    
    def test_structured_data_preprocessing(self):
        """Test structured data preprocessing (CSV, JSON, JSONL, YAML)."""
        # Test JSON preprocessing
        json_data = {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john@example.com",
            "phone": "N/A",
            "address": "",
            "notes": "---",
            "duplicateField": "value",
            "duplicate_field": "value"  # Should be deduplicated
        }
        json_content = json.dumps(json_data)
        
        result = self.preprocessor.preprocess_content(json_content, 'json')
        
        assert result.validation_passed
        assert 'structured_data_cleaning' in result.stages_applied
        
        # Parse the cleaned JSON
        cleaned_data = json.loads(result.processed_content)
        
        # Check field name normalization
        assert 'first_name' in cleaned_data
        assert 'last_name' in cleaned_data
        
        # Check placeholder removal
        assert 'phone' not in cleaned_data or cleaned_data.get('phone') != 'N/A'
        assert 'address' not in cleaned_data or cleaned_data.get('address') != ''
        assert 'notes' not in cleaned_data or cleaned_data.get('notes') != '---'
    
    def test_csv_data_preprocessing(self):
        """Test CSV data preprocessing."""
        csv_content = """FirstName,LastName,Email,Phone,Notes
John,Doe,john@example.com,555-1234,Good customer
Jane,Smith,jane@example.com,N/A,---
John,Doe,john@example.com,555-1234,Good customer
Bob,Johnson,bob@example.com,,
"""
        
        result = self.preprocessor.preprocess_content(csv_content, 'csv')
        
        assert result.validation_passed
        assert 'structured_data_cleaning' in result.stages_applied
        
        # Parse the cleaned CSV
        csv_reader = csv.reader(io.StringIO(result.processed_content))
        rows = list(csv_reader)
        
        # Check header normalization
        header = rows[0]
        assert 'first_name' in header
        assert 'last_name' in header
        
        # Check duplicate removal (should have 3 unique rows including header)
        assert len(rows) == 4  # header + 3 unique data rows
    
    def test_image_ocr_preprocessing(self):
        """Test image OCR preprocessing (PNG, JPG, TIFF)."""
        # Simulate OCR content with common errors
        ocr_content = """
        [WATERMARK]
        LOGO: Company Name
        
        Tlie quick brown fox jumps over tlie lazy dog.
        Tliis is a test of OCR error correction.
        
        rn example of character confusion: cl vs d, ii vs n.
        
        Figure 1: Decorative caption
        
        |___|___|___|  # Border artifacts
        """
        
        result = self.preprocessor.preprocess_content(ocr_content, 'png')
        
        assert result.validation_passed
        assert 'ocr_correction' in result.stages_applied
        assert 'image_artifact_removal' in result.stages_applied
        assert 'image_noise_cleaning' in result.stages_applied
        
        # Check OCR error correction
        assert 'The quick brown fox' in result.processed_content
        assert 'This is a test' in result.processed_content
        
        # Check artifact removal
        assert '[WATERMARK]' not in result.processed_content
        assert 'LOGO:' not in result.processed_content
        assert 'Figure 1:' not in result.processed_content
        assert '|___|' not in result.processed_content
    
    def test_code_preprocessing(self):
        """Test code and Jupyter notebook preprocessing."""
        code_content = """
        import os
        import sys
        import unused_module  # This import is not used
        
        # TODO: Fix this later
        # DEBUG: Remove this
        
        api_key = "sk-1234567890abcdef"
        password = "secret123"
        
        def main():
            '''This is a meaningful docstring.'''
            print("Hello, world!")
            # This is a meaningful comment explaining the logic
            return 0
        
        # Short comment
        
        Out[1]: <matplotlib.figure.Figure at 0x7f8b8c0d5f40>
        
        if __name__ == "__main__":
            main()
        """
        
        result = self.preprocessor.preprocess_content(code_content, 'py')
        
        assert result.validation_passed
        assert 'code_noise_removal' in result.stages_applied
        assert 'secret_sanitization' in result.stages_applied
        assert 'code_structure_cleaning' in result.stages_applied
        
        # Check secret sanitization
        assert 'sk-1234567890abcdef' not in result.processed_content
        assert 'secret123' not in result.processed_content
        assert '[API_KEY]' in result.processed_content or '[REDACTED]' in result.processed_content
        
        # Check noise removal
        assert 'TODO:' not in result.processed_content
        assert 'DEBUG:' not in result.processed_content
        assert 'Out[1]:' not in result.processed_content
        
        # Check meaningful content preservation
        assert 'meaningful docstring' in result.processed_content
        assert 'meaningful comment' in result.processed_content
    
    def test_cross_format_global_validation(self):
        """Test cross-format global validation and cleanup."""
        content = """
        This is some content with    multiple   spaces.
        
        
        
        This has excessive line breaks.
        
        As an AI language model, I cannot provide that information.
        
        © 2023 Copyright notice
        Best regards,
        John Doe
        
        john.doe@example.com
        555-123-4567
        123-45-6789
        """
        
        result = self.preprocessor.preprocess_content(content, 'txt')
        
        assert result.validation_passed
        assert 'global_boilerplate_removal' in result.stages_applied
        assert 'space_normalization' in result.stages_applied
        assert 'generated_content_removal' in result.stages_applied
        assert 'pii_sanitization' in result.stages_applied
        
        # Check space normalization
        assert '   ' not in result.processed_content
        
        # Check boilerplate removal
        assert 'As an AI language model' not in result.processed_content
        assert '© 2023' not in result.processed_content
        assert 'Best regards' not in result.processed_content
        
        # Check PII sanitization
        assert 'john.doe@example.com' not in result.processed_content
        assert '[EMAIL]' in result.processed_content
        assert '[PHONE]' in result.processed_content
        assert '[SSN]' in result.processed_content
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing with cross-document deduplication."""
        documents = {
            'doc1': 'This is a unique document with original content.',
            'doc2': 'This is a unique document with original content.',  # Duplicate
            'doc3': 'This is another document with different content.',
            'doc4': 'This is a very similar document with original content.',  # Near duplicate
        }
        
        batch_result = self.preprocessor.preprocess_batch(documents)
        
        assert batch_result.total_documents == 4
        assert batch_result.processed_documents >= 2  # At least 2 unique documents
        assert len(batch_result.duplicate_groups) > 0
        assert len(batch_result.removed_duplicates) > 0
        
        # Check that duplicates were identified
        assert batch_result.get_success_rate() > 0.5
    
    def test_archive_preprocessing(self):
        """Test archive preprocessing with system file filtering."""
        content_map = {
            'document1.txt': 'This is a valid document.',
            'document2.pdf': 'This is another valid document.',
            '.DS_Store': 'System file content',
            'Thumbs.db': 'Another system file',
            '.gitignore': 'Git ignore file',
            'document1.txt': 'This is a valid document.',  # Duplicate
        }
        
        batch_result = self.preprocessor.preprocess_archive_content('test.zip', content_map)
        
        # System files should be filtered out
        processed_files = list(batch_result.document_results.keys())
        assert '.DS_Store' not in processed_files
        assert 'Thumbs.db' not in processed_files
        assert '.gitignore' not in processed_files
        
        # Valid documents should be processed
        assert any('document1.txt' in f for f in processed_files)
        assert any('document2.pdf' in f for f in processed_files)
    
    def test_quality_validation(self):
        """Test preprocessing quality validation."""
        # Create test results with different quality levels
        results = [
            PreprocessingResult(
                original_content="High quality content",
                processed_content="High quality content",
                content_type="txt",
                quality_score=0.9,
                validation_passed=True
            ),
            PreprocessingResult(
                original_content="Medium quality content",
                processed_content="Medium quality content",
                content_type="txt",
                quality_score=0.7,
                validation_passed=True
            ),
            PreprocessingResult(
                original_content="Low quality content",
                processed_content="",
                content_type="txt",
                quality_score=0.3,
                validation_passed=False,
                validation_errors=["Content too short"]
            )
        ]
        
        validation = self.preprocessor.validate_preprocessing_quality(results)
        
        assert validation['total_documents'] == 3
        assert validation['passed_validation'] == 2
        assert validation['failed_validation'] == 1
        assert validation['quality_distribution']['high'] == 1
        assert validation['quality_distribution']['medium'] == 1
        assert validation['quality_distribution']['low'] == 1
        assert 'average_quality' in validation
        assert len(validation['recommendations']) > 0
    
    def test_dataset_cleanliness_validation(self):
        """Test automated dataset cleanliness validation."""
        dataset = {
            'doc1': 'This is a clean document with proper sentences.',
            'doc2': 'This has broken sentences like this One.',  # Broken sentence
            'doc3': 'bcdfghjklmnpqrstvwxyz' * 10,  # Gibberish
            'doc4': 'This has @@@@@ excessive ##### special !!!!!! characters.',
            'doc5': 'Normal content with proper formatting.',
        }
        
        validation = validate_dataset_cleanliness(dataset)
        
        assert validation['total_documents'] == 5
        assert validation['failed_documents'] > 0
        assert len(validation['issues_found']) > 0
        assert len(validation['recommendations']) > 0
        
        # Check specific issue detection
        issues = [issue['issues'] for issue in validation['issues_found']]
        flat_issues = [item for sublist in issues for item in sublist]
        
        assert any('broken sentences' in issue.lower() for issue in flat_issues)
        assert any('gibberish' in issue.lower() for issue in flat_issues)
        assert any('special characters' in issue.lower() for issue in flat_issues)
    
    def test_preprocessing_statistics(self):
        """Test preprocessing statistics generation."""
        stats = self.preprocessor.get_preprocessing_statistics()
        
        assert 'supported_formats' in stats
        assert 'quality_thresholds' in stats
        assert 'language_settings' in stats
        assert 'deduplication_settings' in stats
        assert 'pattern_counts' in stats
        
        # Check format support
        formats = stats['supported_formats']
        assert 'pdf' in formats['text_formats']
        assert 'html' in formats['web_formats']
        assert 'json' in formats['structured_formats']
        assert 'png' in formats['image_formats']
        assert 'py' in formats['code_formats']
        assert 'zip' in formats['archive_formats']
    
    def test_convenience_function(self):
        """Test convenience function for preprocessing."""
        content = "This is a test document with some content."
        result = preprocess_for_llm_training(content, 'txt', self.config)
        
        assert isinstance(result, PreprocessingResult)
        assert result.validation_passed
        assert len(result.processed_content) > 0
    
    def test_empty_content_handling(self):
        """Test handling of empty or invalid content."""
        # Test empty content
        result = self.preprocessor.preprocess_content("", 'txt')
        assert not result.validation_passed
        assert "Empty or invalid content" in result.validation_errors
        
        # Test None content
        result = self.preprocessor.preprocess_content(None, 'txt')
        assert not result.validation_passed
        
        # Test very short content
        result = self.preprocessor.preprocess_content("Hi", 'txt')
        assert "Content too short" in result.validation_errors
    
    def test_unsupported_format_fallback(self):
        """Test fallback to text processing for unsupported formats."""
        content = "This is content in an unsupported format."
        result = self.preprocessor.preprocess_content(content, 'unknown')
        
        # Should fall back to text processing
        assert result.validation_passed
        assert len(result.processed_content) > 0
    
    def test_error_handling(self):
        """Test error handling in preprocessing."""
        # Test malformed JSON
        malformed_json = '{"key": "value", "invalid": }'
        result = self.preprocessor.preprocess_content(malformed_json, 'json')
        
        # Should fall back to text processing
        assert "Structured data parsing failed" in result.validation_errors
        assert len(result.processed_content) > 0  # Fallback should work
    
    def test_language_filtering(self):
        """Test language detection and filtering."""
        # Test non-English content (if language filtering is enabled)
        spanish_content = "Este es un documento en español con contenido válido."
        result = self.preprocessor.preprocess_content(spanish_content, 'txt')
        
        # Should generate a warning about language
        if self.preprocessor.target_languages == {'en'}:
            assert any('Language' in warning for warning in result.validation_warnings)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # High quality content
        high_quality = "This is a well-written document with proper grammar, punctuation, and structure. It contains meaningful information that would be valuable for training language models."
        result = self.preprocessor.preprocess_content(high_quality, 'txt')
        assert result.quality_score > 0.7
        
        # Low quality content
        low_quality = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        result = self.preprocessor.preprocess_content(low_quality, 'txt')
        assert result.quality_score < 0.5


class TestPreprocessingIntegration:
    """Integration tests for comprehensive preprocessing."""
    
    def test_end_to_end_preprocessing(self):
        """Test end-to-end preprocessing pipeline."""
        # Create a complex document with multiple issues
        complex_content = """
        Page 1 of 5
        © 2023 Test Company
        CONFIDENTIAL DRAFT
        
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Content</h1>
            <p>This is the main content of the document.</p>
            <p>Tlie quick brown fox jumps over tlie lazy dog.</p>
            <div class="ads">Advertisement content</div>
            <p>Contact us at test@example.com or call 555-123-4567.</p>
            <script>analytics();</script>
        </body>
        </html>
        
        As an AI language model, I must inform you that this is generated content.
        
        Best regards,
        Test Author
        """
        
        preprocessor = ComprehensivePreprocessor()
        result = preprocessor.preprocess_content(complex_content, 'html')
        
        # Verify comprehensive cleaning
        assert result.validation_passed
        assert len(result.stages_applied) > 5
        
        # Check that various cleaning stages were applied
        expected_stages = [
            'html_cleaning',
            'web_noise_removal',
            'global_boilerplate_removal',
            'generated_content_removal',
            'pii_sanitization'
        ]
        
        for stage in expected_stages:
            assert stage in result.stages_applied
        
        # Verify content quality
        assert result.quality_score > 0.5
        assert len(result.processed_content) > 50
        
        # Check specific cleaning results
        assert '<html>' not in result.processed_content
        assert 'Page 1 of 5' not in result.processed_content
        assert 'CONFIDENTIAL DRAFT' not in result.processed_content
        assert 'Advertisement content' not in result.processed_content
        assert 'As an AI language model' not in result.processed_content
        assert '[EMAIL]' in result.processed_content
        assert '[PHONE]' in result.processed_content
        
        # Main content should be preserved
        assert 'Main Content' in result.processed_content
        assert 'main content of the document' in result.processed_content


if __name__ == '__main__':
    pytest.main([__file__])