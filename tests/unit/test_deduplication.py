"""
Unit tests for deduplication functionality.

Tests the DeduplicationEngine for exact, near, and fuzzy duplicate detection
with various content types and edge cases.
"""

import pytest
from src.qudata.clean.dedupe import (
    DeduplicationEngine, DuplicateGroup, DeduplicationResult
)
from src.qudata.models import ProcessingError


class TestDeduplicationEngine:
    """Test cases for DeduplicationEngine class."""
    
    def test_exact_duplicate_detection(self):
        """Test exact duplicate detection."""
        config = {'min_content_length': 10}  # Lower threshold for test
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': 'This is a test document with enough content for deduplication.',
            'doc2': 'This is a different document with unique content here.',
            'doc3': 'This is a test document with enough content for deduplication.',  # Exact duplicate of doc1
            'doc4': 'Another unique document with different content altogether.',
            'doc5': 'This is a test document with enough content for deduplication.'   # Another exact duplicate of doc1
        }
        
        result = engine.deduplicate_documents(documents)
        
        assert result.original_count == 5
        assert result.unique_count == 3
        assert result.get_duplicate_count() == 2
        
        # Should have one duplicate group with doc1 as representative
        assert len(result.duplicate_groups) == 1
        group = result.duplicate_groups[0]
        assert group.representative_id == 'doc1'
        assert set(group.duplicate_ids) == {'doc3', 'doc5'}
        assert group.duplicate_type == 'exact'
        
        # All duplicates should have similarity score of 1.0
        for dup_id in group.duplicate_ids:
            assert group.similarity_scores[dup_id] == 1.0
    
    def test_near_duplicate_detection(self):
        """Test near-duplicate detection with high similarity."""
        config = {
            'near_duplicate_threshold': 0.8,
            'case_sensitive': False
        }
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': 'The quick brown fox jumps over the lazy dog.',
            'doc2': 'The quick brown fox jumps over a lazy dog.',  # Very similar
            'doc3': 'A completely different document about cats.',
            'doc4': 'The quick brown fox leaps over the lazy dog.'  # Similar to doc1
        }
        
        result = engine.deduplicate_documents(documents)
        
        assert result.original_count == 4
        assert result.get_duplicate_count() >= 1  # At least one duplicate found
        
        # Check that similar documents are grouped
        found_similar = False
        for group in result.duplicate_groups:
            if group.representative_id == 'doc1':
                # Should find doc2 and/or doc4 as similar
                assert len(group.duplicate_ids) >= 1
                found_similar = True
                break
        
        assert found_similar
    
    def test_fuzzy_duplicate_detection(self):
        """Test fuzzy duplicate detection with word-level similarity."""
        config = {
            'near_duplicate_threshold': 0.95,  # Very high threshold for near-duplicates
            'fuzzy_match_threshold': 0.4,     # Lower threshold for fuzzy matching
        }
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': 'machine learning artificial intelligence data science research',
            'doc2': 'data science machine learning AI algorithms programming',  # Similar words
            'doc3': 'cooking recipes food preparation kitchen utensils',
            'doc4': 'artificial intelligence machine learning models training'  # Similar to doc1
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should find some duplicates (either near or fuzzy)
        total_duplicates = result.get_duplicate_count()
        assert total_duplicates >= 1
    
    def test_empty_and_short_documents(self):
        """Test handling of empty and very short documents."""
        config = {'min_content_length': 10}
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': '',  # Empty
            'doc2': 'Hi',  # Too short
            'doc3': 'This is a proper document with enough content.',
            'doc4': 'Another proper document with sufficient length.',
            'doc5': 'This is a proper document with enough content.'  # Duplicate of doc3
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should only process documents that meet minimum length
        assert result.original_count == 5
        # Only doc3, doc4, doc5 should be processed, with doc5 being a duplicate
        assert result.unique_count == 4  # doc1, doc2 kept + doc3, doc4 unique
    
    def test_case_sensitivity_settings(self):
        """Test case sensitivity configuration."""
        # Case sensitive - use higher threshold to avoid near-duplicate detection
        config_sensitive = {
            'case_sensitive': True,
            'near_duplicate_threshold': 0.95  # Higher threshold
        }
        engine_sensitive = DeduplicationEngine(config_sensitive)
        
        documents = {
            'doc1': 'This is a Test Document.',
            'doc2': 'this is a test document.',  # Different case
        }
        
        result_sensitive = engine_sensitive.deduplicate_documents(documents)
        assert result_sensitive.get_duplicate_count() == 0  # No duplicates due to case
        
        # Case insensitive
        config_insensitive = {
            'case_sensitive': False,
            'near_duplicate_threshold': 0.95
        }
        engine_insensitive = DeduplicationEngine(config_insensitive)
        
        result_insensitive = engine_insensitive.deduplicate_documents(documents)
        assert result_insensitive.get_duplicate_count() == 1  # Should find duplicate
    
    def test_punctuation_handling(self):
        """Test punctuation handling in similarity calculation."""
        config = {'ignore_punctuation': True}
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': 'Hello, world! How are you?',
            'doc2': 'Hello world How are you',  # Same without punctuation
            'doc3': 'Goodbye, world! See you later.'
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should find doc1 and doc2 as duplicates
        assert result.get_duplicate_count() == 1
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization in deduplication."""
        config = {'normalize_whitespace': True}
        engine = DeduplicationEngine(config)
        
        documents = {
            'doc1': 'This is   a    test   document.',
            'doc2': 'This is a test document.',  # Same with normalized whitespace
            'doc3': 'This\tis\na\r\ntest\tdocument.'  # Same with different whitespace
        }
        
        result = engine.deduplicate_documents(documents)
        
        # All should be considered duplicates
        assert result.get_duplicate_count() == 2
        assert result.unique_count == 1
    
    def test_similarity_thresholds(self):
        """Test different similarity thresholds."""
        # High threshold - strict matching
        config_strict = {'near_duplicate_threshold': 0.95}
        engine_strict = DeduplicationEngine(config_strict)
        
        # Low threshold - lenient matching
        config_lenient = {'near_duplicate_threshold': 0.7}
        engine_lenient = DeduplicationEngine(config_lenient)
        
        documents = {
            'doc1': 'The quick brown fox jumps over the lazy dog.',
            'doc2': 'The quick brown fox jumps over a lazy dog.',  # Small difference
            'doc3': 'The fast brown fox leaps over the lazy dog.'  # More differences
        }
        
        result_strict = engine_strict.deduplicate_documents(documents)
        result_lenient = engine_lenient.deduplicate_documents(documents)
        
        # Lenient should find more duplicates than strict
        assert result_lenient.get_duplicate_count() >= result_strict.get_duplicate_count()
    
    def test_duplicate_group_properties(self):
        """Test DuplicateGroup properties and methods."""
        group = DuplicateGroup(
            representative_id='doc1',
            duplicate_ids=['doc2', 'doc3'],
            similarity_scores={'doc2': 0.95, 'doc3': 0.87},
            duplicate_type='near'
        )
        
        assert group.get_all_ids() == ['doc1', 'doc2', 'doc3']
        assert group.get_duplicate_count() == 2
        assert group.duplicate_type == 'near'
    
    def test_deduplication_result_properties(self):
        """Test DeduplicationResult properties and methods."""
        result = DeduplicationResult(
            original_count=10,
            unique_count=7
        )
        
        # Add some duplicate groups
        group1 = DuplicateGroup('doc1', ['doc2', 'doc3'])
        group2 = DuplicateGroup('doc4', ['doc5'])
        result.duplicate_groups = [group1, group2]
        result.removed_ids = {'doc2', 'doc3', 'doc5'}
        
        assert result.get_duplicate_count() == 3
        assert result.get_deduplication_ratio() == 0.3  # 3/10
    
    def test_statistics_generation(self):
        """Test generation of deduplication statistics."""
        engine = DeduplicationEngine()
        
        documents = {
            'doc1': 'Exact duplicate content.',
            'doc2': 'Exact duplicate content.',  # Exact duplicate
            'doc3': 'Near duplicate content here.',
            'doc4': 'Near duplicate content there.',  # Near duplicate
            'doc5': 'Unique content that stands alone.'
        }
        
        result = engine.deduplicate_documents(documents)
        stats = engine.get_duplicate_statistics(result)
        
        assert 'original_count' in stats
        assert 'unique_count' in stats
        assert 'duplicate_count' in stats
        assert 'deduplication_ratio' in stats
        assert 'duplicate_groups' in stats
        assert 'by_type' in stats
        
        assert stats['original_count'] == 5
        assert stats['duplicate_count'] >= 1
    
    def test_cache_management(self):
        """Test cache management functionality."""
        engine = DeduplicationEngine()
        
        documents = {
            'doc1': 'Test document for cache.',
            'doc2': 'Another test document.'
        }
        
        # Process documents to populate cache
        engine.deduplicate_documents(documents)
        
        # Check that cache has entries
        assert len(engine._content_hashes) > 0
        assert len(engine._normalized_content) > 0
        
        # Clear cache
        engine.clear_cache()
        
        # Check that cache is empty
        assert len(engine._content_hashes) == 0
        assert len(engine._normalized_content) == 0
    
    def test_error_handling(self):
        """Test error handling in deduplication."""
        engine = DeduplicationEngine()
        
        # Test with None input
        result = engine.deduplicate_documents(None)
        assert result.original_count == 0
        assert result.unique_count == 0
        
        # Test with empty dictionary
        result = engine.deduplicate_documents({})
        assert result.original_count == 0
        assert result.unique_count == 0
    
    def test_large_document_set(self):
        """Test performance with larger document set."""
        engine = DeduplicationEngine()
        
        # Create a larger set of documents with some duplicates
        documents = {}
        for i in range(100):
            if i % 10 == 0:
                # Every 10th document is a duplicate of the first in that group
                base_id = f'doc{i - (i % 10)}'
                if base_id in documents:
                    documents[f'doc{i}'] = documents[base_id]
                else:
                    documents[f'doc{i}'] = f'Unique content for document {i}'
            else:
                documents[f'doc{i}'] = f'Unique content for document {i}'
        
        result = engine.deduplicate_documents(documents)
        
        assert result.original_count == 100
        assert result.get_duplicate_count() > 0
        assert result.unique_count < result.original_count
    
    def test_configuration_options(self):
        """Test various configuration options."""
        config = {
            'exact_match_threshold': 1.0,
            'near_duplicate_threshold': 0.8,
            'fuzzy_match_threshold': 0.6,
            'normalize_whitespace': True,
            'case_sensitive': False,
            'ignore_punctuation': True,
            'min_content_length': 20,
            'use_hashing': True,
            'chunk_size': 500
        }
        
        engine = DeduplicationEngine(config)
        
        # Verify configuration is applied
        assert engine.exact_match_threshold == 1.0
        assert engine.near_duplicate_threshold == 0.8
        assert engine.fuzzy_match_threshold == 0.6
        assert engine.normalize_whitespace == True
        assert engine.case_sensitive == False
        assert engine.ignore_punctuation == True
        assert engine.min_content_length == 20
        assert engine.use_hashing == True
        assert engine.chunk_size == 500


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_document(self):
        """Test with single document."""
        engine = DeduplicationEngine()
        
        documents = {'doc1': 'Single document content.'}
        result = engine.deduplicate_documents(documents)
        
        assert result.original_count == 1
        assert result.unique_count == 1
        assert result.get_duplicate_count() == 0
        assert len(result.duplicate_groups) == 0
    
    def test_all_identical_documents(self):
        """Test with all documents being identical."""
        engine = DeduplicationEngine()
        
        content = 'All documents have this exact same content.'
        documents = {f'doc{i}': content for i in range(5)}
        
        result = engine.deduplicate_documents(documents)
        
        assert result.original_count == 5
        assert result.unique_count == 1
        assert result.get_duplicate_count() == 4
        assert len(result.duplicate_groups) == 1
    
    def test_unicode_content(self):
        """Test with Unicode content."""
        engine = DeduplicationEngine()
        
        documents = {
            'doc1': 'Hello 世界! This is a test with émojis 🌍.',
            'doc2': 'Hello 世界! This is a test with émojis 🌍.',  # Exact duplicate
            'doc3': 'Hello world! This is a test with emojis 🌍.'  # Similar
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should handle Unicode correctly
        assert result.get_duplicate_count() >= 1
    
    def test_very_long_documents(self):
        """Test with very long documents."""
        engine = DeduplicationEngine()
        
        # Create long documents
        base_content = 'This is a very long document. ' * 1000
        documents = {
            'doc1': base_content,
            'doc2': base_content,  # Exact duplicate
            'doc3': base_content + 'Additional content at the end.'  # Similar
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should handle long documents efficiently
        assert result.get_duplicate_count() >= 1
    
    def test_special_characters(self):
        """Test with special characters and symbols."""
        engine = DeduplicationEngine()
        
        documents = {
            'doc1': '!@#$%^&*()_+-=[]{}|;:,.<>?',
            'doc2': '!@#$%^&*()_+-=[]{}|;:,.<>?',  # Exact duplicate
            'doc3': '!@#$%^&*()_+-=[]{}|;:,.<>/',  # Very similar
        }
        
        result = engine.deduplicate_documents(documents)
        
        # Should handle special characters
        assert result.get_duplicate_count() >= 1