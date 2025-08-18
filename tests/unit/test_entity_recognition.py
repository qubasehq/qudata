"""
Unit tests for entity recognition and tagging functionality.

Tests the EntityRecognizer class for extracting and tagging names, places,
organizations, and other entities from text content.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from qudata.annotate.ner import (
    EntityRecognizer, 
    EntityConfig, 
    create_entity_recognizer,
    SPACY_AVAILABLE
)
from qudata.models import Entity, ProcessingError, ErrorSeverity


class TestEntityConfig:
    """Test EntityConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EntityConfig()
        
        assert config.model_name == "en_core_web_sm"
        assert config.entity_types == ["PERSON", "ORG", "GPE", "LOC", "MISC"]
        assert config.confidence_threshold == 0.5
        assert config.max_entities_per_doc == 100
        assert config.custom_patterns == {}
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_types = ["PERSON", "ORG"]
        custom_patterns = {"EMAIL": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"]}
        
        config = EntityConfig(
            model_name="en_core_web_md",
            entity_types=custom_types,
            confidence_threshold=0.8,
            max_entities_per_doc=50,
            custom_patterns=custom_patterns
        )
        
        assert config.model_name == "en_core_web_md"
        assert config.entity_types == custom_types
        assert config.confidence_threshold == 0.8
        assert config.max_entities_per_doc == 50
        assert config.custom_patterns == custom_patterns


class TestEntityRecognizer:
    """Test EntityRecognizer class."""
    
    @pytest.fixture
    def mock_spacy_doc(self):
        """Create a mock spaCy document with entities."""
        # Mock entity
        mock_ent = Mock()
        mock_ent.text = "John Doe"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 8
        
        # Mock document
        mock_doc = Mock()
        mock_doc.ents = [mock_ent]
        
        return mock_doc
    
    @pytest.fixture
    def mock_nlp(self, mock_spacy_doc):
        """Create a mock spaCy nlp object."""
        mock_nlp = Mock()
        mock_nlp.return_value = mock_spacy_doc
        return mock_nlp
    
    @pytest.fixture
    def recognizer_with_mock(self, mock_nlp):
        """Create EntityRecognizer with mocked spaCy."""
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                return EntityRecognizer()
    
    def test_init_without_spacy(self):
        """Test initialization when spaCy is not available."""
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', False):
            with pytest.raises(ProcessingError) as exc_info:
                EntityRecognizer()
            
            assert exc_info.value.stage == "ner"
            assert exc_info.value.error_type == "DependencyError"
            assert "spaCy is not available" in exc_info.value.message
    
    def test_init_with_invalid_model(self):
        """Test initialization with invalid spaCy model."""
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', side_effect=OSError("Model not found")):
                with patch('qudata.annotate.ner.English', side_effect=Exception("Failed")):
                    with pytest.raises(ProcessingError) as exc_info:
                        EntityRecognizer()
                    
                    assert exc_info.value.stage == "ner"
                    assert exc_info.value.error_type == "ModelLoadError"
    
    def test_init_with_fallback_model(self):
        """Test initialization with fallback to basic English model."""
        mock_english = Mock()
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', side_effect=OSError("Model not found")):
                with patch('qudata.annotate.ner.English', return_value=mock_english):
                    recognizer = EntityRecognizer()
                    assert recognizer.nlp == mock_english
    
    def test_extract_entities_basic(self, recognizer_with_mock):
        """Test basic entity extraction."""
        text = "John Doe works at Microsoft."
        entities = recognizer_with_mock.extract_entities(text)
        
        assert len(entities) == 1
        assert entities[0].text == "John Doe"
        assert entities[0].label == "PERSON"
        assert entities[0].start == 0
        assert entities[0].end == 8
        assert entities[0].confidence == 1.0
    
    def test_extract_entities_empty_text(self, recognizer_with_mock):
        """Test entity extraction with empty text."""
        assert recognizer_with_mock.extract_entities("") == []
        assert recognizer_with_mock.extract_entities("   ") == []
        assert recognizer_with_mock.extract_entities(None) == []
    
    def test_extract_entities_with_filtering(self, mock_nlp):
        """Test entity extraction with type filtering."""
        # Mock entities of different types
        person_ent = Mock()
        person_ent.text = "John Doe"
        person_ent.label_ = "PERSON"
        person_ent.start_char = 0
        person_ent.end_char = 8
        
        org_ent = Mock()
        org_ent.text = "Microsoft"
        org_ent.label_ = "ORG"
        org_ent.start_char = 18
        org_ent.end_char = 27
        
        mock_doc = Mock()
        mock_doc.ents = [person_ent, org_ent]
        mock_nlp.return_value = mock_doc
        
        # Configure to only extract PERSON entities
        config = EntityConfig(entity_types=["PERSON"])
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer(config)
                entities = recognizer.extract_entities("John Doe works at Microsoft.")
                
                assert len(entities) == 1
                assert entities[0].label == "PERSON"
    
    def test_extract_entities_with_max_limit(self, mock_nlp):
        """Test entity extraction with maximum entity limit."""
        # Create multiple mock entities
        entities = []
        for i in range(5):
            ent = Mock()
            ent.text = f"Entity{i}"
            ent.label_ = "PERSON"
            ent.start_char = i * 10
            ent.end_char = i * 10 + 7
            entities.append(ent)
        
        mock_doc = Mock()
        mock_doc.ents = entities
        mock_nlp.return_value = mock_doc
        
        # Configure with max 3 entities
        config = EntityConfig(max_entities_per_doc=3)
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer(config)
                extracted = recognizer.extract_entities("Multiple entities text")
                
                assert len(extracted) == 3
    
    def test_extract_entities_with_custom_patterns(self, mock_nlp):
        """Test entity extraction with custom patterns."""
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        
        config = EntityConfig(
            custom_patterns={
                "EMAIL": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"]
            }
        )
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer(config)
                entities = recognizer.extract_entities("Contact me at john.doe@example.com")
                
                assert len(entities) == 1
                assert entities[0].text == "john.doe@example.com"
                assert entities[0].label == "EMAIL"
                assert entities[0].confidence == 0.8
    
    def test_extract_entities_error_handling(self, mock_nlp):
        """Test error handling during entity extraction."""
        mock_nlp.side_effect = Exception("Processing failed")
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer()
                
                with pytest.raises(ProcessingError) as exc_info:
                    recognizer.extract_entities("Some text")
                
                assert exc_info.value.stage == "ner"
                assert exc_info.value.error_type == "ExtractionError"
    
    def test_map_entity_type(self, recognizer_with_mock):
        """Test entity type mapping."""
        recognizer = recognizer_with_mock
        
        assert recognizer._map_entity_type("PERSON") == "PERSON"
        assert recognizer._map_entity_type("ORG") == "ORG"
        assert recognizer._map_entity_type("GPE") == "GPE"
        assert recognizer._map_entity_type("FAC") == "LOC"
        assert recognizer._map_entity_type("EVENT") == "MISC"
        assert recognizer._map_entity_type("UNKNOWN_TYPE") == "MISC"
    
    def test_deduplicate_entities(self, recognizer_with_mock):
        """Test entity deduplication."""
        recognizer = recognizer_with_mock
        
        # Create overlapping entities
        entity1 = Entity(text="John", label="PERSON", start=0, end=4, confidence=0.8)
        entity2 = Entity(text="John Doe", label="PERSON", start=0, end=8, confidence=0.9)
        entity3 = Entity(text="Microsoft", label="ORG", start=20, end=29, confidence=0.7)
        
        entities = [entity1, entity2, entity3]
        deduplicated = recognizer._deduplicate_entities(entities)
        
        # Should keep the higher confidence overlapping entity
        assert len(deduplicated) == 2
        assert any(e.text == "John Doe" and e.confidence == 0.9 for e in deduplicated)
        assert any(e.text == "Microsoft" for e in deduplicated)
    
    def test_entities_overlap(self, recognizer_with_mock):
        """Test entity overlap detection."""
        recognizer = recognizer_with_mock
        
        entity1 = Entity(text="John", label="PERSON", start=0, end=4, confidence=0.8)
        entity2 = Entity(text="John Doe", label="PERSON", start=0, end=8, confidence=0.9)
        entity3 = Entity(text="Microsoft", label="ORG", start=20, end=29, confidence=0.7)
        
        assert recognizer._entities_overlap(entity1, entity2) == True
        assert recognizer._entities_overlap(entity1, entity3) == False
        assert recognizer._entities_overlap(entity2, entity3) == False
    
    def test_get_entity_statistics(self, recognizer_with_mock):
        """Test entity statistics generation."""
        recognizer = recognizer_with_mock
        
        entities = [
            Entity(text="John Doe", label="PERSON", start=0, end=8, confidence=0.9),
            Entity(text="Jane Smith", label="PERSON", start=10, end=20, confidence=0.8),
            Entity(text="Microsoft", label="ORG", start=30, end=39, confidence=0.7),
        ]
        
        stats = recognizer.get_entity_statistics(entities)
        
        assert stats["total_entities"] == 3
        assert stats["entity_types"]["PERSON"] == 2
        assert stats["entity_types"]["ORG"] == 1
        assert stats["unique_entities"] == 3
        assert stats["avg_confidence"] == pytest.approx(0.8, rel=1e-2)
    
    def test_get_entity_statistics_empty(self, recognizer_with_mock):
        """Test entity statistics with empty list."""
        recognizer = recognizer_with_mock
        
        stats = recognizer.get_entity_statistics([])
        
        assert stats["total_entities"] == 0
        assert stats["entity_types"] == {}
        assert stats["unique_entities"] == 0
        assert stats["avg_confidence"] == 0.0
    
    def test_filter_entities_by_type(self, recognizer_with_mock):
        """Test filtering entities by type."""
        recognizer = recognizer_with_mock
        
        entities = [
            Entity(text="John Doe", label="PERSON", start=0, end=8, confidence=0.9),
            Entity(text="Microsoft", label="ORG", start=10, end=19, confidence=0.8),
            Entity(text="New York", label="GPE", start=20, end=28, confidence=0.7),
        ]
        
        person_entities = recognizer.filter_entities_by_type(entities, ["PERSON"])
        assert len(person_entities) == 1
        assert person_entities[0].label == "PERSON"
        
        location_entities = recognizer.filter_entities_by_type(entities, ["GPE", "LOC"])
        assert len(location_entities) == 1
        assert location_entities[0].label == "GPE"
    
    def test_filter_entities_by_confidence(self, recognizer_with_mock):
        """Test filtering entities by confidence threshold."""
        recognizer = recognizer_with_mock
        
        entities = [
            Entity(text="John Doe", label="PERSON", start=0, end=8, confidence=0.9),
            Entity(text="Microsoft", label="ORG", start=10, end=19, confidence=0.6),
            Entity(text="New York", label="GPE", start=20, end=28, confidence=0.4),
        ]
        
        high_confidence = recognizer.filter_entities_by_confidence(entities, 0.7)
        assert len(high_confidence) == 1
        assert high_confidence[0].confidence == 0.9
        
        medium_confidence = recognizer.filter_entities_by_confidence(entities, 0.5)
        assert len(medium_confidence) == 2
    
    def test_get_supported_entity_types(self, recognizer_with_mock):
        """Test getting supported entity types."""
        recognizer = recognizer_with_mock
        
        supported_types = recognizer.get_supported_entity_types()
        expected_types = ["PERSON", "ORG", "GPE", "LOC", "MISC", "DATE", "MONEY", "QUANTITY"]
        
        for entity_type in expected_types:
            assert entity_type in supported_types
    
    def test_is_model_available(self, recognizer_with_mock):
        """Test model availability check."""
        recognizer = recognizer_with_mock
        
        assert recognizer.is_model_available() == True
        
        recognizer.nlp = None
        assert recognizer.is_model_available() == False


class TestCreateEntityRecognizer:
    """Test factory function for creating EntityRecognizer."""
    
    def test_create_with_default_config(self):
        """Test creating recognizer with default configuration."""
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load'):
                recognizer = create_entity_recognizer()
                
                assert recognizer.config.model_name == "en_core_web_sm"
                assert recognizer.config.confidence_threshold == 0.5
    
    def test_create_with_custom_config(self):
        """Test creating recognizer with custom configuration."""
        config_dict = {
            "model_name": "en_core_web_md",
            "entity_types": ["PERSON", "ORG"],
            "confidence_threshold": 0.8,
            "max_entities_per_doc": 50,
            "custom_patterns": {"EMAIL": [r"\S+@\S+"]}
        }
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            with patch('qudata.annotate.ner.spacy.load'):
                recognizer = create_entity_recognizer(config_dict)
                
                assert recognizer.config.model_name == "en_core_web_md"
                assert recognizer.config.entity_types == ["PERSON", "ORG"]
                assert recognizer.config.confidence_threshold == 0.8
                assert recognizer.config.max_entities_per_doc == 50
                assert "EMAIL" in recognizer.config.custom_patterns


class TestEntityRecognitionIntegration:
    """Integration tests for entity recognition."""
    
    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_real_entity_extraction(self):
        """Test entity extraction with real spaCy model (if available)."""
        try:
            recognizer = EntityRecognizer()
            
            text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            entities = recognizer.extract_entities(text)
            
            # Since we're using basic English model, we might not get entities
            # This test will pass if no exception is raised
            assert isinstance(entities, list)
            
        except ProcessingError:
            # Skip if model is not available
            pytest.skip("spaCy model not available")
    
    def test_custom_pattern_extraction(self):
        """Test extraction with custom patterns."""
        config = EntityConfig(
            custom_patterns={
                "EMAIL": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                "PHONE": [r"\b\d{3}-\d{3}-\d{4}\b"]
            }
        )
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer(config)
                
                text = "Contact John at john.doe@example.com or call 555-123-4567"
                entities = recognizer.extract_entities(text)
                
                # Should extract email and phone
                assert len(entities) == 2
                
                email_entity = next((e for e in entities if e.label == "EMAIL"), None)
                phone_entity = next((e for e in entities if e.label == "PHONE"), None)
                
                assert email_entity is not None
                assert email_entity.text == "john.doe@example.com"
                
                assert phone_entity is not None
                assert phone_entity.text == "555-123-4567"
    
    def test_confidence_threshold_filtering(self):
        """Test filtering by confidence threshold."""
        config = EntityConfig(confidence_threshold=0.9)  # Higher threshold
        
        with patch('qudata.annotate.ner.SPACY_AVAILABLE', True):
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            
            with patch('qudata.annotate.ner.spacy.load', return_value=mock_nlp):
                recognizer = EntityRecognizer(config)
                
                # Add custom pattern with lower confidence (0.8)
                recognizer.config.custom_patterns = {
                    "EMAIL": [r"\S+@\S+"]
                }
                
                text = "Email: test@example.com"
                entities = recognizer.extract_entities(text)
                
                # Should be filtered out due to low confidence (0.8 < 0.9)
                assert len(entities) == 0


if __name__ == "__main__":
    pytest.main([__file__])