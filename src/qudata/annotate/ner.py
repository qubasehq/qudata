"""
Named Entity Recognition module using spaCy.

This module provides entity recognition and tagging capabilities for extracting
names, places, organizations, and other entities from text content.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..models import Entity, ProcessingError, ErrorSeverity


@dataclass
class EntityConfig:
    """Configuration for entity recognition."""
    model_name: str = "en_core_web_sm"
    entity_types: List[str] = None
    confidence_threshold: float = 0.5
    max_entities_per_doc: int = 100
    custom_patterns: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            # Default entity types: Person, Organization, Location, Miscellaneous
            self.entity_types = ["PERSON", "ORG", "GPE", "LOC", "MISC"]
        
        if self.custom_patterns is None:
            self.custom_patterns = {}


class EntityRecognizer:
    """
    Entity recognition and tagging using spaCy NER.
    
    Extracts and tags names, places, organizations, and custom entities from text
    with configurable entity types and confidence thresholds.
    """
    
    def __init__(self, config: EntityConfig = None):
        """
        Initialize the entity recognizer.
        
        Args:
            config: Configuration for entity recognition
            
        Raises:
            ProcessingError: If spaCy is not available or model cannot be loaded
        """
        self.config = config or EntityConfig()
        self.logger = logging.getLogger(__name__)
        
        if not SPACY_AVAILABLE:
            raise ProcessingError(
                stage="ner",
                error_type="DependencyError",
                message="spaCy is not available. Please install spacy>=3.6.0",
                severity=ErrorSeverity.CRITICAL
            )
        
        self.nlp = None
        self._load_model()
        
        # Entity type mapping from spaCy to our standard types
        self.entity_type_mapping = {
            "PERSON": ["PERSON"],
            "ORG": ["ORG"],
            "GPE": ["GPE"],  # Geopolitical entity (countries, cities, states)
            "LOC": ["LOC", "FAC"],  # Locations and facilities
            "MISC": ["EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PRODUCT"],
            "DATE": ["DATE", "TIME"],
            "MONEY": ["MONEY"],
            "QUANTITY": ["QUANTITY", "PERCENT", "ORDINAL", "CARDINAL"]
        }
    
    def _load_model(self) -> None:
        """Load the spaCy model."""
        try:
            self.nlp = spacy.load(self.config.model_name)
            self.logger.info(f"Loaded spaCy model: {self.config.model_name}")
        except OSError:
            # Try to load a basic English model if the specified one is not available
            try:
                self.nlp = English()
                self.logger.warning(f"Could not load {self.config.model_name}, using basic English model")
            except Exception as e:
                raise ProcessingError(
                    stage="ner",
                    error_type="ModelLoadError",
                    message=f"Failed to load spaCy model: {str(e)}",
                    severity=ErrorSeverity.CRITICAL
                )
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of Entity objects with detected entities
            
        Raises:
            ProcessingError: If entity extraction fails
        """
        if not text or not text.strip():
            return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Map spaCy entity type to our standard types
                entity_type = self._map_entity_type(ent.label_)
                
                # Filter by configured entity types
                if entity_type not in self.config.entity_types:
                    continue
                
                # Create entity object
                entity = Entity(
                    text=ent.text.strip(),
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores by default
                )
                
                # Apply confidence threshold (for future compatibility)
                if entity.confidence >= self.config.confidence_threshold:
                    entities.append(entity)
                
                # Limit number of entities per document
                if len(entities) >= self.config.max_entities_per_doc:
                    break
            
            # Apply custom pattern matching if configured
            if self.config.custom_patterns:
                custom_entities = self._extract_custom_entities(text)
                # Apply confidence threshold to custom entities as well
                filtered_custom_entities = [
                    entity for entity in custom_entities 
                    if entity.confidence >= self.config.confidence_threshold
                ]
                entities.extend(filtered_custom_entities)
            
            # Remove duplicates and sort by position
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.start)
            
            self.logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            raise ProcessingError(
                stage="ner",
                error_type="ExtractionError",
                message=f"Failed to extract entities: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """
        Map spaCy entity label to our standard entity type.
        
        Args:
            spacy_label: spaCy entity label (e.g., "PERSON", "ORG")
            
        Returns:
            Mapped entity type
        """
        for our_type, spacy_types in self.entity_type_mapping.items():
            if spacy_label in spacy_types:
                return our_type
        
        # Default to MISC for unknown types
        return "MISC"
    
    def _extract_custom_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using custom patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of entities found using custom patterns
        """
        import re
        
        entities = []
        
        for entity_type, patterns in self.config.custom_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity = Entity(
                            text=match.group().strip(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8  # Lower confidence for pattern-based matches
                        )
                        entities.append(entity)
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities based on text and position overlap.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if self._entities_overlap(entity, existing):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities overlap in position.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities overlap, False otherwise
        """
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """
        Get statistics about extracted entities.
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            Dictionary with entity statistics
        """
        if not entities:
            return {
                "total_entities": 0,
                "entity_types": {},
                "unique_entities": 0,
                "avg_confidence": 0.0
            }
        
        entity_types = {}
        unique_texts = set()
        total_confidence = 0.0
        
        for entity in entities:
            # Count by type
            entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
            
            # Track unique entity texts
            unique_texts.add(entity.text.lower())
            
            # Sum confidence scores
            total_confidence += entity.confidence
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "unique_entities": len(unique_texts),
            "avg_confidence": total_confidence / len(entities) if entities else 0.0
        }
    
    def filter_entities_by_type(self, entities: List[Entity], 
                               entity_types: List[str]) -> List[Entity]:
        """
        Filter entities by type.
        
        Args:
            entities: List of entities to filter
            entity_types: List of entity types to keep
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.label in entity_types]
    
    def filter_entities_by_confidence(self, entities: List[Entity], 
                                    min_confidence: float) -> List[Entity]:
        """
        Filter entities by minimum confidence threshold.
        
        Args:
            entities: List of entities to filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.confidence >= min_confidence]
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types.
        
        Returns:
            List of supported entity type names
        """
        return list(self.entity_type_mapping.keys())
    
    def is_model_available(self) -> bool:
        """
        Check if the NER model is available and loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.nlp is not None


def create_entity_recognizer(config_dict: Dict[str, Any] = None) -> EntityRecognizer:
    """
    Factory function to create an EntityRecognizer from configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured EntityRecognizer instance
    """
    if config_dict is None:
        config_dict = {}
    
    config = EntityConfig(
        model_name=config_dict.get("model_name", "en_core_web_sm"),
        entity_types=config_dict.get("entity_types"),
        confidence_threshold=config_dict.get("confidence_threshold", 0.5),
        max_entities_per_doc=config_dict.get("max_entities_per_doc", 100),
        custom_patterns=config_dict.get("custom_patterns")
    )
    
    return EntityRecognizer(config)
