"""
Annotation and metadata extraction modules.

This package provides functionality for document annotation, categorization,
and metadata extraction.
"""

from .taxonomy import TaxonomyClassifier, CategoryResult, TaxonomyConfig
from .metadata import MetadataExtractor, ExtractedMetadata

__all__ = [
    'TaxonomyClassifier',
    'CategoryResult', 
    'TaxonomyConfig',
    'MetadataExtractor',
    'ExtractedMetadata'
]