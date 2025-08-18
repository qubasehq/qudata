"""
Comprehensive text cleaning and processing module.

This module provides comprehensive text cleaning capabilities including:
- Unicode normalization and whitespace cleanup
- OCR error correction
- Encoding detection and UTF-8 conversion
- Deduplication and duplicate detection
- Boilerplate content removal
- HTML cleaning and emoji removal
- Sentence segmentation and text normalization
- Stopword removal with configurable lists
- Tokenization preview for LLM compatibility
- Integrated cleaning pipelines
"""

from .normalize import (
    TextNormalizer,
    OCRCorrector,
    EncodingDetector,
    NormalizationResult,
    normalize_text_pipeline
)

from .dedupe import (
    DeduplicationEngine,
    DuplicateGroup,
    DeduplicationResult
)

from .boilerplate import (
    BoilerplateRemover,
    BoilerplatePattern,
    BoilerplateRemovalResult
)

from .html_cleaner import (
    HTMLCleaner,
    HTMLCleaningResult,
    clean_html_content
)

from .segment import (
    SentenceSegmenter,
    SegmentationResult,
    segment_text_simple
)

from .stopwords import (
    StopwordRemover,
    StopwordRemovalResult,
    remove_stopwords_simple,
    load_stopwords_from_file
)

from .tokenization import (
    TokenizationPreview,
    TokenizationResult,
    BatchTokenizationStats,
    quick_token_count,
    check_context_fit
)

from .pipeline import (
    ComprehensiveCleaningPipeline,
    CleaningResult,
    BatchCleaningResult
)

__all__ = [
    # Normalization
    'TextNormalizer',
    'OCRCorrector', 
    'EncodingDetector',
    'NormalizationResult',
    'normalize_text_pipeline',
    
    # Deduplication
    'DeduplicationEngine',
    'DuplicateGroup',
    'DeduplicationResult',
    
    # Boilerplate removal
    'BoilerplateRemover',
    'BoilerplatePattern',
    'BoilerplateRemovalResult',
    
    # HTML cleaning
    'HTMLCleaner',
    'HTMLCleaningResult',
    'clean_html_content',
    
    # Text segmentation
    'SentenceSegmenter',
    'SegmentationResult',
    'segment_text_simple',
    
    # Stopword removal
    'StopwordRemover',
    'StopwordRemovalResult',
    'remove_stopwords_simple',
    'load_stopwords_from_file',
    
    # Tokenization preview
    'TokenizationPreview',
    'TokenizationResult',
    'BatchTokenizationStats',
    'quick_token_count',
    'check_context_fit',
    
    # Integrated pipeline
    'ComprehensiveCleaningPipeline',
    'CleaningResult',
    'BatchCleaningResult'
]