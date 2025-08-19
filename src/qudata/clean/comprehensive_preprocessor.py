"""
Comprehensive preprocessing system for high-quality LLM training datasets.

This module implements comprehensive preprocessing across all supported formats:
- Text Documents (PDF, DOCX, ODT, RTF, TXT, MD)
- Web Content (HTML, XML)
- Structured Data (CSV, JSON, JSONL, YAML)
- Images with OCR (PNG, JPG, TIFF)
- Archives (ZIP, TAR, GZ)
- Code & Jupyter Notebooks
- Cross-format validation and deduplication

The system enforces strict data quality standards, removes noise and duplicates,
normalizes formatting, and ensures consistent UTF-8 output for LLM training.
"""

import re
import json
import yaml
import zipfile
import tarfile
import gzip
import csv
import io
import hashlib
import unicodedata
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter

# Import existing cleaning components
from .normalize import TextNormalizer, OCRCorrector, EncodingDetector, NormalizationResult
from .dedupe import DeduplicationEngine, DeduplicationResult
from .boilerplate import BoilerplateRemover, BoilerplateRemovalResult
from .language import LanguageDetector, LanguageResult
from .html_cleaner import HTMLCleaner, HTMLCleaningResult
from ..models import ProcessingError, ErrorSeverity, Document


@dataclass
class PreprocessingResult:
    """Result of comprehensive preprocessing operation."""
    original_content: str
    processed_content: str
    content_type: str
    
    # Processing stages applied
    stages_applied: List[str] = field(default_factory=list)
    
    # Quality metrics
    quality_score: float = 0.0
    content_hash: str = ""
    
    # Removal statistics
    removed_elements: Dict[str, int] = field(default_factory=dict)
    
    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    original_size: int = 0
    processed_size: int = 0
    
    def get_compression_ratio(self) -> float:
        """Get content compression ratio."""
        if self.original_size == 0:
            return 0.0
        return (self.original_size - self.processed_size) / self.original_size
    
    def get_content_hash(self) -> str:
        """Generate content hash for deduplication."""
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                self.processed_content.encode('utf-8')
            ).hexdigest()
        return self.content_hash


@dataclass
class BatchPreprocessingResult:
    """Result of batch preprocessing operation."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    
    # Individual results
    document_results: Dict[str, PreprocessingResult] = field(default_factory=dict)
    
    # Cross-document deduplication
    duplicate_groups: List[List[str]] = field(default_factory=list)
    removed_duplicates: Set[str] = field(default_factory=set)
    
    # Aggregate statistics
    total_original_size: int = 0
    total_processed_size: int = 0
    average_quality_score: float = 0.0
    
    # Content type distribution
    content_type_stats: Dict[str, int] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """Get processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents


class ComprehensivePreprocessor:
    """
    Comprehensive preprocessing system for high-quality LLM training datasets.
    
    Implements format-specific preprocessing for all supported data types with
    strict quality standards and cross-format validation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize comprehensive preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config or {}
        
        # Initialize core components
        self._initialize_components()
        
        # Format-specific settings
        self.text_formats = {'pdf', 'docx', 'odt', 'rtf', 'txt', 'md'}
        self.web_formats = {'html', 'xml'}
        self.structured_formats = {'csv', 'json', 'jsonl', 'yaml'}
        self.image_formats = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}
        self.archive_formats = {'zip', 'tar', 'gz', 'tar.gz'}
        self.code_formats = {'py', 'js', 'java', 'cpp', 'c', 'ipynb'}
        
        # Quality thresholds
        quality_config = self.config.get('quality', {})
        self.min_content_length = quality_config.get('min_content_length', 50)
        self.max_content_length = quality_config.get('max_content_length', 1000000)
        self.min_quality_score = quality_config.get('min_quality_score', 0.6)
        self.max_noise_ratio = quality_config.get('max_noise_ratio', 0.3)
        
        # Language filtering
        lang_config = self.config.get('language', {})
        self.target_languages = set(lang_config.get('target_languages', ['en']))
        self.min_language_confidence = lang_config.get('min_confidence', 0.8)
        
        # Cross-format deduplication settings
        dedup_config = self.config.get('deduplication', {})
        self.similarity_threshold = dedup_config.get('similarity_threshold', 0.85)
        self.enable_cross_format_dedup = dedup_config.get('cross_format', True)
        
        # Preprocessing patterns
        self._initialize_patterns()
    
    def _initialize_components(self):
        """Initialize all preprocessing components."""
        # Core text processing
        self.text_normalizer = TextNormalizer(self.config.get('normalization', {}))
        self.ocr_corrector = OCRCorrector(self.config.get('ocr_correction', {}))
        self.encoding_detector = EncodingDetector(self.config.get('encoding', {}))
        
        # Content cleaning
        self.html_cleaner = HTMLCleaner()
        self.boilerplate_remover = BoilerplateRemover(self.config)
        
        # Language and deduplication
        self.language_detector = LanguageDetector(self.config.get('language_detection', {}))
        self.deduplication_engine = DeduplicationEngine(self.config.get('deduplication', {}))
    
    def _initialize_patterns(self):
        """Initialize preprocessing patterns for different content types."""
        
        # Text document patterns
        self.text_noise_patterns = [
            # Headers and footers
            re.compile(r'^Page \d+ of \d+$', re.MULTILINE),
            re.compile(r'^\d+\s*$', re.MULTILINE),  # Page numbers
            re.compile(r'^©.*?\d{4}.*$', re.MULTILINE),  # Copyright lines
            re.compile(r'^Confidential.*$', re.MULTILINE),
            re.compile(r'^Draft.*$', re.MULTILINE),
            re.compile(r'^Downloaded from.*$', re.MULTILINE),
            re.compile(r'^Last updated:.*$', re.MULTILINE),
            re.compile(r'^Generated on.*$', re.MULTILINE),
            
            # Watermarks and stamps
            re.compile(r'\[WATERMARK\]', re.IGNORECASE),
            re.compile(r'\[STAMP\]', re.IGNORECASE),
            re.compile(r'CONFIDENTIAL', re.IGNORECASE),
            re.compile(r'DRAFT', re.IGNORECASE),
        ]
        
        # Web content patterns
        self.web_noise_patterns = [
            # Navigation and UI elements
            re.compile(r'Skip to main content', re.IGNORECASE),
            re.compile(r'Menu', re.IGNORECASE),
            re.compile(r'Navigation', re.IGNORECASE),
            re.compile(r'Breadcrumb', re.IGNORECASE),
            re.compile(r'Related articles?', re.IGNORECASE),
            re.compile(r'You might also like', re.IGNORECASE),
            re.compile(r'Share this article', re.IGNORECASE),
            re.compile(r'Subscribe to.*newsletter', re.IGNORECASE),
            re.compile(r'Follow us on', re.IGNORECASE),
            
            # Timestamps and metadata
            re.compile(r'Published on.*\d{4}', re.IGNORECASE),
            re.compile(r'Last modified.*\d{4}', re.IGNORECASE),
            re.compile(r'\d+ min read', re.IGNORECASE),
            re.compile(r'Reading time:.*minutes?', re.IGNORECASE),
            
            # Ads and tracking
            re.compile(r'Advertisement', re.IGNORECASE),
            re.compile(r'Sponsored content', re.IGNORECASE),
            re.compile(r'Cookie policy', re.IGNORECASE),
            re.compile(r'Privacy policy', re.IGNORECASE),
        ]
        
        # OCR error patterns
        self.ocr_error_patterns = [
            # Character substitutions
            (re.compile(r'\bl\b'), '1'),  # Standalone l to 1
            (re.compile(r'\bO\b'), '0'),  # Standalone O to 0
            (re.compile(r'rn'), 'm'),     # rn to m
            (re.compile(r'cl'), 'd'),     # cl to d
            (re.compile(r'ii'), 'n'),     # ii to n
            (re.compile(r'vv'), 'w'),     # vv to w
            
            # Spacing issues
            (re.compile(r'([a-z])([A-Z])'), r'\1 \2'),  # Missing space between words
            (re.compile(r'(\w)([.!?])([A-Z])'), r'\1\2 \3'),  # Missing space after punctuation
        ]
        
        # Code cleaning patterns
        self.code_noise_patterns = [
            # Jupyter notebook outputs
            re.compile(r'Out\[\d+\]:.*?(?=In\[|\Z)', re.DOTALL),
            re.compile(r'<matplotlib\..*?>', re.IGNORECASE),
            re.compile(r'<Figure size.*?>', re.IGNORECASE),
            
            # Debug and logging
            re.compile(r'print\s*\(.*?debug.*?\)', re.IGNORECASE),
            re.compile(r'console\.log\s*\(.*?\)', re.IGNORECASE),
            re.compile(r'System\.out\.println\s*\(.*?\)', re.IGNORECASE),
            
            # Secrets patterns (to be removed)
            re.compile(r'api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            re.compile(r'token\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            re.compile(r'secret\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
        ]
        
        # Archive system files to remove
        self.system_files = {
            '.DS_Store', 'Thumbs.db', 'desktop.ini', '.gitignore',
            '__pycache__', '.pyc', '.tmp', '.temp', '~$'
        }
    
    def preprocess_content(self, content: str, content_type: str, 
                          document_id: str = None) -> PreprocessingResult:
        """
        Preprocess content based on its type with comprehensive cleaning.
        
        Args:
            content: Raw content to preprocess
            content_type: Type of content (pdf, html, json, etc.)
            document_id: Optional document identifier
            
        Returns:
            PreprocessingResult with processed content and metadata
        """
        import time
        start_time = time.time()
        
        result = PreprocessingResult(
            original_content=content,
            processed_content=content,
            content_type=content_type.lower(),
            original_size=len(content) if content else 0
        )
        
        if not content or not isinstance(content, str):
            result.validation_errors.append("Empty or invalid content")
            result.validation_passed = False
            return result
        
        try:
            # Route to format-specific preprocessing
            if result.content_type in self.text_formats:
                result = self._preprocess_text_document(content, result)
            elif result.content_type in self.web_formats:
                result = self._preprocess_web_content(content, result)
            elif result.content_type in self.structured_formats:
                result = self._preprocess_structured_data(content, result)
            elif result.content_type in self.image_formats:
                result = self._preprocess_image_content(content, result)
            elif result.content_type in self.code_formats:
                result = self._preprocess_code_content(content, result)
            else:
                # Default text processing
                result = self._preprocess_text_document(content, result)
            
            # Apply cross-format global validation
            result = self._apply_global_validation(result)
            
            # Calculate final metrics
            result.processed_size = len(result.processed_content)
            result.processing_time = time.time() - start_time
            result.quality_score = self._calculate_quality_score(result)
            
        except Exception as e:
            result.validation_errors.append(f"Processing failed: {str(e)}")
            result.validation_passed = False
            result.quality_score = 0.0
        
        return result
    
    def _preprocess_text_document(self, content: str, result: PreprocessingResult) -> PreprocessingResult:
        """Preprocess text documents (PDF, DOCX, ODT, RTF, TXT, MD)."""
        
        # Stage 1: Strip headers, footers, page numbers, and watermarks
        original_content = content
        for pattern in self.text_noise_patterns:
            content = pattern.sub('', content)
        
        if content != original_content:
            result.stages_applied.append("header_footer_removal")
            result.removed_elements["headers_footers"] = len(original_content) - len(content)
        
        # Stage 2: Remove auto-generated text and boilerplate
        boilerplate_result = self.boilerplate_remover.remove_boilerplate(content)
        content = boilerplate_result.cleaned_text
        if boilerplate_result.removed_patterns:
            result.stages_applied.append("boilerplate_removal")
            result.removed_elements["boilerplate"] = len(boilerplate_result.removed_patterns)
        
        # Stage 3: Fix OCR errors and merge broken words/sentences
        ocr_result = self.ocr_corrector.correct_ocr_errors(content)
        content = ocr_result.normalized_text
        if ocr_result.corrections_applied:
            result.stages_applied.append("ocr_correction")
            result.removed_elements["ocr_errors"] = len(ocr_result.corrections_applied)
        
        # Stage 4: Remove noisy spaces, multiple blank lines, irregular indentation
        content = self._clean_whitespace_and_formatting(content)
        result.stages_applied.append("whitespace_normalization")
        
        # Stage 5: Deduplicate repeated paragraphs
        content = self._remove_repeated_paragraphs(content)
        result.stages_applied.append("paragraph_deduplication")
        
        # Stage 6: Clean metadata, revision history, junk characters
        content = self._clean_metadata_and_junk(content)
        result.stages_applied.append("metadata_cleaning")
        
        # Stage 7: Normalize lists, tables, equations, and special symbols
        content = self._normalize_special_elements(content)
        result.stages_applied.append("special_elements_normalization")
        
        result.processed_content = content
        return result
    
    def _preprocess_web_content(self, content: str, result: PreprocessingResult) -> PreprocessingResult:
        """Preprocess web content (HTML, XML)."""
        
        # Stage 1: Clean HTML and remove web-specific noise
        html_result = self.html_cleaner.clean_html(content)
        content = html_result.cleaned_text
        result.stages_applied.append("html_cleaning")
        
        # Stage 2: Remove ads, navigation, sidebars, menus, footers
        original_content = content
        for pattern in self.web_noise_patterns:
            content = pattern.sub('', content)
        
        if content != original_content:
            result.stages_applied.append("web_noise_removal")
            result.removed_elements["web_noise"] = len(original_content) - len(content)
        
        # Stage 3: Keep only main article/content body (already done by HTMLCleaner)
        # Stage 4: Resolve relative links and expand inline references
        content = self._resolve_web_references(content)
        result.stages_applied.append("reference_resolution")
        
        # Stage 5: Drop auto-generated content
        content = self._remove_auto_generated_web_content(content)
        result.stages_applied.append("auto_content_removal")
        
        # Stage 6: Clean timestamps, copyright marks, site banners
        content = self._clean_web_metadata(content)
        result.stages_applied.append("web_metadata_cleaning")
        
        # Apply general text cleaning
        content = self._clean_whitespace_and_formatting(content)
        result.stages_applied.append("whitespace_normalization")
        
        result.processed_content = content
        return result
    
    def _preprocess_structured_data(self, content: str, result: PreprocessingResult) -> PreprocessingResult:
        """Preprocess structured data (CSV, JSON, JSONL, YAML)."""
        
        try:
            if result.content_type == 'json':
                data = json.loads(content)
                cleaned_data = self._clean_json_data(data)
                content = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
                
            elif result.content_type == 'jsonl':
                lines = content.strip().split('\n')
                cleaned_lines = []
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        cleaned_data = self._clean_json_data(data)
                        cleaned_lines.append(json.dumps(cleaned_data, ensure_ascii=False))
                content = '\n'.join(cleaned_lines)
                
            elif result.content_type == 'yaml':
                data = yaml.safe_load(content)
                cleaned_data = self._clean_yaml_data(data)
                content = yaml.dump(cleaned_data, default_flow_style=False, allow_unicode=True)
                
            elif result.content_type == 'csv':
                content = self._clean_csv_data(content)
            
            result.stages_applied.append("structured_data_cleaning")
            
        except Exception as e:
            result.validation_errors.append(f"Structured data parsing failed: {str(e)}")
            # Fall back to text processing
            content = self._preprocess_text_document(content, result).processed_content
        
        result.processed_content = content
        return result
    
    def _preprocess_image_content(self, content: str, result: PreprocessingResult) -> PreprocessingResult:
        """Preprocess OCR content from images."""
        
        # Stage 1: Apply OCR-specific corrections
        ocr_result = self.ocr_corrector.correct_ocr_errors(content)
        content = ocr_result.normalized_text
        if ocr_result.corrections_applied:
            result.stages_applied.append("ocr_correction")
            result.removed_elements["ocr_errors"] = len(ocr_result.corrections_applied)
        
        # Stage 2: Remove watermarks, logos, decorative captions
        content = self._remove_image_artifacts(content)
        result.stages_applied.append("image_artifact_removal")
        
        # Stage 3: Remove noisy borders, page marks, non-text artifacts
        content = self._clean_image_noise(content)
        result.stages_applied.append("image_noise_cleaning")
        
        # Stage 4: Ensure proper text block segmentation
        content = self._segment_text_blocks(content)
        result.stages_applied.append("text_segmentation")
        
        # Apply general text cleaning
        content = self._clean_whitespace_and_formatting(content)
        result.stages_applied.append("whitespace_normalization")
        
        result.processed_content = content
        return result
    
    def _preprocess_code_content(self, content: str, result: PreprocessingResult) -> PreprocessingResult:
        """Preprocess code and Jupyter notebooks."""
        
        # Stage 1: Remove execution outputs, debug logs, binaries
        original_content = content
        for pattern in self.code_noise_patterns:
            content = pattern.sub('', content)
        
        if content != original_content:
            result.stages_applied.append("code_noise_removal")
            result.removed_elements["code_noise"] = len(original_content) - len(content)
        
        # Stage 2: Strip API keys, tokens, passwords, secrets
        content = self._sanitize_code_secrets(content)
        result.stages_applied.append("secret_sanitization")
        
        # Stage 3: Remove unused imports and redundant comments
        content = self._clean_code_structure(content)
        result.stages_applied.append("code_structure_cleaning")
        
        # Stage 4: Normalize code style (basic)
        content = self._normalize_code_style(content)
        result.stages_applied.append("code_style_normalization")
        
        # Stage 5: Keep meaningful docstrings and examples
        content = self._preserve_meaningful_code_content(content)
        result.stages_applied.append("meaningful_content_preservation")
        
        result.processed_content = content
        return result
    
    def _apply_global_validation(self, result: PreprocessingResult) -> PreprocessingResult:
        """Apply cross-format global validation and cleanup."""
        
        content = result.processed_content
        
        # Stage 1: Remove timestamps, boilerplate copyright lines
        content = self._remove_global_boilerplate(content)
        result.stages_applied.append("global_boilerplate_removal")
        
        # Stage 2: Collapse multiple spaces/tabs into single spaces
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        result.stages_applied.append("space_normalization")
        
        # Stage 3: Apply language detection and filtering
        if self.target_languages:
            lang_result = self.language_detector.detect_language(content)
            if (lang_result.language not in self.target_languages or 
                lang_result.confidence < self.min_language_confidence):
                result.validation_warnings.append(
                    f"Language {lang_result.language} (confidence: {lang_result.confidence:.2f}) "
                    f"not in target languages: {self.target_languages}"
                )
        
        # Stage 4: Remove hallucinated/generated boilerplate text
        content = self._remove_generated_content(content)
        result.stages_applied.append("generated_content_removal")
        
        # Stage 5: Final UTF-8 normalization
        norm_result = self.text_normalizer.normalize_text(content)
        content = norm_result.normalized_text
        if norm_result.corrections_applied:
            result.stages_applied.append("final_normalization")
        
        # Stage 6: Validate content quality
        if len(content.strip()) < self.min_content_length:
            result.validation_errors.append(f"Content too short: {len(content)} < {self.min_content_length}")
        
        if len(content) > self.max_content_length:
            result.validation_warnings.append(f"Content very long: {len(content)} > {self.max_content_length}")
        
        # Stage 7: Basic PII sanitization (simplified)
        content = self._basic_pii_sanitization(content)
        result.stages_applied.append("pii_sanitization")
        
        result.processed_content = content.strip()
        return result
    
    def preprocess_batch(self, documents: Dict[str, Tuple[str, str]]) -> BatchPreprocessingResult:
        """
        Preprocess multiple documents with cross-document deduplication.
        
        Args:
            documents: Dict mapping doc_id to (content, content_type) tuples
            
        Returns:
            BatchPreprocessingResult with individual and aggregate results
        """
        batch_result = BatchPreprocessingResult(
            total_documents=len(documents),
            processed_documents=0,
            failed_documents=0
        )
        
        # Stage 1: Process individual documents
        for doc_id, doc_data in documents.items():
            # Handle both formats: string content or (content, content_type) tuple
            if isinstance(doc_data, tuple):
                content, content_type = doc_data
            else:
                content = doc_data
                content_type = 'txt'  # Default to text
            try:
                preprocessing_result = self.preprocess_content(content, content_type, doc_id)
                batch_result.document_results[doc_id] = preprocessing_result
                
                if preprocessing_result.validation_passed:
                    batch_result.processed_documents += 1
                else:
                    batch_result.failed_documents += 1
                
                # Accumulate statistics
                batch_result.total_original_size += preprocessing_result.original_size
                batch_result.total_processed_size += preprocessing_result.processed_size
                
                # Track content types
                content_type_key = preprocessing_result.content_type
                batch_result.content_type_stats[content_type_key] = (
                    batch_result.content_type_stats.get(content_type_key, 0) + 1
                )
                
            except Exception as e:
                batch_result.failed_documents += 1
                error_result = PreprocessingResult(
                    original_content=content,
                    processed_content="",
                    content_type=content_type,
                    validation_passed=False
                )
                error_result.validation_errors.append(f"Processing failed: {str(e)}")
                batch_result.document_results[doc_id] = error_result
        
        # Stage 2: Cross-document deduplication
        if self.enable_cross_format_dedup and batch_result.processed_documents > 1:
            self._perform_cross_document_deduplication(batch_result)
        
        # Stage 3: Calculate aggregate statistics
        if batch_result.processed_documents > 0:
            total_quality = sum(
                result.quality_score for result in batch_result.document_results.values()
                if result.validation_passed
            )
            batch_result.average_quality_score = total_quality / batch_result.processed_documents
        
        return batch_result
    
    def _perform_cross_document_deduplication(self, batch_result: BatchPreprocessingResult):
        """Perform cross-document deduplication."""
        
        # Create content map for deduplication
        content_map = {}
        for doc_id, result in batch_result.document_results.items():
            if result.validation_passed and result.processed_content.strip():
                content_map[doc_id] = result.processed_content
        
        if len(content_map) > 1:
            try:
                dedup_result = self.deduplication_engine.deduplicate_documents(content_map)
                
                # Group duplicates
                duplicate_groups = []
                for group in dedup_result.duplicate_groups:
                    if len(group.document_ids) > 1:
                        duplicate_groups.append(group.document_ids)
                        # Mark all but the first as duplicates
                        for doc_id in group.document_ids[1:]:
                            batch_result.removed_duplicates.add(doc_id)
                            if doc_id in batch_result.document_results:
                                batch_result.document_results[doc_id].validation_warnings.append(
                                    "Document marked as duplicate"
                                )
                
                batch_result.duplicate_groups = duplicate_groups
                
            except Exception as e:
                # Deduplication failed, but continue
                pass
    
    # Helper methods for format-specific cleaning
    
    def _clean_whitespace_and_formatting(self, content: str) -> str:
        """Clean whitespace and formatting issues."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single space
        content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)  # Trim lines
        
        # Fix irregular indentation
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Normalize indentation to spaces
            line = line.expandtabs(4)
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _remove_repeated_paragraphs(self, content: str) -> str:
        """Remove repeated paragraphs and boilerplate."""
        paragraphs = content.split('\n\n')
        seen_paragraphs = set()
        unique_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 20:  # Only check substantial paragraphs
                # Create a normalized version for comparison
                normalized = re.sub(r'\s+', ' ', paragraph.lower())
                if normalized not in seen_paragraphs:
                    seen_paragraphs.add(normalized)
                    unique_paragraphs.append(paragraph)
            elif paragraph:  # Keep short paragraphs as-is
                unique_paragraphs.append(paragraph)
        
        return '\n\n'.join(unique_paragraphs)
    
    def _clean_metadata_and_junk(self, content: str) -> str:
        """Clean metadata, revision history, and junk characters."""
        # Remove revision history patterns
        content = re.sub(r'Revision \d+.*?\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Version \d+\.\d+.*?\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Last modified by.*?\n', '', content, flags=re.IGNORECASE)
        
        # Remove junk characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{4,}', '...', content)
        content = re.sub(r'[-]{4,}', '---', content)
        content = re.sub(r'[=]{4,}', '===', content)
        
        return content
    
    def _normalize_special_elements(self, content: str) -> str:
        """Normalize lists, tables, equations, and special symbols."""
        # Normalize bullet points
        content = re.sub(r'^[\s]*[•·▪▫‣⁃]\s*', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^[\s]*[-*+]\s*', '• ', content, flags=re.MULTILINE)
        
        # Normalize numbered lists
        content = re.sub(r'^[\s]*\d+[.)]\s*', lambda m: f"{m.group().strip()[:-1]}. ", content, flags=re.MULTILINE)
        
        # Normalize special symbols
        symbol_map = {
            '…': '...',
            '–': '-',
            '—': '-',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '«': '"',
            '»': '"',
        }
        
        for special, normal in symbol_map.items():
            content = content.replace(special, normal)
        
        return content
    
    def _resolve_web_references(self, content: str) -> str:
        """Resolve relative links and expand inline references."""
        # Convert relative links to absolute (simplified)
        content = re.sub(r'href="/', 'href="https://example.com/', content)
        content = re.sub(r'src="/', 'src="https://example.com/', content)
        
        # Expand common abbreviations
        abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            'vs.': 'versus',
            'cf.': 'compare',
        }
        
        for abbrev, expansion in abbreviations.items():
            content = content.replace(abbrev, expansion)
        
        return content
    
    def _remove_auto_generated_web_content(self, content: str) -> str:
        """Remove auto-generated web content."""
        # Remove "related posts" sections
        content = re.sub(r'Related Posts?:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'You might also like:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove social sharing
        content = re.sub(r'Share this:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'Follow us:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove comment sections
        content = re.sub(r'Comments?:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'Leave a comment.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        return content
    
    def _clean_web_metadata(self, content: str) -> str:
        """Clean web-specific metadata."""
        # Remove timestamps
        content = re.sub(r'Published:?\s*\w+\s*\d{1,2},?\s*\d{4}', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Updated:?\s*\w+\s*\d{1,2},?\s*\d{4}', '', content, flags=re.IGNORECASE)
        
        # Remove author bylines
        content = re.sub(r'By:?\s*[A-Z][a-z]+\s*[A-Z][a-z]+', '', content)
        content = re.sub(r'Author:?\s*[A-Z][a-z]+\s*[A-Z][a-z]+', '', content)
        
        # Remove reading time
        content = re.sub(r'\d+\s*min(?:ute)?s?\s*read', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _clean_json_data(self, data: Any) -> Any:
        """Clean JSON data recursively."""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Normalize field names
                clean_key = self._normalize_field_name(key)
                clean_value = self._clean_json_data(value)
                
                # Skip empty or placeholder values
                if not self._is_placeholder_value(clean_value):
                    cleaned[clean_key] = clean_value
            return cleaned
            
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                clean_item = self._clean_json_data(item)
                if not self._is_placeholder_value(clean_item):
                    cleaned.append(clean_item)
            return cleaned
            
        elif isinstance(data, str):
            # Clean string values
            data = data.strip()
            if self._is_placeholder_value(data):
                return None
            return data
            
        else:
            return data
    
    def _clean_yaml_data(self, data: Any) -> Any:
        """Clean YAML data (similar to JSON)."""
        return self._clean_json_data(data)
    
    def _clean_csv_data(self, content: str) -> str:
        """Clean CSV data."""
        lines = content.strip().split('\n')
        if not lines:
            return content
        
        # Parse CSV
        csv_reader = csv.reader(io.StringIO(content))
        rows = list(csv_reader)
        
        if not rows:
            return content
        
        # Clean header row
        header = rows[0]
        clean_header = [self._normalize_field_name(col) for col in header]
        
        # Clean data rows
        clean_rows = [clean_header]
        seen_rows = set()
        
        for row in rows[1:]:
            # Clean row values
            clean_row = []
            for value in row:
                clean_value = value.strip() if isinstance(value, str) else str(value)
                if self._is_placeholder_value(clean_value):
                    clean_value = ""
                clean_row.append(clean_value)
            
            # Deduplicate rows
            row_key = tuple(clean_row)
            if row_key not in seen_rows and any(clean_row):  # Skip empty rows
                seen_rows.add(row_key)
                clean_rows.append(clean_row)
        
        # Convert back to CSV
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(clean_rows)
        return output.getvalue()
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field names to snake_case."""
        # Convert to snake_case
        field_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', field_name)
        field_name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', field_name)
        field_name = field_name.lower()
        
        # Replace spaces and special characters with underscores
        field_name = re.sub(r'[^\w]', '_', field_name)
        
        # Remove multiple underscores
        field_name = re.sub(r'_+', '_', field_name)
        
        # Remove leading/trailing underscores
        field_name = field_name.strip('_')
        
        return field_name
    
    def _is_placeholder_value(self, value: Any) -> bool:
        """Check if a value is a placeholder that should be removed."""
        if value is None:
            return True
        
        if isinstance(value, str):
            value = value.strip().lower()
            placeholders = {'', 'n/a', 'na', 'null', 'none', 'undefined', '---', 'tbd', 'todo', 'xxx'}
            return value in placeholders
        
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        
        return False
    
    def _remove_image_artifacts(self, content: str) -> str:
        """Remove image-specific artifacts from OCR content."""
        # Remove watermark indicators
        content = re.sub(r'\[WATERMARK\]', '', content, flags=re.IGNORECASE)
        content = re.sub(r'WATERMARK', '', content, flags=re.IGNORECASE)
        
        # Remove logo indicators
        content = re.sub(r'\[LOGO\]', '', content, flags=re.IGNORECASE)
        content = re.sub(r'LOGO', '', content, flags=re.IGNORECASE)
        
        # Remove decorative captions
        content = re.sub(r'Figure \d+:.*?\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Image \d+:.*?\n', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _clean_image_noise(self, content: str) -> str:
        """Clean noise from image OCR."""
        # Remove border artifacts
        content = re.sub(r'^[|_\-=]+$', '', content, flags=re.MULTILINE)
        
        # Remove page marks
        content = re.sub(r'^\s*\|\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*[_\-]{3,}\s*$', '', content, flags=re.MULTILINE)
        
        # Remove single character lines (likely artifacts)
        content = re.sub(r'^\s*[^\w\s]\s*$', '', content, flags=re.MULTILINE)
        
        return content
    
    def _segment_text_blocks(self, content: str) -> str:
        """Ensure proper text block segmentation."""
        # Split into lines and group into blocks
        lines = content.split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_block:
                    blocks.append(' '.join(current_block))
                    current_block = []
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append(' '.join(current_block))
        
        return '\n\n'.join(blocks)
    
    def _sanitize_code_secrets(self, content: str) -> str:
        """Remove secrets from code content."""
        # Replace API keys, passwords, tokens with placeholders
        content = re.sub(r'(api_key\s*=\s*)["\'][^"\']+["\']', r'\1"[REDACTED]"', content, flags=re.IGNORECASE)
        content = re.sub(r'(password\s*=\s*)["\'][^"\']+["\']', r'\1"[REDACTED]"', content, flags=re.IGNORECASE)
        content = re.sub(r'(token\s*=\s*)["\'][^"\']+["\']', r'\1"[REDACTED]"', content, flags=re.IGNORECASE)
        content = re.sub(r'(secret\s*=\s*)["\'][^"\']+["\']', r'\1"[REDACTED]"', content, flags=re.IGNORECASE)
        
        # Remove hardcoded URLs with credentials
        content = re.sub(r'https?://[^:]+:[^@]+@[^\s]+', 'https://[REDACTED]', content)
        
        return content
    
    def _clean_code_structure(self, content: str) -> str:
        """Clean code structure by removing unused imports and redundant comments."""
        lines = content.split('\n')
        cleaned_lines = []
        
        # Track imports and usage (simplified)
        imports = set()
        used_imports = set()
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty comment lines
            if re.match(r'^\s*#\s*$', line):
                continue
            
            # Skip redundant comments
            if re.match(r'^\s*#.*TODO.*$', line, re.IGNORECASE):
                continue
            if re.match(r'^\s*#.*FIXME.*$', line, re.IGNORECASE):
                continue
            if re.match(r'^\s*#.*DEBUG.*$', line, re.IGNORECASE):
                continue
            
            # Track imports (simplified)
            import_match = re.match(r'^\s*(?:from\s+\w+\s+)?import\s+(\w+)', stripped)
            if import_match:
                imports.add(import_match.group(1))
            
            # Check for import usage (simplified)
            for imp in imports:
                if imp in stripped and not stripped.startswith('import'):
                    used_imports.add(imp)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_code_style(self, content: str) -> str:
        """Basic code style normalization."""
        # Normalize indentation to 4 spaces
        content = content.expandtabs(4)
        
        # Remove trailing whitespace
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove excessive blank lines
        result_lines = []
        blank_count = 0
        
        for line in cleaned_lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Max 2 consecutive blank lines
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _preserve_meaningful_code_content(self, content: str) -> str:
        """Preserve meaningful docstrings and examples."""
        # This is a placeholder - in practice, you'd want more sophisticated
        # analysis to determine what constitutes "meaningful" content
        
        # Keep docstrings
        content = re.sub(r'""".*?"""', lambda m: m.group(0), content, flags=re.DOTALL)
        
        # Keep comments that look like explanations
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            # Keep lines with substantial comments
            if re.match(r'^\s*#.{20,}', line):  # Comments with 20+ chars
                result_lines.append(line)
            elif not re.match(r'^\s*#', line):  # Keep non-comment lines
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_global_boilerplate(self, content: str) -> str:
        """Remove global boilerplate patterns."""
        # Remove timestamp patterns
        content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', content)
        
        # Remove copyright boilerplate
        content = re.sub(r'©\s*\d{4}.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Copyright\s*\d{4}.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        
        # Remove common signatures
        content = re.sub(r'Best regards,?\s*\n.*?(?:\n\n|\Z)', '', content, flags=re.DOTALL)
        content = re.sub(r'Sincerely,?\s*\n.*?(?:\n\n|\Z)', '', content, flags=re.DOTALL)
        
        return content
    
    def _remove_generated_content(self, content: str) -> str:
        """Remove hallucinated/generated boilerplate text."""
        # Remove AI-generated disclaimers
        content = re.sub(r'As an AI.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'I am an AI.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        
        # Remove generic boilerplate
        content = re.sub(r'This document.*?automatically generated.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Generated by.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _calculate_quality_score(self, result: PreprocessingResult) -> float:
        """Calculate comprehensive quality score."""
        if not result.processed_content:
            return 0.0
        
        score_components = []
        
        # Length score (0.0 - 1.0)
        length = len(result.processed_content)
        if length < self.min_content_length:
            length_score = length / self.min_content_length * 0.5
        elif length <= 1000:
            length_score = 0.5 + (length / 1000) * 0.5
        else:
            length_score = 1.0
        score_components.append(('length', length_score, 0.3))
        
        # Processing success score
        processing_score = 1.0 if result.validation_passed else 0.0
        score_components.append(('processing', processing_score, 0.2))
        
        # Content diversity score (based on character variety)
        char_variety = len(set(result.processed_content.lower())) / 26  # Rough estimate
        diversity_score = min(1.0, char_variety)
        score_components.append(('diversity', diversity_score, 0.2))
        
        # Noise removal score (based on compression ratio)
        compression_ratio = result.get_compression_ratio()
        if 0.1 <= compression_ratio <= 0.4:  # Good compression range
            noise_score = 1.0
        elif compression_ratio < 0.1:
            noise_score = 0.8  # Little cleaning done
        else:
            noise_score = max(0.3, 1.0 - (compression_ratio - 0.4) * 2)  # Too much removed
        score_components.append(('noise_removal', noise_score, 0.2))
        
        # Validation score
        validation_score = 1.0 if not result.validation_errors else 0.5
        score_components.append(('validation', validation_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return round(max(0.0, min(1.0, total_score)), 3)
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """Get statistics about the preprocessing configuration."""
        return {
            'supported_formats': {
                'text_formats': list(self.text_formats),
                'web_formats': list(self.web_formats),
                'structured_formats': list(self.structured_formats),
                'image_formats': list(self.image_formats),
                'code_formats': list(self.code_formats),
                'archive_formats': list(self.archive_formats)
            },
            'quality_thresholds': {
                'min_content_length': self.min_content_length,
                'max_content_length': self.max_content_length,
                'min_quality_score': self.min_quality_score,
                'max_noise_ratio': self.max_noise_ratio
            },
            'language_settings': {
                'target_languages': list(self.target_languages),
                'min_language_confidence': self.min_language_confidence
            },
            'deduplication_settings': {
                'similarity_threshold': self.similarity_threshold,
                'cross_format_enabled': self.enable_cross_format_dedup
            },
            'pattern_counts': {
                'text_noise_patterns': len(self.text_noise_patterns),
                'web_noise_patterns': len(self.web_noise_patterns),
                'ocr_error_patterns': len(self.ocr_error_patterns),
                'code_noise_patterns': len(self.code_noise_patterns),
                'system_files': len(self.system_files)
            }
        }
    
    def _basic_pii_sanitization(self, content: str) -> str:
        """Basic PII sanitization (simplified version)."""
        # Email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Phone numbers (US format)
        content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', content)
        content = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '[PHONE]', content)
        
        # Social Security Numbers
        content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
        
        # Credit card numbers (basic pattern)
        content = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', content)
        
        return content


def preprocess_for_llm_training(content: str, content_type: str, 
                               config: Dict[str, Any] = None) -> PreprocessingResult:
    """
    Convenience function for comprehensive preprocessing.
    
    Args:
        content: Raw content to preprocess
        content_type: Type of content (pdf, html, json, etc.)
        config: Optional configuration dictionary
        
    Returns:
        PreprocessingResult with processed content and metadata
    """
    preprocessor = ComprehensivePreprocessor(config)
    return preprocessor.preprocess_content(content, content_type)


def validate_dataset_cleanliness(dataset: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate dataset cleanliness with automated tests.
    
    Args:
        dataset: Dictionary mapping document IDs to content
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        'total_documents': len(dataset),
        'passed_documents': 0,
        'failed_documents': 0,
        'issues_found': [],
        'recommendations': []
    }
    
    for doc_id, content in dataset.items():
        issues = []
        
        # Check for broken sentences
        if re.search(r'[a-z]\s+[A-Z](?![A-Z])', content):
            issues.append("Potential broken sentences detected")
        
        # Check for gibberish (excessive consonant clusters)
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', content)
        if len(consonant_clusters) > len(content) / 1000:  # More than 1 per 1000 chars
            issues.append("Potential gibberish text detected")
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', content)) / len(content) if content else 0
        if special_char_ratio > 0.1:  # More than 10% special characters
            issues.append("Excessive special characters")
        
        # Check for proper UTF-8 encoding
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Invalid UTF-8 encoding")
        
        if issues:
            validation_results['failed_documents'] += 1
            validation_results['issues_found'].append({
                'document_id': doc_id,
                'issues': issues
            })
        else:
            validation_results['passed_documents'] += 1
    
    # Generate recommendations
    if validation_results['failed_documents'] > 0:
        validation_results['recommendations'].append(
            "Run comprehensive preprocessing on failed documents"
        )
    
    if validation_results['failed_documents'] / validation_results['total_documents'] > 0.1:
        validation_results['recommendations'].append(
            "Consider adjusting preprocessing parameters - high failure rate detected"
        )
        
        return validation_results
