"""
Comprehensive text cleaning pipeline that integrates all cleaning components.

This module provides a unified interface for applying all text cleaning operations
including normalization, OCR correction, deduplication, and boilerplate removal
using the enhanced configuration system.
"""

import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import TextNormalizer, OCRCorrector, EncodingDetector, NormalizationResult
from .dedupe import DeduplicationEngine, DeduplicationResult
from .boilerplate import BoilerplateRemover, BoilerplateRemovalResult
from .language import LanguageDetector, LanguageResult, FilterResult
from ..models import ProcessingError, ErrorSeverity, Document


@dataclass
class CleaningResult:
    """Result of comprehensive text cleaning pipeline."""
    original_text: str
    cleaned_text: str
    
    # Individual step results
    normalization_result: Optional[NormalizationResult] = None
    boilerplate_result: Optional[BoilerplateRemovalResult] = None
    language_result: Optional[LanguageResult] = None
    language_filter_result: Optional[FilterResult] = None
    
    # Applied operations
    operations_applied: List[str] = field(default_factory=list)
    
    # Quality metrics
    quality_score: float = 0.0
    length_reduction_ratio: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_length_reduction(self) -> int:
        """Get absolute length reduction in characters."""
        return len(self.original_text) - len(self.cleaned_text)
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio (0.0 = no compression, 1.0 = complete removal)."""
        if not self.original_text:
            return 0.0
        return self.get_length_reduction() / len(self.original_text)


@dataclass
class BatchCleaningResult:
    """Result of batch cleaning operation."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    
    # Individual results
    document_results: Dict[str, CleaningResult] = field(default_factory=dict)
    
    # Deduplication results
    deduplication_result: Optional[DeduplicationResult] = None
    
    # Aggregate statistics
    total_original_length: int = 0
    total_cleaned_length: int = 0
    average_quality_score: float = 0.0
    
    def get_success_rate(self) -> float:
        """Get processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents
    
    def get_overall_compression_ratio(self) -> float:
        """Get overall compression ratio across all documents."""
        if self.total_original_length == 0:
            return 0.0
        return (self.total_original_length - self.total_cleaned_length) / self.total_original_length


class ComprehensiveCleaningPipeline:
    """
    Comprehensive text cleaning pipeline that applies all cleaning operations.
    
    Integrates normalization, OCR correction, boilerplate removal, and deduplication
    with configurable settings and quality scoring.
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the comprehensive cleaning pipeline.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        self.config = config or {}
        
        # Load configuration from file if provided
        if config_file:
            self._load_config_file(config_file)
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline settings
        self.enable_normalization = self.config.get('normalization', {}).get('enabled', True)
        self.enable_ocr_correction = self.config.get('ocr_correction', {}).get('enabled', True)
        self.enable_boilerplate_removal = self.config.get('boilerplate_removal', {}).get('enabled', True)
        self.enable_deduplication = self.config.get('deduplication', {}).get('enabled', True)
        self.enable_language_detection = self.config.get('language_detection', {}).get('enabled', True)
        self.enable_language_filtering = self.config.get('language_filtering', {}).get('enabled', False)
        self.enable_quality_scoring = self.config.get('quality_scoring', {}).get('enabled', True)
        
        # Quality thresholds
        quality_config = self.config.get('quality_scoring', {})
        self.min_quality_score = quality_config.get('thresholds', {}).get('min_quality_score', 0.5)
        self.min_length = quality_config.get('thresholds', {}).get('min_length', 100)
        self.max_boilerplate_ratio = quality_config.get('thresholds', {}).get('max_boilerplate_ratio', 0.4)
    
    def _load_config_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with existing config
                    self._deep_merge_config(self.config, file_config)
        except Exception as e:
            raise ProcessingError(
                stage="config_loading",
                error_type="ConfigurationError",
                message=f"Failed to load config file {config_file}: {str(e)}",
                severity=ErrorSeverity.HIGH
            )
    
    def _deep_merge_config(self, base: Dict, update: Dict):
        """Deep merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _initialize_components(self):
        """Initialize all cleaning components."""
        # Text normalizer
        norm_config = self.config.get('normalization', {})
        self.normalizer = TextNormalizer(norm_config)
        
        # OCR corrector
        ocr_config = self.config.get('ocr_correction', {})
        self.ocr_corrector = OCRCorrector(ocr_config)
        
        # Encoding detector
        self.encoding_detector = EncodingDetector()
        
        # Boilerplate remover
        self.boilerplate_remover = BoilerplateRemover(self.config)
        
        # Deduplication engine
        self.deduplication_engine = DeduplicationEngine(self.config)
        
        # Language detector
        lang_config = self.config.get('language_detection', {})
        self.language_detector = LanguageDetector(lang_config)
    
    def clean_text(self, text: str, document_id: str = None, content_type: str = 'txt') -> CleaningResult:
        """
        Clean a single text using the comprehensive pipeline.
        
        Args:
            text: Input text to clean
            document_id: Optional document identifier
            content_type: Type of content (txt, html, json, etc.)
            
        Returns:
            CleaningResult with cleaned text and metadata
        """
        import time
        start_time = time.time()
        
        result = CleaningResult(
            original_text=text,
            cleaned_text=text
        )
        
        try:
            # Use comprehensive preprocessor for advanced cleaning
            from .comprehensive_preprocessor import ComprehensivePreprocessor
            
            preprocessor = ComprehensivePreprocessor(self.config)
            preprocessing_result = preprocessor.preprocess_content(text, content_type, document_id)
            
            # Map preprocessing result to cleaning result
            result.cleaned_text = preprocessing_result.processed_content
            result.quality_score = preprocessing_result.quality_score
            result.processing_time = preprocessing_result.processing_time
            result.stages_applied = preprocessing_result.stages_applied
            result.validation_passed = preprocessing_result.validation_passed
            result.validation_errors = preprocessing_result.validation_errors
            result.validation_warnings = preprocessing_result.validation_warnings
            
            # Legacy fallback for older pipeline components if comprehensive preprocessing fails
            if not preprocessing_result.validation_passed and preprocessing_result.processed_content:
                current_text = preprocessing_result.processed_content
            elif not preprocessing_result.validation_passed:
                # Fall back to legacy cleaning
                current_text = text
                
                # Step 1: Text normalization
                if self.enable_normalization and current_text:
                    norm_result = self.normalizer.normalize_text(current_text)
                    current_text = norm_result.normalized_text
                result.normalization_result = norm_result
                result.operations_applied.extend(norm_result.corrections_applied)
                
                if norm_result.corrections_applied:
                    result.operations_applied.append("text_normalization")
            
            # Step 2: OCR correction
            if self.enable_ocr_correction and current_text:
                ocr_result = self.ocr_corrector.correct_ocr_errors(current_text)
                current_text = ocr_result.normalized_text
                result.operations_applied.extend(ocr_result.corrections_applied)
                
                if ocr_result.corrections_applied:
                    result.operations_applied.append("ocr_correction")
            
            # Step 3: Boilerplate removal
            if self.enable_boilerplate_removal and current_text:
                boilerplate_result = self.boilerplate_remover.remove_boilerplate(current_text)
                current_text = boilerplate_result.cleaned_text
                result.boilerplate_result = boilerplate_result
                
                if boilerplate_result.removed_patterns:
                    result.operations_applied.append("boilerplate_removal")
            
            # Step 4: Language detection and filtering
            if self.enable_language_detection and current_text:
                language_result = self.language_detector.detect_language(current_text)
                result.language_result = language_result
                result.operations_applied.append("language_detection")
                
                # Optional language filtering
                if self.enable_language_filtering:
                    filter_result = self.language_detector.filter_by_language(current_text)
                    result.language_filter_result = filter_result
                    
                    if not filter_result.should_keep:
                        result.warnings.append(f"Content filtered due to language: {filter_result.reason}")
                        # Optionally set text to empty or mark for removal
                        # For now, we'll keep the text but add a warning
                    
                    result.operations_applied.append("language_filtering")
            
            # Step 5: Final cleanup
            current_text = self._final_cleanup(current_text)
            
            result.cleaned_text = current_text
            
            # Step 6: Quality scoring
            if self.enable_quality_scoring:
                result.quality_score = self._calculate_quality_score(result)
            
            # Calculate metrics
            result.length_reduction_ratio = result.get_compression_ratio()
            result.processing_time = time.time() - start_time
            
        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            result.quality_score = 0.0
        
        return result
    
    def clean_documents(self, documents: Dict[str, str], content_types: Dict[str, str] = None) -> BatchCleaningResult:
        """
        Clean multiple documents with optional deduplication.
        
        Args:
            documents: Dictionary mapping document IDs to content
            
        Returns:
            BatchCleaningResult with individual and aggregate results
        """
        batch_result = BatchCleaningResult(
            total_documents=len(documents),
            processed_documents=0,
            failed_documents=0
        )
        
        # Step 1: Clean individual documents
        for doc_id, content in documents.items():
            try:
                content_type = content_types.get(doc_id, 'txt') if content_types else 'txt'
                cleaning_result = self.clean_text(content, doc_id, content_type)
                batch_result.document_results[doc_id] = cleaning_result
                batch_result.processed_documents += 1
                
                # Accumulate statistics
                batch_result.total_original_length += len(cleaning_result.original_text)
                batch_result.total_cleaned_length += len(cleaning_result.cleaned_text)
                
            except Exception as e:
                batch_result.failed_documents += 1
                # Create error result
                error_result = CleaningResult(
                    original_text=content,
                    cleaned_text="",
                    errors=[f"Processing failed: {str(e)}"]
                )
                batch_result.document_results[doc_id] = error_result
        
        # Step 2: Deduplication (if enabled)
        if self.enable_deduplication and batch_result.processed_documents > 1:
            try:
                # Create cleaned documents dict for deduplication
                cleaned_docs = {
                    doc_id: result.cleaned_text 
                    for doc_id, result in batch_result.document_results.items()
                    if result.cleaned_text and not result.errors
                }
                
                if len(cleaned_docs) > 1:
                    dedup_result = self.deduplication_engine.deduplicate_documents(cleaned_docs)
                    batch_result.deduplication_result = dedup_result
                    
                    # Mark duplicates in individual results
                    for doc_id in dedup_result.removed_ids:
                        if doc_id in batch_result.document_results:
                            batch_result.document_results[doc_id].operations_applied.append("marked_as_duplicate")
                            batch_result.document_results[doc_id].warnings.append("Document marked as duplicate")
                            
            except Exception as e:
                # Deduplication failed, but continue
                pass
        
        # Step 3: Calculate aggregate statistics
        if batch_result.processed_documents > 0:
            total_quality = sum(
                result.quality_score for result in batch_result.document_results.values()
                if not result.errors
            )
            batch_result.average_quality_score = total_quality / batch_result.processed_documents
        
        return batch_result
    
    def _final_cleanup(self, text: str) -> str:
        """Apply final cleanup operations to text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= 3 or not stripped:  # Keep empty lines and lines with 3+ chars
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _calculate_quality_score(self, result: CleaningResult) -> float:
        """Calculate quality score for cleaned text."""
        if not result.cleaned_text:
            return 0.0
        
        quality_config = self.config.get('quality_scoring', {})
        weights = quality_config.get('weights', {})
        
        # Length score
        length_score = self._calculate_length_score(result.cleaned_text)
        
        # Language confidence score
        language_score = self._calculate_language_score(result)
        
        # Boilerplate ratio score
        boilerplate_score = self._calculate_boilerplate_score(result)
        
        # Duplicate penalty (if applicable)
        duplicate_score = 1.0  # Would be adjusted based on deduplication results
        
        # PII penalty (simplified)
        pii_score = self._calculate_pii_score(result.cleaned_text)
        
        # Weighted average
        total_score = (
            length_score * weights.get('length_score', 0.2) +
            language_score * weights.get('language_confidence', 0.2) +
            boilerplate_score * weights.get('boilerplate_ratio', 0.3) +
            duplicate_score * weights.get('duplicate_penalty', 0.2) +
            pii_score * weights.get('pii_penalty', 0.1)
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length."""
        if not text:
            return 0.0
        
        quality_config = self.config.get('quality_scoring', {})
        optimal_length = quality_config.get('optimal_length', 1000)
        min_length = quality_config.get('thresholds', {}).get('min_length', 100)
        
        length = len(text)
        
        if length < min_length:
            return length / min_length * 0.5  # Penalty for short text
        elif length <= optimal_length:
            return 0.5 + (length / optimal_length) * 0.5  # Scale from 0.5 to 1.0
        else:
            # Diminishing returns for very long text
            excess_ratio = (length - optimal_length) / optimal_length
            return max(0.8, 1.0 - excess_ratio * 0.1)
    
    def _calculate_boilerplate_score(self, result: CleaningResult) -> float:
        """Calculate score based on boilerplate removal."""
        if not result.boilerplate_result:
            return 1.0
        
        removal_ratio = result.boilerplate_result.get_removal_ratio()
        
        # Good score if moderate boilerplate removal, penalty for excessive removal
        if removal_ratio <= 0.1:
            return 1.0  # Little to no boilerplate
        elif removal_ratio <= self.max_boilerplate_ratio:
            return 1.0 - (removal_ratio / self.max_boilerplate_ratio) * 0.3
        else:
            return 0.5  # Too much boilerplate removed
    
    def _calculate_language_score(self, result: CleaningResult) -> float:
        """Calculate score based on language detection confidence."""
        if not result.language_result:
            return 1.0  # No language detection performed
        
        # Base score on confidence and reliability
        confidence_score = result.language_result.confidence
        reliability_bonus = 0.1 if result.language_result.is_reliable else 0.0
        
        # Penalty if language filtering failed
        filter_penalty = 0.0
        if result.language_filter_result and not result.language_filter_result.should_keep:
            filter_penalty = 0.3
        
        return max(0.0, min(1.0, confidence_score + reliability_bonus - filter_penalty))
    
    def _calculate_pii_score(self, text: str) -> float:
        """Calculate score based on PII detection (simplified)."""
        # This would integrate with PII detection patterns
        # For now, return a default score
        return 1.0
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pipeline configuration."""
        return {
            'components_enabled': {
                'normalization': self.enable_normalization,
                'ocr_correction': self.enable_ocr_correction,
                'boilerplate_removal': self.enable_boilerplate_removal,
                'deduplication': self.enable_deduplication,
                'language_detection': self.enable_language_detection,
                'language_filtering': self.enable_language_filtering,
                'quality_scoring': self.enable_quality_scoring
            },
            'boilerplate_patterns': len(self.boilerplate_remover.patterns),
            'quality_thresholds': {
                'min_quality_score': self.min_quality_score,
                'min_length': self.min_length,
                'max_boilerplate_ratio': self.max_boilerplate_ratio
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate pipeline configuration and return any issues."""
        issues = []
        
        # Check required components
        if not any([self.enable_normalization, self.enable_boilerplate_removal]):
            issues.append("At least one cleaning component should be enabled")
        
        # Check quality scoring configuration
        if self.enable_quality_scoring:
            quality_config = self.config.get('quality_scoring', {})
            if not quality_config.get('weights'):
                issues.append("Quality scoring enabled but no weights configured")
        
        # Check boilerplate patterns
        if self.enable_boilerplate_removal and len(self.boilerplate_remover.patterns) == 0:
            issues.append("Boilerplate removal enabled but no patterns loaded")
        
        return issues