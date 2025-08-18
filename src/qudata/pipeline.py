"""
Main processing pipeline that connects all QuData components.

This module provides the primary interface for running complete data processing
workflows from ingestion through export.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .models import Document, ProcessingResult, ProcessingError, ErrorSeverity
from .config import ConfigManager, PipelineConfig
from .ingest import FileTypeDetector, PlainTextExtractor, PDFExtractor, DocumentExtractor
from .clean import ComprehensiveCleaningPipeline
from .annotate import TaxonomyClassifier, MetadataExtractor
from .score import QualityScorer
from .export import ContentSegmenter
from .export.llmbuilder import LLMBuilderConnector
from .analyze import AnalysisEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""
    success: bool
    documents_processed: int
    documents_failed: int
    processing_time: float
    stage_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[ProcessingError] = field(default_factory=list)
    output_paths: Dict[str, str] = field(default_factory=dict)


class QuDataPipeline:
    """
    Main QuData processing pipeline that orchestrates all components.
    
    Provides a unified interface for running complete data processing workflows
    from raw data ingestion through final export for LLM training.
    """
    
    def __init__(self, config: PipelineConfig = None, config_path: str = None):
        """
        Initialize the QuData pipeline.
        
        Args:
            config: Pipeline configuration object
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            config_manager = ConfigManager()
            self.config = config_manager.load_pipeline_config(config_path)
        else:
            self.config = config or PipelineConfig()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("QuData pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # File detection and extraction
        self.file_detector = FileTypeDetector()
        self.extractors = {
            'txt': PlainTextExtractor(self.config.ingest.model_dump()),
            'pdf': PDFExtractor(self.config.ingest.model_dump()),
            'docx': DocumentExtractor(self.config.ingest.model_dump()),
        }
        
        # Cleaning pipeline
        self.cleaning_pipeline = ComprehensiveCleaningPipeline(
            self.config.clean.model_dump()
        )
        
        # Annotation components
        self.taxonomy_classifier = TaxonomyClassifier(
            self.config.annotate.model_dump()
        )
        self.metadata_extractor = MetadataExtractor()
        
        # Quality scoring
        self.quality_scorer = QualityScorer(self.config.score.model_dump())
        
        # Content segmentation
        self.content_segmenter = ContentSegmenter(
            self.config.pack.model_dump()
        )
        
        # Analysis engine
        self.analysis_engine = AnalysisEngine(
            self.config.model_dump().get('analysis', {})
        )
        
        # LLMBuilder integration
        self.llmbuilder_connector = LLMBuilderConnector(
            self.config.export.model_dump().get('llmbuilder', {})
        )
    
    def process_directory(self, input_dir: str, output_dir: str) -> PipelineResult:
        """
        Process all files in a directory through the complete pipeline.
        
        Args:
            input_dir: Input directory containing raw files
            output_dir: Output directory for processed results
            
        Returns:
            PipelineResult with processing statistics and outputs
        """
        start_time = datetime.now()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting pipeline processing: {input_dir} -> {output_dir}")
        
        result = PipelineResult(
            success=True,
            documents_processed=0,
            documents_failed=0,
            processing_time=0.0
        )
        
        try:
            # Stage 1: Ingestion
            documents = self._ingest_files(input_path)
            result.stage_results['ingestion'] = {
                'files_found': len(list(input_path.rglob('*'))),
                'documents_extracted': len(documents)
            }
            
            if not documents:
                logger.warning("No documents extracted from input directory")
                result.success = False
                return result
            
            # Stage 2: Cleaning
            cleaned_documents = self._clean_documents(documents)
            result.stage_results['cleaning'] = {
                'documents_cleaned': len(cleaned_documents),
                'cleaning_stats': self._get_cleaning_stats(cleaned_documents)
            }
            
            # Stage 3: Annotation
            annotated_documents = self._annotate_documents(cleaned_documents)
            result.stage_results['annotation'] = {
                'documents_annotated': len(annotated_documents)
            }
            
            # Stage 4: Quality Scoring
            scored_documents = self._score_documents(annotated_documents)
            result.stage_results['scoring'] = {
                'documents_scored': len(scored_documents),
                'average_quality': self._calculate_average_quality(scored_documents)
            }
            
            # Stage 5: Content Segmentation
            segmented_documents = self._segment_documents(scored_documents)
            result.stage_results['segmentation'] = {
                'documents_segmented': len(segmented_documents)
            }
            
            # Stage 6: Analysis
            analysis_result = self.analysis_engine.analyze_dataset(segmented_documents)
            result.stage_results['analysis'] = analysis_result.to_dict()
            
            # Stage 7: Export
            export_paths = self._export_documents(segmented_documents, output_path)
            result.output_paths = export_paths
            result.stage_results['export'] = {
                'formats_exported': len(export_paths),
                'export_paths': export_paths
            }
            
            result.documents_processed = len(segmented_documents)
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            result.success = False
            result.errors.append(ProcessingError(
                stage="pipeline",
                error_type="PipelineError",
                message=str(e),
                severity=ErrorSeverity.CRITICAL
            ))
        
        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed in {result.processing_time:.2f}s")
        
        return result
    
    def _ingest_files(self, input_path: Path) -> List[Document]:
        """Ingest and extract content from files."""
        documents = []
        
        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                try:
                    # Detect file type
                    file_type = self.file_detector.detect_file_type(str(file_path))
                    
                    # Get appropriate extractor
                    extractor = self.extractors.get(file_type)
                    if not extractor:
                        logger.warning(f"No extractor for file type: {file_type}")
                        continue
                    
                    # Extract content
                    extracted = extractor.extract(str(file_path))
                    
                    # Create document
                    from .models import create_document_from_extracted
                    document = create_document_from_extracted(
                        extracted, 
                        f"doc_{len(documents)}"
                    )
                    documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return documents
    
    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean document content."""
        cleaned_documents = []
        
        # Prepare documents for batch cleaning
        doc_content = {doc.id: doc.content for doc in documents}
        
        # Run cleaning pipeline
        batch_result = self.cleaning_pipeline.clean_documents(doc_content)
        
        # Update documents with cleaned content
        for document in documents:
            if document.id in batch_result.document_results:
                cleaning_result = batch_result.document_results[document.id]
                if not cleaning_result.errors:
                    document.content = cleaning_result.cleaned_text
                    cleaned_documents.append(document)
        
        return cleaned_documents
    
    def _annotate_documents(self, documents: List[Document]) -> List[Document]:
        """Annotate documents with metadata and categories."""
        annotated_documents = []
        
        for document in documents:
            try:
                # Classify taxonomy
                category_result = self.taxonomy_classifier.classify_document(document)
                document.metadata.domain = category_result.primary_category
                document.metadata.topics = category_result.categories
                
                # Extract additional metadata
                extracted_metadata = self.metadata_extractor.extract_metadata(document)
                if extracted_metadata.author:
                    document.metadata.author = extracted_metadata.author
                if extracted_metadata.creation_date:
                    document.metadata.creation_date = extracted_metadata.creation_date
                
                annotated_documents.append(document)
                
            except Exception as e:
                logger.error(f"Failed to annotate document {document.id}: {e}")
        
        return annotated_documents
    
    def _score_documents(self, documents: List[Document]) -> List[Document]:
        """Score document quality."""
        scored_documents = []
        
        for document in documents:
            try:
                quality_result = self.quality_scorer.score_document(document)
                document.metadata.quality_score = quality_result.overall_score
                scored_documents.append(document)
            except Exception as e:
                logger.error(f"Failed to score document {document.id}: {e}")
        
        return scored_documents
    
    def _segment_documents(self, documents: List[Document]) -> List[Document]:
        """Segment documents for training formats."""
        # For now, return documents as-is
        # Content segmentation would be applied during export
        return documents
    
    def _export_documents(self, documents: List[Document], output_path: Path) -> Dict[str, str]:
        """Export documents in various formats."""
        export_paths = {}
        
        try:
            # Create dataset
            from .models import Dataset, DatasetMetadata
            dataset = Dataset(
                id="processed_dataset",
                name="QuData Processed Dataset",
                version="1.0",
                documents=documents,
                metadata=DatasetMetadata()
            )
            
            # Export to LLMBuilder
            export_result = self.llmbuilder_connector.export_to_llmbuilder(
                dataset, str(output_path / "llmbuilder")
            )
            
            if export_result.success:
                export_paths['llmbuilder'] = export_result.export_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
        
        return export_paths
    
    def _get_cleaning_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return {
            'documents_cleaned': len(documents),
            'average_length': sum(len(doc.content) for doc in documents) / len(documents) if documents else 0
        }
    
    def _calculate_average_quality(self, documents: List[Document]) -> float:
        """Calculate average quality score."""
        if not documents:
            return 0.0
        
        total_quality = sum(doc.metadata.quality_score for doc in documents)
        return total_quality / len(documents)
    
    def process_files(self, file_paths: List[str]) -> 'Dataset':
        """
        Process a list of files and return a dataset.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dataset containing processed documents
        """
        from .models import Dataset, DatasetMetadata, QualityMetrics
        
        documents = []
        
        for file_path in file_paths:
            try:
                # Detect file type
                file_type = self.file_detector.detect_file_type(file_path)
                
                # Get appropriate extractor
                extractor = self.extractors.get(file_type)
                if not extractor:
                    logger.warning(f"No extractor for file type: {file_type}")
                    continue
                
                # Extract content
                extracted = extractor.extract(file_path)
                
                # Create document
                from .models import create_document_from_extracted
                document = create_document_from_extracted(
                    extracted, 
                    f"doc_{len(documents)}"
                )
                
                # Clean document
                doc_content = {document.id: document.content}
                batch_result = self.cleaning_pipeline.clean_documents(doc_content)
                
                if document.id in batch_result.document_results:
                    cleaning_result = batch_result.document_results[document.id]
                    if not cleaning_result.errors:
                        document.content = cleaning_result.cleaned_text
                
                # Annotate document
                category_result = self.taxonomy_classifier.classify_document(document)
                document.metadata.domain = category_result.primary_category
                document.metadata.topics = category_result.categories
                
                # Score document
                quality_result = self.quality_scorer.score_document(document)
                document.metadata.quality_score = quality_result.overall_score
                
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Calculate overall quality metrics
        if documents:
            avg_quality = sum(doc.metadata.quality_score for doc in documents) / len(documents)
            quality_metrics = QualityMetrics(
                overall_score=avg_quality,
                content_quality=avg_quality,
                metadata_completeness=0.8,  # Placeholder
                language_consistency=0.9,   # Placeholder
                domain_relevance=0.7        # Placeholder
            )
        else:
            quality_metrics = QualityMetrics(
                overall_score=0.0,
                content_quality=0.0,
                metadata_completeness=0.0,
                language_consistency=0.0,
                domain_relevance=0.0
            )
        
        # Create dataset
        dataset = Dataset(
            id=f"dataset_{int(datetime.now().timestamp())}",
            name="Processed Dataset",
            version="1.0",
            documents=documents,
            metadata=DatasetMetadata(),
            quality_metrics=quality_metrics
        )
        
        return dataset
    
    def export_dataset(self, dataset: 'Dataset', format_name: str) -> str:
        """
        Export dataset to specified format.
        
        Args:
            dataset: Dataset to export
            format_name: Export format (jsonl, parquet, csv)
            
        Returns:
            Path to exported file
        """
        import tempfile
        import json
        from pathlib import Path
        
        # Create temporary export directory
        temp_dir = Path(tempfile.mkdtemp())
        
        if format_name == "jsonl":
            export_path = temp_dir / f"{dataset.id}.jsonl"
            with open(export_path, 'w', encoding='utf-8') as f:
                for doc in dataset.documents:
                    doc_data = {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": {
                            "source_path": doc.source_path,
                            "file_type": doc.metadata.file_type,
                            "language": doc.metadata.language,
                            "domain": doc.metadata.domain,
                            "quality_score": doc.metadata.quality_score
                        }
                    }
                    f.write(json.dumps(doc_data) + "\n")
        
        elif format_name == "csv":
            export_path = temp_dir / f"{dataset.id}.csv"
            with open(export_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("id,content,source_path,file_type,language,domain,quality_score\n")
                
                # Write data
                for doc in dataset.documents:
                    # Escape content for CSV
                    content = doc.content.replace('"', '""').replace('\n', ' ')
                    f.write(f'"{doc.id}","{content}","{doc.source_path}","{doc.metadata.file_type}","{doc.metadata.language}","{doc.metadata.domain}",{doc.metadata.quality_score}\n')
        
        elif format_name == "parquet":
            try:
                import pandas as pd
                
                # Create DataFrame
                data = []
                for doc in dataset.documents:
                    data.append({
                        "id": doc.id,
                        "content": doc.content,
                        "source_path": doc.source_path,
                        "file_type": doc.metadata.file_type,
                        "language": doc.metadata.language,
                        "domain": doc.metadata.domain,
                        "quality_score": doc.metadata.quality_score
                    })
                
                df = pd.DataFrame(data)
                export_path = temp_dir / f"{dataset.id}.parquet"
                df.to_parquet(export_path, index=False)
                
            except ImportError:
                # Fallback to JSON if pandas not available
                export_path = temp_dir / f"{dataset.id}.json"
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump([doc.to_dict() for doc in dataset.documents], f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format_name}")
        
        return str(export_path)