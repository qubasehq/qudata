"""
Core data models for QuData.

This module contains the fundamental data structures used throughout the processing pipeline,
including document models, metadata structures, and processing results.
"""

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ErrorSeverity(Enum):
    """Severity levels for processing errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorAction(Enum):
    """Actions to take when handling errors."""
    RETRY = "retry"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"
    USE_FALLBACK = "use_fallback"


class ProcessingStage(Enum):
    """Processing pipeline stages."""
    INGEST = "ingest"
    CLEAN = "clean"
    ANNOTATE = "annotate"
    SCORE = "score"
    PACK = "pack"
    EXPORT = "export"


@dataclass
class Entity:
    """Represents a named entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


@dataclass
class TableData:
    """Represents tabular data extracted from documents."""
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary."""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption
        }


@dataclass
class ImageData:
    """Represents image data extracted from documents."""
    path: str
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert image data to dictionary."""
        return {
            "path": self.path,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "width": self.width,
            "height": self.height
        }


@dataclass
class DocumentStructure:
    """Represents the structural elements of a document."""
    headings: List[str] = field(default_factory=list)
    paragraphs: int = 0
    tables: int = 0
    images: int = 0
    code_blocks: int = 0
    lists: int = 0
    links: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert structure to dictionary."""
        return {
            "headings": self.headings,
            "paragraphs": self.paragraphs,
            "tables": self.tables,
            "images": self.images,
            "code_blocks": self.code_blocks,
            "lists": self.lists,
            "links": self.links
        }


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    file_type: str
    size_bytes: int
    language: str
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    domain: str = "uncategorized"
    topics: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    source_url: Optional[str] = None
    encoding: str = "utf-8"
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "language": self.language,
            "author": self.author,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": self.modification_date.isoformat() if self.modification_date else None,
            "domain": self.domain,
            "topics": self.topics,
            "entities": [entity.to_dict() for entity in self.entities],
            "source_url": self.source_url,
            "encoding": self.encoding,
            "quality_score": self.quality_score
        }


@dataclass
class ProcessingError(Exception):
    """Represents an error that occurred during processing."""
    stage: str
    error_type: str
    message: str
    severity: ErrorSeverity
    document_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "document_id": self.document_id,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "context": self.context
        }


@dataclass
class Document:
    """Core document model containing content and metadata."""
    id: str
    source_path: str
    content: str
    metadata: DocumentMetadata
    structure: DocumentStructure = field(default_factory=DocumentStructure)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            "id": self.id,
            "source_path": self.source_path,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "structure": self.structure.to_dict(),
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "version": self.version,
            "tables": [table.to_dict() for table in self.tables],
            "images": [image.to_dict() for image in self.images]
        }
    
    def to_json(self) -> str:
        """Convert document to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def get_word_count(self) -> int:
        """Get word count of document content."""
        return len(self.content.split())
    
    def get_character_count(self) -> int:
        """Get character count of document content."""
        return len(self.content)
    
    def has_tables(self) -> bool:
        """Check if document has tables."""
        return len(self.tables) > 0
    
    def has_images(self) -> bool:
        """Check if document has images."""
        return len(self.images) > 0


@dataclass
class ProcessingResult:
    """Result of processing a document through the pipeline."""
    success: bool
    document: Optional[Document] = None
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ProcessingError) -> None:
        """Add an error to the processing result."""
        self.errors.append(error)
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.success = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning to the processing result."""
        self.warnings.append(message)
    
    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)
    
    def get_critical_errors(self) -> List[ProcessingError]:
        """Get list of critical errors."""
        return [error for error in self.errors if error.severity == ErrorSeverity.CRITICAL]
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.get_critical_errors()) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert processing result to dictionary for serialization."""
        return {
            "success": self.success,
            "document": self.document.to_dict() if self.document else None,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings,
            "processing_time": self.processing_time,
            "stage_results": self.stage_results
        }


class ProcessingContext:
    """Context information for processing operations."""
    
    def __init__(self, document_id: str, stage: ProcessingStage, config: Dict[str, Any] = None):
        """
        Initialize processing context.
        
        Args:
            document_id: Unique identifier for the document being processed
            stage: Current processing stage
            config: Configuration for this processing context
        """
        self.document_id = document_id
        self.stage = stage
        self.config = config or {}
        self.start_time = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the processing context."""
        self.metadata[key] = value
    
    def get_elapsed_time(self) -> float:
        """Get elapsed processing time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "document_id": self.document_id,
            "stage": self.stage.value,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "elapsed_time": self.get_elapsed_time(),
            "metadata": self.metadata
        }


class ErrorHandler:
    """Base class for handling processing errors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize error handler.
        
        Args:
            config: Configuration for error handling behavior
        """
        self.config = config or {}
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = self.config.get("max_retries", 3)
        self.log_errors = self.config.get("log_errors", True)
    
    def handle_error(self, error: ProcessingError, context: ProcessingContext) -> ErrorAction:
        """
        Handle a processing error and determine the appropriate action.
        
        Args:
            error: The processing error that occurred
            context: The processing context
            
        Returns:
            The action to take for this error
        """
        # Log error if enabled
        if self.log_errors:
            self.log_error(error, context)
        
        # Check retry count
        retry_key = f"{context.document_id}:{context.stage.value}:{error.error_type}"
        current_retries = self.retry_counts.get(retry_key, 0)
        
        # Determine action based on error severity and retry count
        if error.severity == ErrorSeverity.CRITICAL:
            return ErrorAction.FAIL_FAST
        elif error.severity == ErrorSeverity.HIGH:
            if current_retries < self.max_retries:
                self.retry_counts[retry_key] = current_retries + 1
                return ErrorAction.RETRY
            else:
                return ErrorAction.SKIP
        elif error.severity == ErrorSeverity.MEDIUM:
            if current_retries < self.max_retries:
                self.retry_counts[retry_key] = current_retries + 1
                return ErrorAction.USE_FALLBACK
            else:
                return ErrorAction.SKIP
        else:  # LOW severity
            return ErrorAction.SKIP
    
    def should_retry(self, error: ProcessingError, context: ProcessingContext) -> bool:
        """Check if an error should be retried."""
        retry_key = f"{context.document_id}:{context.stage.value}:{error.error_type}"
        current_retries = self.retry_counts.get(retry_key, 0)
        return current_retries < self.max_retries and error.severity != ErrorSeverity.CRITICAL
    
    def log_error(self, error: ProcessingError, context: ProcessingContext) -> None:
        """Log an error with appropriate formatting."""
        print(f"[{error.severity.value.upper()}] {error.stage}: {error.message}")
        if error.document_id:
            print(f"  Document: {error.document_id}")
        if context:
            print(f"  Stage: {context.stage.value}")
            print(f"  Elapsed time: {context.get_elapsed_time():.2f}s")
        if error.stack_trace:
            print(f"  Stack trace: {error.stack_trace}")
    
    def reset_retry_counts(self) -> None:
        """Reset all retry counts."""
        self.retry_counts.clear()
    
    def get_retry_count(self, document_id: str, stage: ProcessingStage, error_type: str) -> int:
        """Get current retry count for a specific error."""
        retry_key = f"{document_id}:{stage.value}:{error_type}"
        return self.retry_counts.get(retry_key, 0)


class FileMetadata:
    """Metadata about a file before processing."""
    
    def __init__(self, file_path: str, file_type: str, size_bytes: int):
        """
        Initialize file metadata.
        
        Args:
            file_path: Path to the file
            file_type: Type/extension of the file
            size_bytes: Size of the file in bytes
        """
        self.file_path = file_path
        self.file_type = file_type
        self.size_bytes = size_bytes
        self.creation_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "creation_time": self.creation_time.isoformat()
        }


class ExtractedContent:
    """Content extracted from a file."""
    
    def __init__(self, content: str, metadata: FileMetadata):
        """
        Initialize extracted content.
        
        Args:
            content: The extracted text content
            metadata: Metadata about the source file
        """
        self.content = content
        self.metadata = metadata
        self.structure: Optional[DocumentStructure] = None
        self.tables: List[TableData] = []
        self.images: List[ImageData] = []
        self.extraction_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "structure": self.structure.to_dict() if self.structure else None,
            "tables": [table.to_dict() for table in self.tables],
            "images": [image.to_dict() for image in self.images],
            "extraction_time": self.extraction_time.isoformat()
        }


class BaseExtractor(ABC):
    """Abstract base class for all file extractors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base extractor.
        
        Args:
            config: Configuration for the extractor
        """
        self.config = config or {}
        self.error_handler = ErrorHandler(config)
    
    @abstractmethod
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file to extract content from
            
        Returns:
            ExtractedContent object containing the extracted content and metadata
            
        Raises:
            ProcessingError: If extraction fails
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_type: str) -> bool:
        """
        Check if this extractor supports the given file type.
        
        Args:
            file_type: The file type to check (e.g., 'pdf', 'docx', 'txt')
            
        Returns:
            True if the extractor supports this file type, False otherwise
        """
        pass
    
    def get_metadata(self, file_path: str) -> FileMetadata:
        """
        Get basic metadata about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileMetadata object with basic file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        return FileMetadata(
            file_path=file_path,
            file_type=file_extension,
            size_bytes=file_size
        )
    
    def create_processing_error(self, stage: str, error_type: str, message: str, 
                               severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                               document_id: str = None, stack_trace: str = None) -> ProcessingError:
        """Helper method to create processing errors."""
        return ProcessingError(
            stage=stage,
            error_type=error_type,
            message=message,
            severity=severity,
            document_id=document_id,
            stack_trace=stack_trace
        )
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that a file can be processed.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid for processing
            
        Raises:
            ProcessingError: If file validation fails
        """
        if not os.path.exists(file_path):
            raise self.create_processing_error(
                stage="validation",
                error_type="FileNotFound",
                message=f"File not found: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        
        if not os.access(file_path, os.R_OK):
            raise self.create_processing_error(
                stage="validation",
                error_type="PermissionError",
                message=f"Cannot read file: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        
        # Check file size limits if configured
        max_size = self.config.get("max_file_size_bytes")
        if max_size:
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                raise self.create_processing_error(
                    stage="validation",
                    error_type="FileTooLarge",
                    message=f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)",
                    severity=ErrorSeverity.HIGH
                )
        
        return True


# Utility functions for working with models

def create_document_from_extracted(extracted: ExtractedContent, document_id: str) -> Document:
    """
    Create a Document from ExtractedContent.
    
    Args:
        extracted: The extracted content
        document_id: Unique identifier for the document
        
    Returns:
        Document object
    """
    # Convert FileMetadata to DocumentMetadata
    doc_metadata = DocumentMetadata(
        file_type=extracted.metadata.file_type,
        size_bytes=extracted.metadata.size_bytes,
        language="unknown",  # Will be detected later
        encoding="utf-8"
    )
    
    return Document(
        id=document_id,
        source_path=extracted.metadata.file_path,
        content=extracted.content,
        metadata=doc_metadata,
        structure=extracted.structure or DocumentStructure(),
        tables=extracted.tables,
        images=extracted.images
    )


def serialize_processing_result(result: ProcessingResult) -> str:
    """
    Serialize a ProcessingResult to JSON string.
    
    Args:
        result: The processing result to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def deserialize_processing_result(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a ProcessingResult from JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Dictionary representation of the processing result
    """
    return json.loads(json_str)


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset."""
    overall_score: float = 0.0
    length_score: float = 0.0
    language_score: float = 0.0
    coherence_score: float = 0.0
    uniqueness_score: float = 0.0
    completeness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quality metrics to dictionary."""
        return {
            "overall_score": self.overall_score,
            "length_score": self.length_score,
            "language_score": self.language_score,
            "coherence_score": self.coherence_score,
            "uniqueness_score": self.uniqueness_score,
            "completeness_score": self.completeness_score
        }


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    creation_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset metadata to dictionary."""
        return {
            "creation_date": self.creation_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "description": self.description,
            "tags": self.tags,
            "source": self.source,
            "license": self.license
        }


@dataclass
class DatasetSplits:
    """Dataset splits for training/validation/test."""
    train_documents: List[str] = field(default_factory=list)  # Document IDs
    validation_documents: List[str] = field(default_factory=list)
    test_documents: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset splits to dictionary."""
        return {
            "train_documents": self.train_documents,
            "validation_documents": self.validation_documents,
            "test_documents": self.test_documents
        }


@dataclass
class Dataset:
    """A collection of documents with metadata and quality metrics."""
    id: str
    name: str
    version: str
    documents: List[Document] = field(default_factory=list)
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    splits: Optional[DatasetSplits] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "documents": [doc.to_dict() for doc in self.documents],
            "metadata": self.metadata.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "splits": self.splits.to_dict() if self.splits else None
        }
    
    def to_json(self) -> str:
        """Convert dataset to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def get_document_count(self) -> int:
        """Get total number of documents in the dataset."""
        return len(self.documents)
    
    def get_total_word_count(self) -> int:
        """Get total word count across all documents."""
        return sum(doc.get_word_count() for doc in self.documents)
    
    def get_total_character_count(self) -> int:
        """Get total character count across all documents."""
        return sum(doc.get_character_count() for doc in self.documents)
    
    def get_languages(self) -> List[str]:
        """Get list of unique languages in the dataset."""
        return list(set(doc.metadata.language for doc in self.documents))
    
    def get_domains(self) -> List[str]:
        """Get list of unique domains in the dataset."""
        return list(set(doc.metadata.domain for doc in self.documents))
    
    def get_documents_by_domain(self, domain: str) -> List[Document]:
        """Get all documents in a specific domain."""
        return [doc for doc in self.documents if doc.metadata.domain == domain]
    
    def get_documents_by_language(self, language: str) -> List[Document]:
        """Get all documents in a specific language."""
        return [doc for doc in self.documents if doc.metadata.language == language]
    
    def add_document(self, document: Document) -> None:
        """Add a document to the dataset."""
        self.documents.append(document)
        self.metadata.last_modified = datetime.now()
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the dataset by ID."""
        original_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.id != document_id]
        if len(self.documents) < original_count:
            self.metadata.last_modified = datetime.now()
            return True
        return False
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None
    
    def update_quality_metrics(self, metrics: QualityMetrics) -> None:
        """Update the dataset's quality metrics."""
        self.quality_metrics = metrics
        self.metadata.last_modified = datetime.now()
    
    def create_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     test_ratio: float = 0.1) -> DatasetSplits:
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio: Proportion of documents for training
            val_ratio: Proportion of documents for validation
            test_ratio: Proportion of documents for testing
            
        Returns:
            DatasetSplits object with document IDs assigned to each split
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        import random
        doc_ids = [doc.id for doc in self.documents]
        random.shuffle(doc_ids)
        
        total_docs = len(doc_ids)
        train_count = int(total_docs * train_ratio)
        val_count = int(total_docs * val_ratio)
        
        splits = DatasetSplits(
            train_documents=doc_ids[:train_count],
            validation_documents=doc_ids[train_count:train_count + val_count],
            test_documents=doc_ids[train_count + val_count:]
        )
        
        self.splits = splits
        return splits