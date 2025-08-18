"""
Unit tests for core data models in QuData.

Tests cover data model validation, serialization, and base class functionality.
"""

import json
import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qudata.models import (
    Document, DocumentMetadata, ProcessingResult, ProcessingError,
    ErrorSeverity, ErrorAction, ProcessingStage, Entity, TableData, ImageData,
    DocumentStructure, BaseExtractor, ErrorHandler, ProcessingContext,
    FileMetadata, ExtractedContent, create_document_from_extracted,
    serialize_processing_result, deserialize_processing_result
)


class TestEntity:
    """Test cases for the Entity dataclass."""
    
    def test_entity_creation(self):
        """Test creating an Entity with all fields."""
        entity = Entity(
            text="John Doe",
            label="PERSON",
            start=0,
            end=8,
            confidence=0.95
        )
        
        assert entity.text == "John Doe"
        assert entity.label == "PERSON"
        assert entity.start == 0
        assert entity.end == 8
        assert entity.confidence == 0.95
    
    def test_entity_default_confidence(self):
        """Test Entity with default confidence value."""
        entity = Entity(text="Apple", label="ORG", start=10, end=15)
        assert entity.confidence == 0.0
    
    def test_entity_to_dict(self):
        """Test Entity serialization to dictionary."""
        entity = Entity(text="Test", label="MISC", start=0, end=4, confidence=0.8)
        result = entity.to_dict()
        
        expected = {
            "text": "Test",
            "label": "MISC",
            "start": 0,
            "end": 4,
            "confidence": 0.8
        }
        assert result == expected


class TestTableData:
    """Test cases for the TableData dataclass."""
    
    def test_table_data_creation(self):
        """Test creating TableData with headers and rows."""
        table = TableData(
            headers=["Name", "Age", "City"],
            rows=[["John", "25", "NYC"], ["Jane", "30", "LA"]],
            caption="Employee Data"
        )
        
        assert table.headers == ["Name", "Age", "City"]
        assert len(table.rows) == 2
        assert table.caption == "Employee Data"
    
    def test_table_data_no_caption(self):
        """Test TableData without caption."""
        table = TableData(
            headers=["Col1", "Col2"],
            rows=[["A", "B"]]
        )
        assert table.caption is None
    
    def test_table_data_to_dict(self):
        """Test TableData serialization."""
        table = TableData(
            headers=["Name", "Age"],
            rows=[["John", "25"]],
            caption="Test Table"
        )
        result = table.to_dict()
        
        expected = {
            "headers": ["Name", "Age"],
            "rows": [["John", "25"]],
            "caption": "Test Table"
        }
        assert result == expected


class TestImageData:
    """Test cases for the ImageData dataclass."""
    
    def test_image_data_creation(self):
        """Test creating ImageData with all fields."""
        image = ImageData(
            path="image.png",
            caption="Test Image",
            alt_text="A test image",
            width=800,
            height=600
        )
        
        assert image.path == "image.png"
        assert image.caption == "Test Image"
        assert image.alt_text == "A test image"
        assert image.width == 800
        assert image.height == 600
    
    def test_image_data_to_dict(self):
        """Test ImageData serialization."""
        image = ImageData(path="test.jpg", caption="Test")
        result = image.to_dict()
        
        expected = {
            "path": "test.jpg",
            "caption": "Test",
            "alt_text": None,
            "width": None,
            "height": None
        }
        assert result == expected


class TestDocumentStructure:
    """Test cases for the DocumentStructure dataclass."""
    
    def test_structure_defaults(self):
        """Test DocumentStructure with default values."""
        structure = DocumentStructure()
        
        assert structure.headings == []
        assert structure.paragraphs == 0
        assert structure.tables == 0
        assert structure.images == 0
        assert structure.code_blocks == 0
        assert structure.lists == 0
        assert structure.links == 0
    
    def test_structure_with_values(self):
        """Test DocumentStructure with custom values."""
        structure = DocumentStructure(
            headings=["Introduction", "Conclusion"],
            paragraphs=5,
            tables=2,
            images=3,
            code_blocks=1,
            lists=4,
            links=10
        )
        
        assert len(structure.headings) == 2
        assert structure.paragraphs == 5
        assert structure.tables == 2
        assert structure.images == 3
        assert structure.code_blocks == 1
        assert structure.lists == 4
        assert structure.links == 10
    
    def test_structure_to_dict(self):
        """Test DocumentStructure serialization."""
        structure = DocumentStructure(
            headings=["Title"],
            paragraphs=3,
            tables=1
        )
        result = structure.to_dict()
        
        expected = {
            "headings": ["Title"],
            "paragraphs": 3,
            "tables": 1,
            "images": 0,
            "code_blocks": 0,
            "lists": 0,
            "links": 0
        }
        assert result == expected


class TestDocumentMetadata:
    """Test cases for the DocumentMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating DocumentMetadata with all fields."""
        creation_date = datetime(2024, 1, 1, 12, 0, 0)
        entity = Entity("Test", "MISC", 0, 4, 0.8)
        
        metadata = DocumentMetadata(
            file_type="pdf",
            size_bytes=1024,
            language="en",
            author="John Doe",
            creation_date=creation_date,
            domain="technology",
            topics=["AI", "ML"],
            entities=[entity],
            source_url="https://example.com",
            quality_score=0.85
        )
        
        assert metadata.file_type == "pdf"
        assert metadata.size_bytes == 1024
        assert metadata.language == "en"
        assert metadata.author == "John Doe"
        assert metadata.creation_date == creation_date
        assert metadata.domain == "technology"
        assert metadata.topics == ["AI", "ML"]
        assert len(metadata.entities) == 1
        assert metadata.source_url == "https://example.com"
        assert metadata.quality_score == 0.85
    
    def test_metadata_defaults(self):
        """Test DocumentMetadata with default values."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=512,
            language="en"
        )
        
        assert metadata.author is None
        assert metadata.creation_date is None
        assert metadata.domain == "uncategorized"
        assert metadata.topics == []
        assert metadata.entities == []
        assert metadata.encoding == "utf-8"
        assert metadata.quality_score == 0.0
    
    def test_metadata_to_dict(self):
        """Test converting DocumentMetadata to dictionary."""
        creation_date = datetime(2024, 1, 1, 12, 0, 0)
        entity = Entity("Test", "MISC", 0, 4, 0.8)
        
        metadata = DocumentMetadata(
            file_type="pdf",
            size_bytes=1024,
            language="en",
            author="John Doe",
            creation_date=creation_date,
            entities=[entity]
        )
        
        result = metadata.to_dict()
        
        assert result["file_type"] == "pdf"
        assert result["size_bytes"] == 1024
        assert result["author"] == "John Doe"
        assert result["creation_date"] == "2024-01-01T12:00:00"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Test"


class TestProcessingError:
    """Test cases for the ProcessingError dataclass."""
    
    def test_error_creation(self):
        """Test creating a ProcessingError."""
        error = ProcessingError(
            stage="extraction",
            error_type="FileNotFound",
            message="File does not exist",
            severity=ErrorSeverity.HIGH,
            document_id="doc123"
        )
        
        assert error.stage == "extraction"
        assert error.error_type == "FileNotFound"
        assert error.message == "File does not exist"
        assert error.severity == ErrorSeverity.HIGH
        assert error.document_id == "doc123"
        assert isinstance(error.timestamp, datetime)
    
    def test_error_to_dict(self):
        """Test converting ProcessingError to dictionary."""
        error = ProcessingError(
            stage="cleaning",
            error_type="ValidationError",
            message="Invalid content",
            severity=ErrorSeverity.MEDIUM
        )
        
        result = error.to_dict()
        
        assert result["stage"] == "cleaning"
        assert result["error_type"] == "ValidationError"
        assert result["message"] == "Invalid content"
        assert result["severity"] == "medium"
        assert "timestamp" in result
    
    def test_error_as_exception(self):
        """Test that ProcessingError can be raised as an exception."""
        error = ProcessingError(
            stage="test",
            error_type="TestError",
            message="Test error",
            severity=ErrorSeverity.LOW
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            raise error
        
        assert exc_info.value.stage == "test"
        assert exc_info.value.error_type == "TestError"


class TestDocument:
    """Test cases for the Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a Document with all fields."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en"
        )
        
        structure = DocumentStructure(
            headings=["Introduction", "Conclusion"],
            paragraphs=5,
            tables=1
        )
        
        doc = Document(
            id="doc123",
            source_path="/path/to/file.txt",
            content="This is test content",
            metadata=metadata,
            structure=structure
        )
        
        assert doc.id == "doc123"
        assert doc.source_path == "/path/to/file.txt"
        assert doc.content == "This is test content"
        assert doc.version == "1.0"
        assert isinstance(doc.processing_timestamp, datetime)
    
    def test_document_methods(self):
        """Test Document utility methods."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en"
        )
        
        doc = Document(
            id="doc123",
            source_path="/path/to/file.txt",
            content="This is a test document with multiple words",
            metadata=metadata
        )
        
        # Test word count
        assert doc.get_word_count() == 8  # "This is a test document with multiple words"
        
        # Test character count
        assert doc.get_character_count() == 43  # Length of "This is a test document with multiple words"
        
        # Test has_tables and has_images
        assert not doc.has_tables()
        assert not doc.has_images()
        
        # Add tables and images
        doc.tables.append(TableData(["A"], [["1"]]))
        doc.images.append(ImageData("test.jpg"))
        
        assert doc.has_tables()
        assert doc.has_images()
    
    def test_document_to_dict(self):
        """Test converting Document to dictionary."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en"
        )
        
        doc = Document(
            id="doc123",
            source_path="/path/to/file.txt",
            content="Test content",
            metadata=metadata
        )
        
        result = doc.to_dict()
        
        assert result["id"] == "doc123"
        assert result["source_path"] == "/path/to/file.txt"
        assert result["content"] == "Test content"
        assert "metadata" in result
        assert "structure" in result
        assert "processing_timestamp" in result
    
    def test_document_to_json(self):
        """Test converting Document to JSON string."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en"
        )
        
        doc = Document(
            id="doc123",
            source_path="/path/to/file.txt",
            content="Test content",
            metadata=metadata
        )
        
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "doc123"
        assert parsed["content"] == "Test content"


class TestProcessingResult:
    """Test cases for the ProcessingResult dataclass."""
    
    def test_processing_result_success(self):
        """Test creating a successful ProcessingResult."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en"
        )
        
        doc = Document(
            id="doc123",
            source_path="/path/to/file.txt",
            content="Test content",
            metadata=metadata
        )
        
        result = ProcessingResult(
            success=True,
            document=doc,
            processing_time=1.5
        )
        
        assert result.success is True
        assert result.document == doc
        assert result.processing_time == 1.5
        assert result.errors == []
        assert result.warnings == []
    
    def test_processing_result_add_error(self):
        """Test adding errors to ProcessingResult."""
        result = ProcessingResult(success=True)
        
        error = ProcessingError(
            stage="test",
            error_type="TestError",
            message="Test error",
            severity=ErrorSeverity.HIGH
        )
        
        result.add_error(error)
        
        assert len(result.errors) == 1
        assert result.success is False  # Should be set to False for HIGH severity
        assert result.get_error_count() == 1
    
    def test_processing_result_add_warning(self):
        """Test adding warnings to ProcessingResult."""
        result = ProcessingResult(success=True)
        
        result.add_warning("This is a warning")
        
        assert len(result.warnings) == 1
        assert result.warnings[0] == "This is a warning"
        assert result.success is True  # Should remain True for warnings
    
    def test_processing_result_critical_errors(self):
        """Test critical error handling."""
        result = ProcessingResult(success=True)
        
        # Add a critical error
        critical_error = ProcessingError(
            stage="test",
            error_type="CriticalError",
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        
        # Add a medium error
        medium_error = ProcessingError(
            stage="test",
            error_type="MediumError",
            message="Medium failure",
            severity=ErrorSeverity.MEDIUM
        )
        
        result.add_error(critical_error)
        result.add_error(medium_error)
        
        assert result.has_critical_errors()
        critical_errors = result.get_critical_errors()
        assert len(critical_errors) == 1
        assert critical_errors[0].severity == ErrorSeverity.CRITICAL
    
    def test_processing_result_to_dict(self):
        """Test converting ProcessingResult to dictionary."""
        result = ProcessingResult(
            success=True,
            processing_time=2.0
        )
        
        result.add_warning("Test warning")
        
        dict_result = result.to_dict()
        
        assert dict_result["success"] is True
        assert dict_result["processing_time"] == 2.0
        assert len(dict_result["warnings"]) == 1
        assert dict_result["document"] is None


class TestProcessingContext:
    """Test cases for the ProcessingContext class."""
    
    def test_context_creation(self):
        """Test creating a ProcessingContext."""
        config = {"param1": "value1"}
        context = ProcessingContext("doc123", ProcessingStage.INGEST, config)
        
        assert context.document_id == "doc123"
        assert context.stage == ProcessingStage.INGEST
        assert context.config == config
        assert isinstance(context.start_time, datetime)
    
    def test_context_add_metadata(self):
        """Test adding metadata to ProcessingContext."""
        context = ProcessingContext("doc123", ProcessingStage.CLEAN)
        context.add_metadata("key1", "value1")
        context.add_metadata("key2", 42)
        
        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == 42
    
    def test_context_elapsed_time(self):
        """Test getting elapsed time from ProcessingContext."""
        context = ProcessingContext("doc123", ProcessingStage.ANNOTATE)
        elapsed = context.get_elapsed_time()
        
        assert isinstance(elapsed, float)
        assert elapsed >= 0
    
    def test_context_to_dict(self):
        """Test converting ProcessingContext to dictionary."""
        context = ProcessingContext("doc123", ProcessingStage.SCORE, {"test": True})
        context.add_metadata("test_key", "test_value")
        
        result = context.to_dict()
        
        assert result["document_id"] == "doc123"
        assert result["stage"] == "score"
        assert result["config"]["test"] is True
        assert result["metadata"]["test_key"] == "test_value"
        assert "start_time" in result
        assert "elapsed_time" in result


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def test_error_handler_creation(self):
        """Test creating an ErrorHandler."""
        config = {"max_retries": 5, "log_errors": False}
        handler = ErrorHandler(config)
        
        assert handler.config == config
        assert handler.max_retries == 5
        assert handler.log_errors is False
    
    def test_error_handler_default_config(self):
        """Test ErrorHandler with default configuration."""
        handler = ErrorHandler()
        
        assert handler.max_retries == 3
        assert handler.log_errors is True
    
    def test_handle_critical_error(self):
        """Test handling critical errors."""
        handler = ErrorHandler()
        context = ProcessingContext("doc123", ProcessingStage.INGEST)
        
        error = ProcessingError(
            stage="extraction",
            error_type="CriticalError",
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        
        action = handler.handle_error(error, context)
        assert action == ErrorAction.FAIL_FAST
    
    def test_handle_high_error_with_retries(self):
        """Test handling high severity errors with retries available."""
        handler = ErrorHandler()
        context = ProcessingContext("doc123", ProcessingStage.CLEAN)
        
        error = ProcessingError(
            stage="extraction",
            error_type="HighError",
            message="High severity error",
            severity=ErrorSeverity.HIGH
        )
        
        # First attempt should retry
        action = handler.handle_error(error, context)
        assert action == ErrorAction.RETRY
        
        # After max retries, should skip
        for _ in range(3):  # max_retries = 3
            handler.handle_error(error, context)
        
        action = handler.handle_error(error, context)
        assert action == ErrorAction.SKIP
    
    def test_should_retry(self):
        """Test retry logic."""
        handler = ErrorHandler()
        context = ProcessingContext("doc123", ProcessingStage.PACK)
        
        # Critical errors should not retry
        critical_error = ProcessingError(
            stage="extraction",
            error_type="CriticalError",
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        assert not handler.should_retry(critical_error, context)
        
        # Other errors should retry within limits
        medium_error = ProcessingError(
            stage="extraction",
            error_type="MediumError",
            message="Medium error",
            severity=ErrorSeverity.MEDIUM
        )
        assert handler.should_retry(medium_error, context)
    
    def test_retry_count_tracking(self):
        """Test retry count tracking."""
        handler = ErrorHandler()
        context = ProcessingContext("doc123", ProcessingStage.EXPORT)
        
        error = ProcessingError(
            stage="test",
            error_type="TestError",
            message="Test error",
            severity=ErrorSeverity.MEDIUM
        )
        
        # Initial count should be 0
        assert handler.get_retry_count("doc123", ProcessingStage.EXPORT, "TestError") == 0
        
        # Handle error (should increment count)
        handler.handle_error(error, context)
        assert handler.get_retry_count("doc123", ProcessingStage.EXPORT, "TestError") == 1
        
        # Reset counts
        handler.reset_retry_counts()
        assert handler.get_retry_count("doc123", ProcessingStage.EXPORT, "TestError") == 0


class MockExtractor(BaseExtractor):
    """Mock implementation of BaseExtractor for testing."""
    
    def extract(self, file_path: str) -> ExtractedContent:
        """Mock extract method."""
        metadata = self.get_metadata(file_path)
        return ExtractedContent("Mock content", metadata)
    
    def supports_format(self, file_type: str) -> bool:
        """Mock supports_format method."""
        return file_type in ["txt", "md"]


class TestBaseExtractor:
    """Test cases for the BaseExtractor abstract class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_temp_file(self, filename: str, content: str = "test content") -> str:
        """Create a temporary file for testing."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_extractor_creation(self):
        """Test creating a BaseExtractor implementation."""
        config = {"param1": "value1"}
        extractor = MockExtractor(config)
        
        assert extractor.config == config
        assert isinstance(extractor.error_handler, ErrorHandler)
    
    def test_supports_format(self):
        """Test the supports_format method."""
        extractor = MockExtractor()
        
        assert extractor.supports_format("txt") is True
        assert extractor.supports_format("md") is True
        assert extractor.supports_format("pdf") is False
    
    def test_get_metadata(self):
        """Test getting file metadata."""
        extractor = MockExtractor()
        file_path = self._create_temp_file("test.txt", "test content")
        
        metadata = extractor.get_metadata(file_path)
        
        assert metadata.file_path == file_path
        assert metadata.file_type == "txt"
        assert metadata.size_bytes > 0
    
    def test_get_metadata_file_not_found(self):
        """Test getting metadata for non-existent file."""
        extractor = MockExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.get_metadata("/path/to/nonexistent.txt")
    
    def test_create_processing_error(self):
        """Test creating processing errors."""
        extractor = MockExtractor()
        
        error = extractor.create_processing_error(
            stage="test",
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.HIGH,
            document_id="doc123"
        )
        
        assert error.stage == "test"
        assert error.error_type == "TestError"
        assert error.message == "Test message"
        assert error.severity == ErrorSeverity.HIGH
        assert error.document_id == "doc123"
    
    def test_validate_file_success(self):
        """Test successful file validation."""
        extractor = MockExtractor()
        file_path = self._create_temp_file("valid.txt")
        
        result = extractor.validate_file(file_path)
        assert result is True
    
    def test_validate_file_not_found(self):
        """Test file validation with non-existent file."""
        extractor = MockExtractor()
        
        with pytest.raises(ProcessingError) as exc_info:
            extractor.validate_file("/nonexistent/file.txt")
        
        assert exc_info.value.error_type == "FileNotFound"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_validate_file_too_large(self):
        """Test file validation with size limit."""
        config = {"max_file_size_bytes": 10}  # 10 bytes limit
        extractor = MockExtractor(config)
        
        # Create file larger than limit
        file_path = self._create_temp_file("large.txt", "This is a large file content")
        
        with pytest.raises(ProcessingError) as exc_info:
            extractor.validate_file(file_path)
        
        assert exc_info.value.error_type == "FileTooLarge"
        assert exc_info.value.severity == ErrorSeverity.HIGH


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_document_from_extracted(self):
        """Test creating Document from ExtractedContent."""
        file_metadata = FileMetadata("/path/to/file.txt", "txt", 1024)
        structure = DocumentStructure(paragraphs=3)
        
        extracted = ExtractedContent("Test content", file_metadata)
        extracted.structure = structure
        
        doc = create_document_from_extracted(extracted, "doc123")
        
        assert doc.id == "doc123"
        assert doc.source_path == "/path/to/file.txt"
        assert doc.content == "Test content"
        assert doc.metadata.file_type == "txt"
        assert doc.metadata.size_bytes == 1024
        assert doc.structure.paragraphs == 3
    
    def test_serialize_processing_result(self):
        """Test serializing ProcessingResult to JSON."""
        result = ProcessingResult(success=True, processing_time=1.5)
        result.add_warning("Test warning")
        
        json_str = serialize_processing_result(result)
        parsed = json.loads(json_str)
        
        assert parsed["success"] is True
        assert parsed["processing_time"] == 1.5
        assert len(parsed["warnings"]) == 1
    
    def test_deserialize_processing_result(self):
        """Test deserializing ProcessingResult from JSON."""
        json_data = {
            "success": True,
            "document": None,
            "errors": [],
            "warnings": ["Test warning"],
            "processing_time": 2.0,
            "stage_results": {}
        }
        
        json_str = json.dumps(json_data)
        result = deserialize_processing_result(json_str)
        
        assert result["success"] is True
        assert result["processing_time"] == 2.0
        assert len(result["warnings"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])