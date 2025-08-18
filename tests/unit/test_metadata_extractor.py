"""
Unit tests for MetadataExtractor.

Tests the metadata extraction functionality including author, date,
source URL, and title extraction from document content and file properties.
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.qudata.annotate.metadata import MetadataExtractor, ExtractedMetadata
from src.qudata.models import Document, DocumentMetadata, ProcessingError


class TestExtractedMetadata:
    """Test ExtractedMetadata class."""
    
    def test_extracted_metadata_creation(self):
        """Test creating ExtractedMetadata."""
        metadata = ExtractedMetadata(
            author="John Doe",
            creation_date=datetime(2023, 1, 15),
            source_url="https://example.com",
            document_type="pdf",
            confidence=0.8
        )
        
        assert metadata.author == "John Doe"
        assert metadata.creation_date == datetime(2023, 1, 15)
        assert metadata.source_url == "https://example.com"
        assert metadata.document_type == "pdf"
        assert metadata.confidence == 0.8
    
    def test_extracted_metadata_defaults(self):
        """Test ExtractedMetadata with default values."""
        metadata = ExtractedMetadata()
        
        assert metadata.author is None
        assert metadata.creation_date is None
        assert metadata.source_url is None
        assert metadata.document_type is None
        assert metadata.confidence == 0.0
        assert metadata.extraction_method == "unknown"


class TestMetadataExtractor:
    """Test MetadataExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create MetadataExtractor instance for testing."""
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=1000,
            language="en"
        )
        
        return Document(
            id="test_doc_1",
            source_path="/path/to/test.txt",
            content="Sample document content",
            metadata=metadata
        )
    
    def test_init_default(self):
        """Test initializing extractor with defaults."""
        extractor = MetadataExtractor()
        
        assert extractor.config == {}
        assert len(extractor._author_patterns) > 0
        assert len(extractor._date_patterns) > 0
        assert len(extractor._url_patterns) > 0
        assert len(extractor._title_patterns) > 0
    
    def test_init_with_config(self):
        """Test initializing extractor with config."""
        config = {"test_setting": "value"}
        extractor = MetadataExtractor(config=config)
        
        assert extractor.config == config
    
    def test_extract_author_simple(self, extractor):
        """Test extracting author from simple patterns."""
        content = "Author: John Doe\nThis is the document content."
        
        author = extractor._extract_author(content)
        
        assert author == "John Doe"
    
    def test_extract_author_by_pattern(self, extractor):
        """Test extracting author using 'by' pattern."""
        content = "Written by Jane Smith\nDocument content here."
        
        author = extractor._extract_author(content)
        
        assert author == "Jane Smith"
    
    def test_extract_author_html_meta(self, extractor):
        """Test extracting author from HTML meta tag."""
        content = '<meta name="author" content="Alice Johnson">\nHTML content'
        
        author = extractor._extract_author(content)
        
        assert author == "Alice Johnson"
    
    def test_extract_author_not_found(self, extractor):
        """Test author extraction when no author is found."""
        content = "This document has no author information."
        
        author = extractor._extract_author(content)
        
        assert author is None
    
    def test_extract_date_iso_format(self, extractor):
        """Test extracting date in ISO format."""
        content = "Created: 2023-05-15\nDocument content"
        
        date = extractor._extract_date(content)
        
        assert date == datetime(2023, 5, 15)
    
    def test_extract_date_us_format(self, extractor):
        """Test extracting date in US format."""
        content = "Date: 05/15/2023\nDocument content"
        
        date = extractor._extract_date(content)
        
        assert date == datetime(2023, 5, 15)
    
    def test_extract_date_written_format(self, extractor):
        """Test extracting date in written format."""
        content = "Published: May 15, 2023\nDocument content"
        
        date = extractor._extract_date(content)
        
        assert date == datetime(2023, 5, 15)
    
    def test_extract_date_not_found(self, extractor):
        """Test date extraction when no date is found."""
        content = "This document has no date information."
        
        date = extractor._extract_date(content)
        
        assert date is None
    
    def test_extract_url_simple(self, extractor):
        """Test extracting URL from simple pattern."""
        content = "Source: https://example.com/article\nContent here"
        
        url = extractor._extract_url(content)
        
        assert url == "https://example.com/article"
    
    def test_extract_url_in_text(self, extractor):
        """Test extracting URL from within text."""
        content = "Visit https://docs.example.com for more information."
        
        url = extractor._extract_url(content)
        
        assert url == "https://docs.example.com"
    
    def test_extract_url_not_found(self, extractor):
        """Test URL extraction when no URL is found."""
        content = "This document has no URL information."
        
        url = extractor._extract_url(content)
        
        assert url is None
    
    def test_extract_title_simple(self, extractor):
        """Test extracting title from simple pattern."""
        content = "Title: Introduction to Machine Learning\nContent here"
        
        title = extractor._extract_title(content)
        
        assert title == "Introduction to Machine Learning"
    
    def test_extract_title_html(self, extractor):
        """Test extracting title from HTML."""
        content = "<title>Web Page Title</title>\nHTML content"
        
        title = extractor._extract_title(content)
        
        assert title == "Web Page Title"
    
    def test_extract_title_markdown_h1(self, extractor):
        """Test extracting title from Markdown H1."""
        content = "# Document Title\n\nMarkdown content here"
        
        title = extractor._extract_title(content)
        
        assert title == "Document Title"
    
    def test_extract_title_not_found(self, extractor):
        """Test title extraction when no title is found."""
        content = "This document has no title information."
        
        title = extractor._extract_title(content)
        
        assert title is None
    
    def test_parse_date_string_iso(self, extractor):
        """Test parsing ISO date string."""
        date = extractor._parse_date_string("2023-05-15")
        
        assert date == datetime(2023, 5, 15)
    
    def test_parse_date_string_us_format(self, extractor):
        """Test parsing US format date string."""
        date = extractor._parse_date_string("05/15/2023")
        
        assert date == datetime(2023, 5, 15)
    
    def test_parse_date_string_written(self, extractor):
        """Test parsing written date string."""
        date = extractor._parse_date_string("May 15, 2023")
        
        assert date == datetime(2023, 5, 15)
    
    def test_parse_date_string_invalid(self, extractor):
        """Test parsing invalid date string."""
        date = extractor._parse_date_string("invalid date")
        
        assert date is None
    
    @patch('os.path.exists', return_value=True)
    @patch('os.stat')
    def test_extract_file_metadata(self, mock_stat, mock_exists, extractor):
        """Test extracting metadata from file system."""
        # Mock file stats
        mock_stat.return_value = MagicMock()
        mock_stat.return_value.st_ctime = 1684137600  # 2023-05-15 12:00:00
        mock_stat.return_value.st_mtime = 1684224000  # 2023-05-16 12:00:00
        
        metadata = extractor._extract_file_metadata("/path/to/test.pdf")
        
        assert metadata is not None
        assert metadata['document_type'] == 'pdf'
        assert isinstance(metadata['creation_date'], datetime)
        assert isinstance(metadata['modification_date'], datetime)
    
    @patch('os.path.exists', return_value=False)
    def test_extract_file_metadata_missing_file(self, mock_exists, extractor):
        """Test extracting metadata from missing file."""
        metadata = extractor._extract_file_metadata("/nonexistent/file.txt")
        
        assert metadata is None
    
    def test_extract_content_metadata_comprehensive(self, extractor):
        """Test extracting metadata from content with multiple fields."""
        content = """
        Title: Complete Guide to Python
        Author: John Doe
        Date: 2023-05-15
        Source: https://example.com/python-guide
        
        This is a comprehensive guide to Python programming.
        """
        
        metadata = extractor._extract_content_metadata(content)
        
        assert metadata.title == "Complete Guide to Python"
        assert metadata.author == "John Doe"
        assert metadata.creation_date == datetime(2023, 5, 15)
        assert metadata.source_url == "https://example.com/python-guide"
        assert metadata.confidence == 1.0  # All 4 fields extracted
        assert metadata.extraction_method == "content_patterns"
    
    def test_extract_content_metadata_partial(self, extractor):
        """Test extracting partial metadata from content."""
        content = """
        Author: Jane Smith
        
        This document only has author information.
        """
        
        metadata = extractor._extract_content_metadata(content)
        
        assert metadata.author == "Jane Smith"
        assert metadata.title is None
        assert metadata.creation_date is None
        assert metadata.source_url is None
        assert metadata.confidence == 0.25  # 1 out of 4 fields
    
    def test_extract_metadata_from_document(self, extractor, sample_document):
        """Test extracting metadata from a complete document."""
        # Add content with metadata
        sample_document.content = """
        Title: Test Document
        Author: Test Author
        Date: 2023-05-15
        
        This is test content.
        """
        
        with patch.object(extractor, '_extract_file_metadata', return_value=None):
            metadata = extractor.extract_metadata(sample_document)
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.creation_date == datetime(2023, 5, 15)
        assert metadata.confidence > 0
    
    def test_extract_metadata_error_handling(self, extractor):
        """Test error handling in metadata extraction."""
        # Create document that will cause an error
        document = Document(
            id="error_doc",
            source_path="/nonexistent/path",
            content=None,  # This should cause an error
            metadata=None
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            extractor.extract_metadata(document)
        
        assert exc_info.value.stage == "metadata_extraction"
        assert exc_info.value.error_type == "ExtractionError"
        assert exc_info.value.document_id == "error_doc"
    
    def test_update_document_metadata(self, extractor, sample_document):
        """Test updating document with extracted metadata."""
        extracted = ExtractedMetadata(
            author="New Author",
            creation_date=datetime(2023, 6, 1),
            source_url="https://newurl.com",
            document_type="pdf"
        )
        
        updated_doc = extractor.update_document_metadata(sample_document, extracted)
        
        assert updated_doc.metadata.author == "New Author"
        assert updated_doc.metadata.creation_date == datetime(2023, 6, 1)
        assert updated_doc.metadata.source_url == "https://newurl.com"
        assert updated_doc.metadata.file_type == "pdf"
    
    def test_update_document_metadata_no_existing_metadata(self, extractor):
        """Test updating document that has no existing metadata."""
        document = Document(
            id="no_meta_doc",
            source_path="/path/to/file.txt",
            content="Content",
            metadata=None
        )
        
        extracted = ExtractedMetadata(
            author="Author Name",
            document_type="txt"
        )
        
        updated_doc = extractor.update_document_metadata(document, extracted)
        
        assert updated_doc.metadata is not None
        assert updated_doc.metadata.author == "Author Name"
        assert updated_doc.metadata.file_type == "txt"
    
    def test_get_extraction_stats(self, extractor):
        """Test getting extraction statistics."""
        documents = []
        
        # Create documents with varying metadata
        for i in range(3):
            content = f"Author: Author {i}\nTitle: Document {i}\nContent here"
            doc = Document(
                id=f"doc_{i}",
                source_path=f"/path/doc_{i}.txt",
                content=content,
                metadata=DocumentMetadata(file_type="txt", size_bytes=100, language="en")
            )
            documents.append(doc)
        
        stats = extractor.get_extraction_stats(documents)
        
        assert stats['total_documents'] == 3
        assert stats['author_extracted'] == 3
        assert stats['title_extracted'] == 3
        assert stats['avg_confidence'] > 0
        assert 'content_patterns' in stats['extraction_methods']
    
    def test_get_extraction_stats_empty_list(self, extractor):
        """Test getting extraction statistics for empty document list."""
        stats = extractor.get_extraction_stats([])
        
        assert stats['total_documents'] == 0
        assert stats['author_extracted'] == 0
        assert stats['avg_confidence'] == 0.0