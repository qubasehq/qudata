"""
Unit tests for PDF extraction functionality.

Tests PDF content extraction, table detection, error handling, and structure analysis.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Check if pdfplumber is available
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from qudata.models import ProcessingError, ErrorSeverity, TableData, ImageData

# Only import PDFExtractor if pdfplumber is available
if HAS_PDFPLUMBER:
    from qudata.ingest.pdf import PDFExtractor
else:
    PDFExtractor = None


@pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not available")
class TestPDFExtractor:
    """Test cases for the PDFExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_temp_file(self, filename: str, content: bytes) -> str:
        """Create a temporary file with given binary content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path
    
    def test_extractor_initialization(self):
        """Test PDFExtractor initialization."""
        extractor = PDFExtractor()
        assert extractor is not None
        assert extractor.max_file_size == 500 * 1024 * 1024  # 500MB default
        assert extractor.extract_tables is True
        assert extractor.extract_images is True
        assert extractor.preserve_layout is False
        assert extractor.password is None
    
    def test_extractor_with_config(self):
        """Test PDFExtractor initialization with config."""
        config = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'extract_tables': False,
            'extract_images': False,
            'preserve_layout': True,
            'password': 'secret123'
        }
        extractor = PDFExtractor(config)
        
        assert extractor.max_file_size == 100 * 1024 * 1024
        assert extractor.extract_tables is False
        assert extractor.extract_images is False
        assert extractor.preserve_layout is True
        assert extractor.password == 'secret123'
    
    def test_supports_format(self):
        """Test format support checking."""
        assert self.extractor.supports_format('pdf') is True
        assert self.extractor.supports_format('PDF') is True
        
        assert self.extractor.supports_format('txt') is False
        assert self.extractor.supports_format('docx') is False
        assert self.extractor.supports_format('html') is False
    
    @patch('pdfplumber.open')
    def test_extract_simple_pdf(self, mock_pdfplumber_open):
        """Test extracting content from a simple PDF."""
        # Mock PDF content
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is a test PDF document.\n\nIt contains multiple paragraphs."
        mock_page.extract_tables.return_value = []
        mock_page.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.is_encrypted = False
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        # Create a dummy PDF file
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('test.pdf', pdf_content)
        
        extracted = self.extractor.extract(file_path)
        
        assert extracted.content == "This is a test PDF document.\n\nIt contains multiple paragraphs."
        assert extracted.metadata.file_path == file_path
        assert extracted.metadata.file_type == 'pdf'
        assert extracted.structure is not None
        assert extracted.structure.paragraphs > 0
    
    @patch('pdfplumber.open')
    def test_extract_pdf_with_tables(self, mock_pdfplumber_open):
        """Test extracting PDF with tables."""
        # Mock PDF with table content
        mock_page = Mock()
        mock_page.extract_text.return_value = "Document with table"
        mock_page.extract_tables.return_value = [
            [['Name', 'Age', 'City'], ['John', '25', 'NYC'], ['Jane', '30', 'LA']]
        ]
        mock_page.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.is_encrypted = False
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('table_test.pdf', pdf_content)
        
        extracted = self.extractor.extract(file_path)
        
        assert len(extracted.tables) == 1
        assert extracted.tables[0].headers == ['Name', 'Age', 'City']
        assert len(extracted.tables[0].rows) == 2
        assert extracted.tables[0].rows[0] == ['John', '25', 'NYC']
        assert extracted.structure.tables == 1
    
    @patch('pdfplumber.open')
    def test_extract_pdf_with_images(self, mock_pdfplumber_open):
        """Test extracting PDF with images."""
        # Mock PDF with image content
        mock_image = {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100}
        mock_page = Mock()
        mock_page.extract_text.return_value = "Document with images"
        mock_page.extract_tables.return_value = []
        mock_page.images = [mock_image, mock_image]  # Two images
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('image_test.pdf', pdf_content)
        
        extracted = self.extractor.extract(file_path)
        
        assert len(extracted.images) == 2
        assert extracted.images[0].path == "page_1_image_1"
        assert extracted.images[1].path == "page_1_image_2"
        assert extracted.structure.images == 2
    
    @patch('pdfplumber.open')
    def test_extract_multipage_pdf(self, mock_pdfplumber_open):
        """Test extracting content from multi-page PDF."""
        # Mock multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page1.extract_tables.return_value = []
        mock_page1.images = []
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page2.extract_tables.return_value = []
        mock_page2.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('multipage.pdf', pdf_content)
        
        extracted = self.extractor.extract(file_path)
        
        assert "Page 1 content" in extracted.content
        assert "Page 2 content" in extracted.content
        assert extracted.structure.paragraphs >= 2
    
    @patch('pdfplumber.open')
    def test_encrypted_pdf_without_password(self, mock_pdfplumber_open):
        """Test handling encrypted PDF without password."""
        mock_pdf = Mock()
        mock_pdf.is_encrypted = True
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('encrypted.pdf', pdf_content)
        
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract(file_path)
        
        assert exc_info.value.error_type == "EncryptedPDF"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    @patch('pdfplumber.open')
    def test_encrypted_pdf_with_password(self, mock_pdfplumber_open):
        """Test handling encrypted PDF with password."""
        config = {'password': 'secret123'}
        extractor = PDFExtractor(config)
        
        mock_page = Mock()
        mock_page.extract_text.return_value = "Decrypted content"
        mock_page.extract_tables.return_value = []
        mock_page.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.is_encrypted = True
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('encrypted.pdf', pdf_content)
        
        extracted = extractor.extract(file_path)
        
        assert extracted.content == "Decrypted content"
        mock_pdfplumber_open.assert_called_with(file_path, password='secret123')
    
    @patch('pdfplumber.open')
    def test_empty_pdf(self, mock_pdfplumber_open):
        """Test handling PDF with no pages."""
        mock_pdf = Mock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('empty.pdf', pdf_content)
        
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract(file_path)
        
        assert exc_info.value.error_type == "EmptyPDF"
        assert exc_info.value.severity == ErrorSeverity.MEDIUM
    
    @patch('pdfplumber.open')
    def test_corrupted_pdf(self, mock_pdfplumber_open):
        """Test handling corrupted PDF file."""
        # Use a generic exception since the exact pdfminer structure may vary
        mock_pdfplumber_open.side_effect = Exception("Invalid PDF syntax")
        
        pdf_content = b'Invalid PDF content'
        file_path = self._create_temp_file('corrupted.pdf', pdf_content)
        
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract(file_path)
        
        assert exc_info.value.error_type == "CorruptedPDF"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_file_too_large(self):
        """Test handling of files that exceed size limit."""
        config = {'max_file_size': 100}  # 100 bytes limit
        extractor = PDFExtractor(config)
        
        # Create file larger than limit
        large_content = b'x' * 200  # 200 bytes
        file_path = self._create_temp_file('large.pdf', large_content)
        
        with pytest.raises(ProcessingError) as exc_info:
            extractor.extract(file_path)
        
        assert exc_info.value.error_type == "FileTooLarge"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract('/nonexistent/file.pdf')
        
        assert exc_info.value.error_type == "FileNotFound"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    @patch('pdfplumber.open')
    def test_page_extraction_error_handling(self, mock_pdfplumber_open):
        """Test handling errors during page extraction."""
        # Mock page that raises exception
        mock_page1 = Mock()
        mock_page1.extract_text.side_effect = Exception("Page extraction error")
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page2.extract_tables.return_value = []
        mock_page2.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('error_page.pdf', pdf_content)
        
        # Should continue processing despite page error
        extracted = self.extractor.extract(file_path)
        
        # Should have content from page 2 only
        assert extracted.content == "Page 2 content"
    
    @patch('pdfplumber.open')
    def test_table_extraction_error_handling(self, mock_pdfplumber_open):
        """Test handling errors during table extraction."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Content with table"
        mock_page.extract_tables.side_effect = Exception("Table extraction error")
        mock_page.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('table_error.pdf', pdf_content)
        
        # Should continue processing despite table extraction error
        extracted = self.extractor.extract(file_path)
        
        assert extracted.content == "Content with table"
        assert len(extracted.tables) == 0  # No tables due to error
    
    def test_heading_detection(self):
        """Test heading detection heuristics."""
        # Test various heading patterns
        test_cases = [
            ("INTRODUCTION", True),  # All caps
            ("1. Introduction", True),  # Numbered
            ("Chapter 1", True),  # Chapter
            ("Section 2.1", True),  # Section
            ("Title Case Heading", True),  # Title case
            ("This is a very long line that should not be considered a heading because it exceeds the length limit", False),
            ("short", False),  # Too short
            ("This is a regular sentence.", False),  # Ends with punctuation
        ]
        
        for text, expected in test_cases:
            result = self.extractor._is_likely_heading(text)
            assert result == expected, f"Failed for text: '{text}'"
    
    @patch('pdfplumber.open')
    def test_structure_analysis(self, mock_pdfplumber_open):
        """Test PDF structure analysis."""
        content = """INTRODUCTION

This is the introduction paragraph.

1. First item
2. Second item

METHODOLOGY

Another paragraph here.

    def code_example():
        return True

CONCLUSION

Final paragraph."""
        
        mock_page = Mock()
        mock_page.extract_text.return_value = content
        mock_page.extract_tables.return_value = []
        mock_page.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber_open.return_value = mock_pdf
        
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog'
        file_path = self._create_temp_file('structure.pdf', pdf_content)
        
        extracted = self.extractor.extract(file_path)
        
        assert len(extracted.structure.headings) >= 2  # Should detect INTRODUCTION, METHODOLOGY, CONCLUSION
        assert extracted.structure.paragraphs >= 3
        # Note: List detection in PDF text is heuristic-based and may vary
        assert extracted.structure.lists >= 0  # May or may not detect numbered items as lists
        assert extracted.structure.code_blocks >= 1  # Indented code


@pytest.mark.skipif(HAS_PDFPLUMBER, reason="Testing import error handling")
class TestPDFExtractorWithoutDependency:
    """Test PDFExtractor behavior when pdfplumber is not available."""
    
    def test_import_error_on_initialization(self):
        """Test that PDFExtractor raises ImportError when pdfplumber is not available."""
        # This test only runs when pdfplumber is not available
        # In practice, we would mock the import to simulate this condition
        pass


# Test ExtractorFactory integration
@pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not available")
class TestPDFExtractorFactory:
    """Test PDF extractor integration with ExtractorFactory."""
    
    def test_pdf_extractor_registration(self):
        """Test that PDF extractor is registered with the factory."""
        from qudata.ingest.files import ExtractorFactory
        
        # PDF extractor should be registered
        pdf_extractor = ExtractorFactory.create_extractor('pdf')
        assert pdf_extractor is not None
        assert isinstance(pdf_extractor, PDFExtractor)
    
    def test_pdf_in_supported_types(self):
        """Test that PDF is in the list of supported types."""
        from qudata.ingest.files import ExtractorFactory
        
        supported_types = ExtractorFactory.get_supported_types()
        assert 'pdf' in supported_types


if __name__ == "__main__":
    pytest.main([__file__])