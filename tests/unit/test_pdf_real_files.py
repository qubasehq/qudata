"""
Tests for PDF extraction using real PDF files.

Tests the PDF extractor with actual PDF files instead of mocks.
"""

import os
import sys
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Check if pdfplumber is available
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from qudata.ingest.pdf import PDFExtractor
from qudata.models import ProcessingError


@pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not available")
class TestPDFExtractionRealFiles:
    """Test PDF extraction with real PDF files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor()
        self.sample_dir = Path(__file__).parent.parent / "sample_data" / "pdfs"
    
    def test_extract_simple_pdf(self):
        """Test extracting content from simple PDF."""
        pdf_file = self.sample_dir / "sample_simple.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        extracted = self.extractor.extract(str(pdf_file))
        
        assert extracted.content is not None
        assert len(extracted.content) > 0
        assert "Sample PDF Document" in extracted.content
        assert extracted.metadata.file_type == 'pdf'
        assert extracted.structure is not None
        assert extracted.structure.paragraphs >= 1
    
    def test_extract_table_pdf(self):
        """Test extracting PDF with tables."""
        pdf_file = self.sample_dir / "sample_table.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        extracted = self.extractor.extract(str(pdf_file))
        
        assert extracted.content is not None
        assert len(extracted.content) > 0
        assert len(extracted.tables) >= 1
        
        # Check table structure
        table = extracted.tables[0]
        assert len(table.headers) >= 3  # Should have Name, Age, City, Occupation
        assert len(table.rows) >= 3     # Should have data rows
        assert extracted.structure.tables >= 1
    
    def test_extract_multipage_pdf(self):
        """Test extracting multi-page PDF."""
        pdf_file = self.sample_dir / "sample_multipage.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        extracted = self.extractor.extract(str(pdf_file))
        
        assert extracted.content is not None
        assert len(extracted.content) > 0
        assert "Page 1" in extracted.content
        assert "Page 2" in extracted.content
        assert "Page 3" in extracted.content
        assert extracted.structure.paragraphs >= 3
    
    def test_extract_structured_pdf(self):
        """Test extracting structured PDF with headings."""
        pdf_file = self.sample_dir / "sample_structured.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        extracted = self.extractor.extract(str(pdf_file))
        
        assert extracted.content is not None
        assert len(extracted.content) > 0
        assert "STRUCTURED DOCUMENT" in extracted.content
        assert "INTRODUCTION" in extracted.content
        assert len(extracted.structure.headings) >= 3
    
    def test_extract_corrupted_pdf(self):
        """Test handling corrupted PDF file."""
        pdf_file = self.sample_dir / "sample_corrupted.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        # Should raise ProcessingError for corrupted file
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract(str(pdf_file))
        
        assert exc_info.value.severity.value in ['high', 'medium']
    
    def test_pdf_extractor_configuration(self):
        """Test PDF extractor with different configurations."""
        pdf_file = self.sample_dir / "sample_simple.pdf"
        
        if not pdf_file.exists():
            pytest.skip(f"Sample PDF not found: {pdf_file}")
        
        # Test with tables disabled
        config = {'extract_tables': False, 'extract_images': False}
        extractor = PDFExtractor(config)
        extracted = extractor.extract(str(pdf_file))
        
        assert len(extracted.tables) == 0
        assert len(extracted.images) == 0
        assert len(extracted.content) > 0
    
    def test_pdf_supports_format(self):
        """Test PDF format support."""
        assert self.extractor.supports_format('pdf') is True
        assert self.extractor.supports_format('PDF') is True
        assert self.extractor.supports_format('txt') is False
    
    def test_pdf_file_not_found(self):
        """Test handling non-existent PDF file."""
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract('/nonexistent/file.pdf')
        
        assert exc_info.value.error_type == "FileNotFound"
        assert exc_info.value.severity.value == "high"


if __name__ == "__main__":
    pytest.main([__file__])