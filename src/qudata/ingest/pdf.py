"""
PDF extraction module for LLMDataForge.

This module provides PDF content extraction using pdfplumber for text and table extraction,
with robust error handling for corrupted files and structure preservation.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Optional dependency for PDF processing
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, TableData, ImageData
)


class PDFExtractor(BaseExtractor):
    """
    Extractor for PDF files using pdfplumber.
    
    Handles text extraction, table detection, and structure analysis
    with robust error handling for corrupted or problematic PDF files.
    """
    
    SUPPORTED_FORMATS = {'pdf'}
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PDFExtractor.
        
        Args:
            config: Configuration dictionary with options like:
                - max_file_size: Maximum file size to process (default: 500MB)
                - extract_tables: Whether to extract tables (default: True)
                - extract_images: Whether to extract image metadata (default: True)
                - preserve_layout: Whether to preserve text layout (default: False)
                - password: PDF password if encrypted (default: None)
        """
        super().__init__(config)
        
        if not HAS_PDFPLUMBER:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            )
        
        self.max_file_size = self.config.get('max_file_size', 500 * 1024 * 1024)  # 500MB
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)
        self.preserve_layout = self.config.get('preserve_layout', False)
        self.password = self.config.get('password', None)
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file to extract content from
            
        Returns:
            ExtractedContent object containing the extracted content and metadata
            
        Raises:
            ProcessingError: If extraction fails
        """
        try:
            # Validate file first
            self.validate_file(file_path)
            
            # Get basic file metadata
            file_metadata = self.get_metadata(file_path)
            
            # Extract content from PDF
            content, structure, tables, images = self._extract_pdf_content(file_path)
            
            # Create extracted content object
            extracted = ExtractedContent(content, file_metadata)
            extracted.structure = structure
            extracted.tables = tables
            extracted.images = images
            
            return extracted
            
        except ProcessingError:
            raise
        except Exception as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="PDFExtractionError",
                message=f"Failed to extract content from PDF {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def supports_format(self, file_type: str) -> bool:
        """
        Check if this extractor supports the given file type.
        
        Args:
            file_type: The file type to check
            
        Returns:
            True if the extractor supports this file type, False otherwise
        """
        return file_type.lower() in self.SUPPORTED_FORMATS
    
    def _extract_pdf_content(self, file_path: str) -> Tuple[str, DocumentStructure, List[TableData], List[ImageData]]:
        """
        Extract content, structure, tables, and images from PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (content, structure, tables, images)
            
        Raises:
            ProcessingError: If PDF cannot be processed
        """
        try:
            with pdfplumber.open(file_path, password=self.password) as pdf:
                # Check if PDF is encrypted and we don't have password
                try:
                    if hasattr(pdf, 'is_encrypted') and pdf.is_encrypted and not self.password:
                        # Skip encrypted PDFs gracefully - return empty content
                        empty_structure = DocumentStructure(
                            headings=[],
                            paragraphs=0,
                            tables=0,
                            images=0,
                            code_blocks=0,
                            lists=0
                        )
                        return "", empty_structure, [], []
                except AttributeError:
                    # is_encrypted not available in this version, continue
                    pass
                
                # Extract content from all pages
                all_text = []
                all_tables = []
                all_images = []
                page_count = len(pdf.pages)
                
                if page_count == 0:
                    raise self.create_processing_error(
                        stage="extraction",
                        error_type="EmptyPDF",
                        message=f"PDF file {file_path} contains no pages",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text from page
                        if self.preserve_layout:
                            page_text = page.extract_text(layout=True)
                        else:
                            page_text = page.extract_text()
                        
                        if page_text:
                            all_text.append(page_text)
                        
                        # Extract tables if enabled
                        if self.extract_tables:
                            page_tables = self._extract_tables_from_page(page, page_num)
                            all_tables.extend(page_tables)
                        
                        # Extract image metadata if enabled
                        if self.extract_images:
                            page_images = self._extract_images_from_page(page, page_num)
                            all_images.extend(page_images)
                            
                    except Exception as e:
                        # Log page-level errors but continue processing
                        print(f"Warning: Error extracting page {page_num}: {str(e)}")
                        continue
                
                # Combine all text
                full_text = '\n\n'.join(all_text) if all_text else ""
                
                # Analyze document structure
                structure = self._analyze_pdf_structure(full_text, all_tables, all_images, page_count)
                
                return full_text, structure, all_tables, all_images
                
        except FileNotFoundError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="FileNotFound",
                message=f"PDF file not found: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except PermissionError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="PermissionError",
                message=f"Permission denied accessing PDF file: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except Exception as e:
            # Handle various PDF parsing errors
            error_message = str(e)
            if "syntax" in error_message.lower() or "corrupt" in error_message.lower():
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="CorruptedPDF",
                    message=f"PDF file {file_path} appears to be corrupted: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    stack_trace=str(e)
                )
            elif "parsing" in error_message.lower() or "parser" in error_message.lower():
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="PDFParsingError",
                    message=f"Error parsing PDF file {file_path}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    stack_trace=str(e)
                )
            else:
                # Re-raise as generic extraction error
                raise
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[TableData]:
        """
        Extract tables from a PDF page.
        
        Args:
            page: pdfplumber page object
            page_num: Page number for reference
            
        Returns:
            List of TableData objects
        """
        tables = []
        
        try:
            # Extract tables using pdfplumber's table detection
            page_tables = page.extract_tables()
            
            for table_idx, table in enumerate(page_tables):
                if table and len(table) > 0:
                    # First row as headers, rest as data
                    headers = table[0] if table[0] else []
                    rows = table[1:] if len(table) > 1 else []
                    
                    # Clean up None values and convert to strings
                    clean_headers = [str(cell) if cell is not None else "" for cell in headers]
                    clean_rows = []
                    for row in rows:
                        clean_row = [str(cell) if cell is not None else "" for cell in row]
                        clean_rows.append(clean_row)
                    
                    # Only add table if it has meaningful content
                    if clean_headers and any(header.strip() for header in clean_headers):
                        table_data = TableData(
                            headers=clean_headers,
                            rows=clean_rows,
                            caption=f"Table {table_idx + 1} from page {page_num}"
                        )
                        tables.append(table_data)
                        
        except Exception as e:
            # Log table extraction errors but don't fail the entire extraction
            print(f"Warning: Error extracting tables from page {page_num}: {str(e)}")
        
        return tables
    
    def _extract_images_from_page(self, page, page_num: int) -> List[ImageData]:
        """
        Extract image metadata from a PDF page.
        
        Args:
            page: pdfplumber page object
            page_num: Page number for reference
            
        Returns:
            List of ImageData objects
        """
        images = []
        
        try:
            # Get image objects from the page
            if hasattr(page, 'images'):
                for img_idx, img in enumerate(page.images):
                    # Create image metadata
                    image_data = ImageData(
                        path=f"page_{page_num}_image_{img_idx + 1}",
                        caption=f"Image {img_idx + 1} from page {page_num}",
                        alt_text=f"Image extracted from PDF page {page_num}"
                    )
                    images.append(image_data)
                    
        except Exception as e:
            # Log image extraction errors but don't fail the entire extraction
            print(f"Warning: Error extracting images from page {page_num}: {str(e)}")
        
        return images
    
    def _analyze_pdf_structure(self, text: str, tables: List[TableData], 
                              images: List[ImageData], page_count: int) -> DocumentStructure:
        """
        Analyze the structure of extracted PDF content.
        
        Args:
            text: Extracted text content
            tables: Extracted tables
            images: Extracted images
            page_count: Number of pages in PDF
            
        Returns:
            DocumentStructure object with analysis results
        """
        if not text:
            return DocumentStructure(
                headings=[],
                paragraphs=0,
                tables=len(tables),
                images=len(images),
                code_blocks=0,
                lists=0
            )
        
        lines = text.split('\n')
        
        # Find potential headings (lines that are short, capitalized, or have specific patterns)
        headings = []
        paragraphs = 0
        lists = 0
        code_blocks = 0
        
        in_paragraph = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                in_paragraph = False
                continue
            
            # Detect headings (various heuristics for PDF text)
            if self._is_likely_heading(line_stripped):
                headings.append(line_stripped)
                continue
            
            # Detect lists
            if re.match(r'^\s*[-•*]\s+', line) or re.match(r'^\s*\d+[\.\)]\s+', line):
                lists += 1
                continue
            
            # Detect code blocks (lines with consistent indentation or special characters)
            if re.match(r'^\s{4,}', line) or re.match(r'^[{}();]+', line_stripped):
                code_blocks += 1
                continue
            
            # Count paragraphs
            if not in_paragraph and len(line_stripped) > 10:
                paragraphs += 1
                in_paragraph = True
        
        return DocumentStructure(
            headings=headings,
            paragraphs=paragraphs,
            tables=len(tables),
            images=len(images),
            code_blocks=code_blocks,
            lists=lists
        )
    
    def _is_likely_heading(self, line: str) -> bool:
        """
        Determine if a line is likely a heading based on various heuristics.
        
        Args:
            line: Text line to analyze
            
        Returns:
            True if line is likely a heading
        """
        # Skip very long lines
        if len(line) > 100:
            return False
        
        # Skip very short lines
        if len(line) < 3:
            return False
        
        # Check for common heading patterns
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # Numbered headings (1. Introduction)
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title Case
            r'^Chapter\s+\d+',   # Chapter headings
            r'^Section\s+\d+',   # Section headings
            r'^Appendix\s+[A-Z]', # Appendix headings
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True
        
        # Check if line is mostly uppercase and not too long
        if line.isupper() and 5 <= len(line) <= 50:
            return True
        
        # Check if line ends without punctuation (except colon)
        if not line.endswith(('.', '!', '?', ',', ';')) and len(line) <= 80:
            # Check if it's title case
            words = line.split()
            if len(words) >= 2 and all(word[0].isupper() if word else False for word in words):
                return True
        
        return False