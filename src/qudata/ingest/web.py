"""
Web content extraction module for LLMDataForge.

This module provides extraction capabilities for HTML files using BeautifulSoup4
with readability-based content extraction and semantic structure preservation.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Required dependency for HTML processing
try:
    from bs4 import BeautifulSoup, Comment
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Optional dependency for readability
try:
    from readability import Document as ReadabilityDocument
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, TableData, ImageData, parse_file_size
)


class WebExtractor(BaseExtractor):
    """
    Extractor for HTML files using BeautifulSoup4 with readability.
    
    Handles content extraction with noise removal, table detection,
    image metadata extraction, and semantic structure preservation.
    """
    
    SUPPORTED_FORMATS = {'html', 'htm', 'xhtml'}
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the WebExtractor.
        
        Args:
            config: Configuration dictionary with options like:
                - max_file_size: Maximum file size to process (default: 50MB)
                - extract_tables: Whether to extract tables (default: True)
                - extract_images: Whether to extract image metadata (default: True)
                - use_readability: Whether to use readability for content extraction (default: True)
                - preserve_links: Whether to preserve link information (default: True)
                - remove_scripts: Whether to remove script tags (default: True)
                - remove_styles: Whether to remove style tags (default: True)
        """
        super().__init__(config)
        
        if not HAS_BS4:
            raise ImportError(
                "beautifulsoup4 is required for HTML extraction. "
                "Install it with: pip install beautifulsoup4"
            )
        
        self.max_file_size = parse_file_size(self.config.get('max_file_size', 50 * 1024 * 1024))  # 50MB
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)
        self.use_readability = self.config.get('use_readability', True) and HAS_READABILITY
        self.preserve_links = self.config.get('preserve_links', True)
        self.remove_scripts = self.config.get('remove_scripts', True)
        self.remove_styles = self.config.get('remove_styles', True)
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from an HTML file.
        
        Args:
            file_path: Path to the HTML file to extract content from
            
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
            
            # Check file size
            if file_metadata.size_bytes > self.max_file_size:
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="FileTooLarge",
                    message=f"File size ({file_metadata.size_bytes} bytes) exceeds maximum ({self.max_file_size} bytes)",
                    severity=ErrorSeverity.HIGH
                )
            
            # Extract content from HTML
            content, structure, tables, images = self._extract_html_content(file_path)
            
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
                error_type="HTMLExtractionError",
                message=f"Failed to extract content from HTML {file_path}: {str(e)}",
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
    
    def _extract_html_content(self, file_path: str) -> Tuple[str, DocumentStructure, List[TableData], List[ImageData]]:
        """
        Extract content, structure, tables, and images from HTML.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Tuple of (content, structure, tables, images)
            
        Raises:
            ProcessingError: If HTML cannot be processed
        """
        try:
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Use readability if available and enabled
            if self.use_readability:
                try:
                    doc = ReadabilityDocument(html_content)
                    main_content = doc.summary()
                    title = doc.title()
                    
                    # Parse the cleaned content
                    soup = BeautifulSoup(main_content, 'html.parser')
                    
                    # Also parse original for metadata extraction
                    original_soup = BeautifulSoup(html_content, 'html.parser')
                    
                except Exception as e:
                    print(f"Warning: Readability extraction failed, falling back to BeautifulSoup: {str(e)}")
                    soup = BeautifulSoup(html_content, 'html.parser')
                    original_soup = soup
                    title = self._extract_title(soup)
            else:
                soup = BeautifulSoup(html_content, 'html.parser')
                original_soup = soup
                title = self._extract_title(soup)
            
            # Clean the soup
            cleaned_soup = self._clean_html(soup)
            
            # Extract content and structure
            content, structure = self._extract_content_and_structure(cleaned_soup, title)
            
            # Extract tables if enabled
            tables = []
            if self.extract_tables:
                tables = self._extract_tables(cleaned_soup)
            
            # Extract images if enabled
            images = []
            if self.extract_images:
                images = self._extract_images(original_soup, file_path)
            
            return content, structure, tables, images
            
        except FileNotFoundError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="FileNotFound",
                message=f"HTML file not found: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except PermissionError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="PermissionError",
                message=f"Permission denied accessing HTML file: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except UnicodeDecodeError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="EncodingError",
                message=f"Could not decode HTML file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
        except Exception as e:
            # Handle various HTML parsing errors
            error_message = str(e)
            if "parser" in error_message.lower():
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="HTMLParsingError",
                    message=f"Error parsing HTML file {file_path}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    stack_trace=str(e)
                )
            else:
                # Re-raise as generic extraction error
                raise
    
    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Clean HTML by removing unwanted elements.
        
        Args:
            soup: BeautifulSoup object to clean
            
        Returns:
            Cleaned BeautifulSoup object
        """
        # Remove script tags if configured
        if self.remove_scripts:
            for script in soup.find_all('script'):
                script.decompose()
        
        # Remove style tags if configured
        if self.remove_styles:
            for style in soup.find_all('style'):
                style.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove common noise elements
        noise_selectors = [
            'nav', 'footer', 'aside', '.sidebar', '.navigation',
            '.menu', '.ads', '.advertisement', '.social', '.share',
            '.comments', '.comment-form', '.breadcrumb', '.pagination'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove elements with common noise classes/ids
        noise_patterns = [
            'ad', 'ads', 'advertisement', 'banner', 'popup', 'modal',
            'cookie', 'newsletter', 'subscription', 'social', 'share'
        ]
        
        for pattern in noise_patterns:
            # Remove by class
            for element in soup.find_all(class_=lambda x: x and pattern in ' '.join(x).lower()):
                element.decompose()
            
            # Remove by id
            for element in soup.find_all(id=lambda x: x and pattern in x.lower()):
                element.decompose()
        
        return soup
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract title from HTML document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Document title
        """
        # Try title tag first
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            return title_tag.get_text().strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.get_text().strip():
            return h1_tag.get_text().strip()
        
        # Try meta title
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title.get('content').strip()
        
        return "Untitled Document"
    
    def _extract_content_and_structure(self, soup: BeautifulSoup, title: str) -> Tuple[str, DocumentStructure]:
        """
        Extract main content and analyze document structure.
        
        Args:
            soup: Cleaned BeautifulSoup object
            title: Document title
            
        Returns:
            Tuple of (content, structure)
        """
        content_parts = []
        headings = []
        paragraphs = 0
        lists = 0
        code_blocks = 0
        links = 0
        
        # Add title if available
        if title and title != "Untitled Document":
            content_parts.append(f"# {title}")
            headings.append(title)
        
        # Add script and style content if preservation is enabled
        if not self.remove_scripts:
            for script in soup.find_all('script'):
                script_text = script.get_text().strip()
                if script_text:
                    content_parts.append(f"[Script: {script_text}]")
        
        if not self.remove_styles:
            for style in soup.find_all('style'):
                style_text = style.get_text().strip()
                if style_text:
                    content_parts.append(f"[Style: {style_text}]")
        
        # Process elements in document order
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 
                                     'ul', 'ol', 'li', 'pre', 'code', 'blockquote', 'article']):
            
            text = element.get_text().strip()
            if not text:
                continue
            
            # Handle headings
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                heading_prefix = '#' * level
                content_parts.append(f"{heading_prefix} {text}")
                headings.append(text)
            
            # Handle paragraphs and divs
            elif element.name in ['p', 'div', 'article']:
                # Skip if this is just a container with other block elements
                if element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol']):
                    continue
                
                content_parts.append(text)
                paragraphs += 1
                
                # Count links in this paragraph
                if self.preserve_links:
                    paragraph_links = element.find_all('a', href=True)
                    links += len(paragraph_links)
            
            # Handle lists
            elif element.name in ['ul', 'ol']:
                list_items = element.find_all('li', recursive=False)
                if list_items:
                    list_content = []
                    for i, li in enumerate(list_items):
                        li_text = li.get_text().strip()
                        if li_text:
                            if element.name == 'ol':
                                list_content.append(f"{i+1}. {li_text}")
                            else:
                                list_content.append(f"- {li_text}")
                    
                    if list_content:
                        content_parts.append('\n'.join(list_content))
                        lists += len(list_items)
            
            # Handle individual list items (if not processed as part of a list)
            elif element.name == 'li':
                # Check if parent is already processed
                if element.parent and element.parent.name not in ['ul', 'ol']:
                    content_parts.append(f"- {text}")
                    lists += 1
            
            # Handle code blocks
            elif element.name in ['pre', 'code']:
                if element.name == 'pre':
                    content_parts.append(f"```\n{text}\n```")
                    code_blocks += 1
                else:
                    # Inline code
                    content_parts.append(f"`{text}`")
            
            # Handle blockquotes
            elif element.name == 'blockquote':
                quoted_lines = [f"> {line}" for line in text.split('\n') if line.strip()]
                content_parts.append('\n'.join(quoted_lines))
                paragraphs += 1
        
        # Combine all content
        full_content = '\n\n'.join(content_parts) if content_parts else ""
        
        # Create document structure
        structure = DocumentStructure(
            headings=headings,
            paragraphs=paragraphs,
            tables=0,  # Will be set by table extraction
            images=0,  # Will be set by image extraction
            code_blocks=code_blocks,
            lists=lists,
            links=links
        )
        
        return full_content, structure
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[TableData]:
        """
        Extract tables from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of TableData objects
        """
        tables = []
        
        for table_idx, table in enumerate(soup.find_all('table'), 1):
            try:
                headers = []
                rows = []
                
                # Extract headers from thead or first tr
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                else:
                    # Try first row as headers
                    first_row = table.find('tr')
                    if first_row:
                        first_row_cells = first_row.find_all(['th', 'td'])
                        # If first row has th elements, use as headers
                        if first_row.find('th'):
                            headers = [th.get_text().strip() for th in first_row_cells]
                
                # Extract data rows from tbody or remaining tr elements
                tbody = table.find('tbody')
                if tbody:
                    data_rows = tbody.find_all('tr')
                else:
                    data_rows = table.find_all('tr')
                    # Skip header row if we found headers
                    if headers and data_rows:
                        data_rows = data_rows[1:]
                
                for row in data_rows:
                    row_data = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    if row_data:  # Only add non-empty rows
                        rows.append(row_data)
                
                # Extract table caption
                caption_elem = table.find('caption')
                caption = caption_elem.get_text().strip() if caption_elem else f"Table {table_idx}"
                
                # Only add table if it has meaningful content
                if headers or rows:
                    table_data = TableData(
                        headers=headers,
                        rows=rows,
                        caption=caption
                    )
                    tables.append(table_data)
                    
            except Exception as e:
                # Log table extraction errors but don't fail the entire extraction
                print(f"Warning: Error extracting table {table_idx}: {str(e)}")
                continue
        
        return tables
    
    def _extract_images(self, soup: BeautifulSoup, base_path: str) -> List[ImageData]:
        """
        Extract image metadata from HTML.
        
        Args:
            soup: BeautifulSoup object
            base_path: Base path for resolving relative URLs
            
        Returns:
            List of ImageData objects
        """
        images = []
        
        for img_idx, img in enumerate(soup.find_all('img'), 1):
            try:
                src = img.get('src', '')
                alt = img.get('alt', '')
                title = img.get('title', '')
                width = img.get('width')
                height = img.get('height')
                
                # Convert width/height to integers if possible
                try:
                    if width is not None:
                        # Handle string values that might contain 'px' or other units
                        if isinstance(width, str):
                            width = re.sub(r'[^\d]', '', width)
                        width = int(width) if width else None
                    else:
                        width = None
                except (ValueError, TypeError):
                    width = None
                
                try:
                    if height is not None:
                        # Handle string values that might contain 'px' or other units
                        if isinstance(height, str):
                            height = re.sub(r'[^\d]', '', height)
                        height = int(height) if height else None
                    else:
                        height = None
                except (ValueError, TypeError):
                    height = None
                
                # Resolve relative URLs
                if src and not src.startswith(('http://', 'https://', 'data:')):
                    # For file paths, just keep the relative path
                    if not src.startswith('/'):
                        src = os.path.join(os.path.dirname(base_path), src)
                
                image_data = ImageData(
                    path=src or f"image_{img_idx}",
                    caption=title or alt or f"Image {img_idx}",
                    alt_text=alt,
                    width=width,
                    height=height
                )
                images.append(image_data)
                
            except Exception as e:
                # Log image extraction errors but don't fail the entire extraction
                print(f"Warning: Error extracting image {img_idx}: {str(e)}")
                continue
        
        return images