"""
File content extractors for LLMDataForge.

This module contains concrete implementations of file extractors for various
file formats, starting with plain text and markdown files.
"""

import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

# Optional dependency for encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, parse_file_size
)


class PlainTextExtractor(BaseExtractor):
    """
    Extractor for plain text files (.txt) and markdown files (.md).
    
    Handles encoding detection, content extraction, and basic structure analysis
    for text-based files.
    """
    
    SUPPORTED_FORMATS = {'txt', 'text', 'markdown', 'md', 'mdown', 'mkd'}
    
    # Common encoding fallbacks
    ENCODING_FALLBACKS = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PlainTextExtractor.
        
        Args:
            config: Configuration dictionary with options like:
                - max_file_size: Maximum file size to process (default: 10MB)
                - encoding_detection: Whether to use automatic encoding detection
                - preserve_whitespace: Whether to preserve original whitespace
        """
        super().__init__(config)
        self.max_file_size = parse_file_size(self.config.get('max_file_size', 10 * 1024 * 1024))  # 10MB default
        self.encoding_detection = self.config.get('encoding_detection', True)
        self.preserve_whitespace = self.config.get('preserve_whitespace', False)

    def _coerce_size(self, value: Any) -> int:
        """Convert size config to integer bytes. Accepts int or strings like '100MB', '50kb'."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip().lower()
            try:
                # Pure number string
                return int(float(s))
            except ValueError:
                pass
            # Parse with unit suffix
            units = {
                'b': 1,
                'kb': 1024,
                'k': 1024,
                'mb': 1024**2,
                'm': 1024**2,
                'gb': 1024**3,
                'g': 1024**3,
                'tb': 1024**4,
                't': 1024**4,
            }
            # split number and unit
            import re
            m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s*$", s)
            if m:
                num = float(m.group(1))
                unit = m.group(2)
                factor = units.get(unit, None)
                if factor is not None:
                    return int(num * factor)
        # Fallback: default 100MB
        return 100 * 1024 * 1024
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from a plain text or markdown file.
        
        Args:
            file_path: Path to the file to extract content from
            
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
            
            # Detect encoding and read content
            content = self._read_file_content(file_path)
            
            # If markdown, normalize markdown syntax to readable plain text
            if file_metadata.file_type in ['markdown', 'md', 'mdown', 'mkd']:
                content = self._markdown_to_text(content)

            # Create extracted content object
            extracted = ExtractedContent(content, file_metadata)
            
            # Analyze structure based on file type
            if file_metadata.file_type in ['markdown', 'md', 'mdown', 'mkd']:
                extracted.structure = self._analyze_markdown_structure(content)
            else:
                extracted.structure = self._analyze_text_structure(content)
            
            return extracted
            
        except ProcessingError:
            raise
        except Exception as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="ExtractionError",
                message=f"Failed to extract content from {file_path}: {str(e)}",
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
    
    def _read_file_content(self, file_path: str) -> str:
        """
        Read file content with encoding detection.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            ProcessingError: If file cannot be read
        """
        encoding = 'utf-8'  # Default encoding
        
        try:
            # Detect encoding if enabled
            if self.encoding_detection:
                encoding = self._detect_encoding(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Normalize whitespace if not preserving original
            if not self.preserve_whitespace:
                content = self._normalize_whitespace(content)
            
            return content
            
        except UnicodeDecodeError as e:
            # Try fallback encodings
            for fallback_encoding in self.ENCODING_FALLBACKS:
                if fallback_encoding != encoding:
                    try:
                        with open(file_path, 'r', encoding=fallback_encoding, errors='replace') as f:
                            content = f.read()
                        
                        if not self.preserve_whitespace:
                            content = self._normalize_whitespace(content)
                        
                        return content
                    except UnicodeDecodeError:
                        continue
            
            raise self.create_processing_error(
                stage="extraction",
                error_type="EncodingError",
                message=f"Could not decode file {file_path} with any supported encoding",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
        
        except IOError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="FileReadError",
                message=f"Could not read file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet library if available.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding string
        """
        if not HAS_CHARDET:
            # If chardet is not available, default to UTF-8
            return 'utf-8'
        
        try:
            with open(file_path, 'rb') as f:
                # Read a sample of the file for encoding detection
                sample_size = min(32768, os.path.getsize(file_path))  # 32KB or file size
                raw_data = f.read(sample_size)
            
            # Use chardet to detect encoding
            result = chardet.detect(raw_data)
            
            if result and result['encoding'] and result['confidence'] > 0.7:
                return result['encoding']
            else:
                # Fall back to UTF-8 if detection is uncertain
                return 'utf-8'
                
        except Exception:
            # If detection fails, default to UTF-8
            return 'utf-8'
    
    def _normalize_whitespace(self, content: str) -> str:
        """
        Normalize whitespace in content.
        
        Args:
            content: Raw content string
            
        Returns:
            Content with normalized whitespace
        """
        # Replace multiple consecutive whitespace with single space
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Normalize line endings to \n
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _markdown_to_text(self, content: str) -> str:
        """
        Convert common Markdown syntax to clean plain text with sensible spacing.
        This is a lightweight transform to avoid adding heavy deps.
        """
        text = content
        # Normalize line endings first
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove front-matter fences if present (--- ... --- at top)
        text = re.sub(r'^---\n[\s\S]*?\n---\n', '', text, flags=re.MULTILINE)

        # Headings: ensure blank line before and after, drop leading #
        def _heading_repl(m):
            title = m.group(2).strip()
            return f"\n\n{title}\n\n"
        text = re.sub(r'^(#{1,6})\s+(.*)$', _heading_repl, text, flags=re.MULTILINE)

        # Images: ![alt](url) -> alt
        text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)

        # Links: [text](url) -> text
        text = re.sub(r'\[([^\]]+)]\([^)]*\)', r'\1', text)

        # Inline code: `code` -> code
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Bold/italic: **text** or *text* or __text__ or _text_ -> text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Code fences: remove the ```lang fences but keep code content
        text = re.sub(r'^```.*$\n?', '', text, flags=re.MULTILINE)

        # Lists: ensure a single space after dash/asterisk and keep one item per line
        text = re.sub(r'^(\s*)[-*]\s*', r'\1- ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)\d+\.[\t ]*', r'\1- ', text, flags=re.MULTILINE)

        # Blockquotes: > text -> text
        text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)

        # Tables: drop Markdown table pipes but keep cell text separated by spaces
        # Replace table header separators like |---|---|
        text = re.sub(r'^\s*\|?[\s:\-\|]+\|\s*$', '', text, flags=re.MULTILINE)
        # Replace pipes with a single space
        text = re.sub(r'\s*\|\s*', ' ', text)

        # Remove residual HTML entities lightly
        text = re.sub(r'&nbsp;?', ' ', text)

        # Collapse excessive internal spaces but preserve paragraph breaks
        # First collapse spaces/tabs within lines
        text = re.sub(r'[ \t]+', ' ', text)
        # Trim spaces at line ends
        text = re.sub(r'[ \t]+\n', '\n', text)
        # Collapse 3+ newlines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace
        return text.strip()
    
    def _analyze_text_structure(self, content: str) -> DocumentStructure:
        """
        Analyze structure of plain text content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            DocumentStructure object with analysis results
        """
        lines = content.split('\n')
        
        # Count paragraphs (non-empty lines or groups of lines)
        paragraphs = 0
        in_paragraph = False
        
        for line in lines:
            line = line.strip()
            if line:
                if not in_paragraph:
                    paragraphs += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        
        # Look for list patterns
        list_patterns = [
            r'^\s*[-*+]\s+',  # Bullet lists
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
        ]
        
        lists = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    lists += 1
                    break
        
        # Look for potential headings (lines that are short and followed by empty line)
        headings = []
        for i, line in enumerate(lines):
            line = line.strip()
            if (line and len(line) < 100 and len(line) > 5 and
                i < len(lines) - 1 and not lines[i + 1].strip() and
                not any(pattern in line for pattern in ['-', '*', '+', '.']) and
                not line.endswith('.') and not line.endswith(',')):
                # Potential heading - short line followed by empty line, not ending with punctuation
                headings.append(line)
        
        return DocumentStructure(
            headings=headings,
            paragraphs=paragraphs,
            tables=0,  # Plain text doesn't have structured tables
            images=0,  # Plain text doesn't have images
            code_blocks=0,  # Will be detected in markdown
            lists=lists
        )
    
    def _analyze_markdown_structure(self, content: str) -> DocumentStructure:
        """
        Analyze structure of markdown content.
        
        Args:
            content: Markdown content to analyze
            
        Returns:
            DocumentStructure object with analysis results
        """
        lines = content.split('\n')
        
        headings = []
        paragraphs = 0
        code_blocks = 0
        lists = 0
        tables = 0
        images = 0
        links = 0
        
        in_paragraph = False
        in_code_block = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                in_paragraph = False
                continue
            
            # Code blocks
            if line_stripped.startswith('```'):
                if in_code_block:
                    in_code_block = False
                else:
                    code_blocks += 1
                    in_code_block = True
                continue
            
            # Skip content inside code blocks
            if in_code_block:
                continue
            
            # Headings
            if line_stripped.startswith('#'):
                heading_match = re.match(r'^(#{1,6})\s+(.+)', line_stripped)
                if heading_match:
                    headings.append(heading_match.group(2))
                continue
            
            # Tables (simple detection) - only count header rows or first occurrence
            if '|' in line_stripped and line_stripped.count('|') >= 2:
                # Check if this is a table separator line (|---|---|)
                if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line_stripped):
                    # This is a table separator, don't count it
                    continue
                # Check if previous line was also a table to avoid double counting
                prev_line_idx = lines.index(line) - 1 if line in lines else -1
                if (prev_line_idx >= 0 and prev_line_idx < len(lines) and 
                    '|' in lines[prev_line_idx].strip()):
                    # Previous line was also a table, don't count this row
                    continue
                tables += 1
                continue
            
            # Lists
            list_patterns = [
                r'^\s*[-*+]\s+',  # Bullet lists
                r'^\s*\d+\.\s+',  # Numbered lists
            ]
            
            is_list = False
            for pattern in list_patterns:
                if re.match(pattern, line):
                    lists += 1
                    is_list = True
                    break
            
            if is_list:
                continue
            
            # Images
            if re.search(r'!\[.*?\]\(.*?\)', line_stripped):
                images += 1
            
            # Links
            links += len(re.findall(r'\[.*?\]\(.*?\)', line_stripped))
            
            # Regular paragraphs
            if not in_paragraph:
                paragraphs += 1
                in_paragraph = True
        
        return DocumentStructure(
            headings=headings,
            paragraphs=paragraphs,
            tables=tables,
            images=images,
            code_blocks=code_blocks,
            lists=lists,
            links=links
        )


class ExtractorFactory:
    """
    Factory class for creating appropriate extractors based on file type.
    """
    
    _extractors = {
        'txt': PlainTextExtractor,
        'text': PlainTextExtractor,
        'markdown': PlainTextExtractor,
        'md': PlainTextExtractor,
        'mdown': PlainTextExtractor,
        'mkd': PlainTextExtractor,
    }
    
    @classmethod
    def _register_pdf_extractor(cls):
        """Register PDF extractor if available."""
        try:
            from .pdf import PDFExtractor
            cls._extractors['pdf'] = PDFExtractor
        except ImportError:
            pass  # PDF extractor not available
    
    @classmethod
    def _register_document_extractor(cls):
        """Register Document extractor if available."""
        try:
            from .document import DocumentExtractor
            cls._extractors['docx'] = DocumentExtractor
            cls._extractors['doc'] = DocumentExtractor
        except ImportError:
            pass  # Document extractor not available
    
    @classmethod
    def _register_web_extractor(cls):
        """Register Web extractor if available."""
        try:
            from .web import WebExtractor
            cls._extractors['html'] = WebExtractor
            cls._extractors['htm'] = WebExtractor
            cls._extractors['xhtml'] = WebExtractor
        except ImportError:
            pass  # Web extractor not available
    
    @classmethod
    def _register_structured_extractor(cls):
        """Register Structured extractor if available."""
        try:
            from .structured import StructuredExtractor
            cls._extractors['csv'] = StructuredExtractor
            cls._extractors['json'] = StructuredExtractor
            cls._extractors['jsonl'] = StructuredExtractor
            cls._extractors['tsv'] = StructuredExtractor
        except ImportError:
            pass  # Structured extractor not available
    
    @classmethod
    def _register_image_extractor(cls):
        """Register Image extractor for SVG files."""
        try:
            from .image import ImageExtractor
            cls._extractors['svg'] = ImageExtractor
        except ImportError:
            pass  # Image extractor not available
    
    # Auto-register extractors when module is loaded
    @classmethod
    def _auto_register_extractors(cls):
        """Auto-register available extractors."""
        cls._register_pdf_extractor()
        cls._register_document_extractor()
        cls._register_web_extractor()
        cls._register_structured_extractor()
        cls._register_image_extractor()
    
    @classmethod
    def create_extractor(cls, file_type: str, config: Dict[str, Any] = None) -> Optional[BaseExtractor]:
        """
        Create an appropriate extractor for the given file type.
        
        Args:
            file_type: The file type to create an extractor for
            config: Configuration dictionary for the extractor
            
        Returns:
            BaseExtractor instance or None if no extractor is available
        """
        extractor_class = cls._extractors.get(file_type.lower())
        if extractor_class:
            return extractor_class(config)
        return None
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """
        Get list of supported file types.
        
        Returns:
            List of supported file type strings
        """
        return list(cls._extractors.keys())
    
    @classmethod
    def register_extractor(cls, file_type: str, extractor_class: type):
        """
        Register a new extractor for a file type.
        
        Args:
            file_type: File type string
            extractor_class: BaseExtractor subclass
        """
        cls._extractors[file_type.lower()] = extractor_class


# Auto-register extractors when module is loaded
ExtractorFactory._auto_register_extractors()
