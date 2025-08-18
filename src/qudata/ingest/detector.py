"""
File type detection module for LLMDataForge.

This module provides file type detection using file signatures (magic numbers)
and file extensions to accurately identify file formats for processing.
"""

import os
import mimetypes
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FileTypeDetector:
    """
    Detects file types using file signatures and extensions.
    
    Uses a combination of magic number detection and file extension
    analysis to accurately identify file formats.
    """
    
    # File signatures (magic numbers) for common file types
    FILE_SIGNATURES = {
        # PDF files
        b'\x25\x50\x44\x46': 'pdf',  # %PDF
        
        # Microsoft Office files (ZIP-based)
        b'\x50\x4B\x03\x04': 'office_zip',  # PK.. (ZIP header)
        b'\x50\x4B\x05\x06': 'office_zip',  # PK.. (ZIP empty archive)
        b'\x50\x4B\x07\x08': 'office_zip',  # PK.. (ZIP spanned archive)
        
        # Legacy Microsoft Office files
        b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1': 'office_legacy',  # OLE2 header
        
        # HTML files
        b'<!DOCTYPE html': 'html',
        b'<!doctype html': 'html',
        b'<html': 'html',
        b'<HTML': 'html',
        
        # XML files
        b'<?xml': 'xml',
        
        # JSON files (common patterns)
        b'{\n': 'json',
        b'{\r\n': 'json',
        b'{"': 'json',
        b'[\n': 'json',
        b'[\r\n': 'json',
        b'[{': 'json',
        
        # RTF files
        b'{\\rtf1': 'rtf',
        
        # Image files
        b'\xFF\xD8\xFF': 'jpeg',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'png',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'BM': 'bmp',  # BMP files
        b'II*\x00': 'tiff',  # TIFF little endian
        b'MM\x00*': 'tiff',  # TIFF big endian
        b'RIFF': 'webp',  # WebP (need to check for WEBP in header)
        
        # Archive files
        b'\x1F\x8B': 'gzip',
        b'\x42\x5A\x68': 'bzip2',
        b'\x37\x7A\xBC\xAF\x27\x1C': '7zip',
    }
    
    # File extensions mapping
    EXTENSION_MAPPING = {
        '.txt': 'txt',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.mdown': 'markdown',
        '.mkd': 'markdown',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.xlsx': 'xlsx',
        '.xls': 'xls',
        '.pptx': 'pptx',
        '.ppt': 'ppt',
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'xhtml',
        '.xml': 'xml',
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.rtf': 'rtf',
        '.odt': 'odt',
        '.ods': 'ods',
        '.odp': 'odp',
        '.epub': 'epub',
        '.mobi': 'mobi',
        '.log': 'log',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'cfg',
        '.conf': 'conf',
        # Image formats
        '.png': 'png',
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.gif': 'gif',
        '.bmp': 'bmp',
        '.tiff': 'tiff',
        '.tif': 'tiff',
        '.webp': 'webp',
    }
    
    # Supported file types for processing
    SUPPORTED_TYPES = {
        'txt', 'markdown', 'pdf', 'docx', 'doc', 'html', 'htm', 
        'json', 'jsonl', 'csv', 'tsv', 'xml', 'rtf', 'epub',
        # Image formats (for OCR)
        'png', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'
    }
    
    def __init__(self):
        """Initialize the file type detector."""
        # Initialize mimetypes for additional detection
        mimetypes.init()
    
    def detect_file_type(self, file_path: str) -> Tuple[str, float]:
        """
        Detect the file type of a given file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Tuple of (file_type, confidence_score)
            confidence_score is between 0.0 and 1.0
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file can't be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {file_path}")
        
        # Get file extension
        file_extension = Path(file_path).suffix.lower()
        
        # Try signature detection first (highest confidence)
        signature_type = self._detect_by_signature(file_path)
        if signature_type:
            return self._resolve_signature_type(signature_type, file_extension), 0.95
        
        # Try extension detection
        extension_type = self._detect_by_extension(file_extension)
        if extension_type:
            return extension_type, 0.8
        
        # Try MIME type detection
        mime_type = self._detect_by_mime_type(file_path)
        if mime_type:
            return mime_type, 0.6
        
        # Try content analysis for text files
        content_type = self._detect_by_content(file_path)
        if content_type:
            return content_type, 0.4
        
        # Default to unknown
        return 'unknown', 0.0
    
    def is_supported(self, file_type: str) -> bool:
        """
        Check if a file type is supported for processing.
        
        Args:
            file_type: The file type to check
            
        Returns:
            True if the file type is supported, False otherwise
        """
        return file_type.lower() in self.SUPPORTED_TYPES
    
    def get_supported_types(self) -> List[str]:
        """
        Get a list of all supported file types.
        
        Returns:
            List of supported file type strings
        """
        return sorted(list(self.SUPPORTED_TYPES))
    
    def _detect_by_signature(self, file_path: str) -> Optional[str]:
        """Detect file type by reading file signature (magic numbers)."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 32 bytes for signature detection
                header = f.read(32)
                
                # Check against known signatures
                for signature, file_type in self.FILE_SIGNATURES.items():
                    if header.startswith(signature):
                        return file_type
                
                # Special case: check for text-based signatures in larger chunk
                if len(header) < 32:
                    # File is very small, read more if possible
                    f.seek(0)
                    header = f.read(512)
                
                # Check for HTML/XML patterns in larger chunk
                header_lower = header.lower()
                if b'<!doctype html' in header_lower or b'<html' in header_lower:
                    return 'html'
                if b'<?xml' in header_lower:
                    return 'xml'
                if header_lower.startswith(b'{"') or header_lower.startswith(b'[{'):
                    return 'json'
                    
        except (IOError, OSError):
            pass
        
        return None
    
    def _detect_by_extension(self, file_extension: str) -> Optional[str]:
        """Detect file type by file extension."""
        return self.EXTENSION_MAPPING.get(file_extension.lower())
    
    def _detect_by_mime_type(self, file_path: str) -> Optional[str]:
        """Detect file type using MIME type detection."""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                # Map common MIME types to our file types
                mime_mapping = {
                    'text/plain': 'txt',
                    'text/markdown': 'markdown',
                    'text/html': 'html',
                    'text/xml': 'xml',
                    'application/xml': 'xml',
                    'application/json': 'json',
                    'text/csv': 'csv',
                    'application/pdf': 'pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'application/msword': 'doc',
                    'application/rtf': 'rtf',
                    'text/rtf': 'rtf',
                    'application/epub+zip': 'epub',
                }
                return mime_mapping.get(mime_type)
        except Exception:
            pass
        
        return None
    
    def _detect_by_content(self, file_path: str) -> Optional[str]:
        """Detect file type by analyzing file content."""
        try:
            # Try to read as text and analyze content patterns
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
                
                # Check if content is mostly binary (non-printable characters)
                if content:
                    printable_ratio = sum(1 for c in content if c.isprintable() or c.isspace()) / len(content)
                    if printable_ratio < 0.7:  # Less than 70% printable characters
                        return None  # Likely binary file
                
                # Check for common patterns
                content_lower = content.lower().strip()
                
                # JSON patterns
                if (content_lower.startswith('{') and '}' in content_lower) or \
                   (content_lower.startswith('[') and ']' in content_lower):
                    return 'json'
                
                # CSV patterns (look for comma-separated values)
                lines = content.split('\n')[:5]  # Check first 5 lines
                if len(lines) >= 2:
                    comma_count = sum(line.count(',') for line in lines)
                    if comma_count >= len(lines):  # At least one comma per line on average
                        return 'csv'
                
                # HTML patterns
                if any(tag in content_lower for tag in ['<html', '<head', '<body', '<div', '<p>']):
                    return 'html'
                
                # XML patterns
                if content_lower.startswith('<?xml') or '<' in content_lower and '>' in content_lower:
                    return 'xml'
                
                # Markdown patterns
                if any(pattern in content for pattern in ['# ', '## ', '### ', '* ', '- ', '1. ']):
                    return 'markdown'
                
                # Default to plain text if it's readable text
                if content and (content.isprintable() or all(ord(c) < 128 for c in content)):
                    return 'txt'
                    
        except (UnicodeDecodeError, IOError, OSError):
            pass
        
        return None
    
    def _resolve_signature_type(self, signature_type: str, file_extension: str) -> str:
        """Resolve ambiguous signature types using file extension."""
        if signature_type == 'office_zip':
            # ZIP-based Office files and other ZIP formats - use extension to determine specific type
            ext_mapping = {
                '.docx': 'docx',
                '.xlsx': 'xlsx', 
                '.pptx': 'pptx',
                '.odt': 'odt',
                '.ods': 'ods',
                '.odp': 'odp',
                '.epub': 'epub',
            }
            return ext_mapping.get(file_extension, 'zip')
        
        elif signature_type == 'office_legacy':
            # Legacy Office files
            ext_mapping = {
                '.doc': 'doc',
                '.xls': 'xls',
                '.ppt': 'ppt',
            }
            return ext_mapping.get(file_extension, 'ole2')
        

        
        return signature_type
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            stat = os.stat(file_path)
            file_type, confidence = self.detect_file_type(file_path)
            
            return {
                'path': file_path,
                'name': os.path.basename(file_path),
                'extension': Path(file_path).suffix.lower(),
                'size_bytes': stat.st_size,
                'detected_type': file_type,
                'confidence': confidence,
                'is_supported': self.is_supported(file_type),
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime,
            }
        except (OSError, IOError) as e:
            return {
                'path': file_path,
                'error': str(e),
                'detected_type': 'error',
                'confidence': 0.0,
                'is_supported': False,
            }