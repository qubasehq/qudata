"""
Metadata extraction for documents.

This module provides functionality to extract metadata such as author, date,
source, and document type from various document formats and content.
"""

import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..models import Document, DocumentMetadata, ProcessingError, ErrorSeverity


@dataclass
class ExtractedMetadata:
    """Result of metadata extraction."""
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    source_url: Optional[str] = None
    document_type: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0
    extraction_method: str = "unknown"


class MetadataExtractor:
    """Extract metadata from document content and file properties."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata extractor.
        
        Args:
            config: Configuration for metadata extraction
        """
        self.config = config or {}
        self._author_patterns = self._compile_author_patterns()
        self._date_patterns = self._compile_date_patterns()
        self._url_patterns = self._compile_url_patterns()
        self._title_patterns = self._compile_title_patterns()
    
    def _compile_author_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for author extraction."""
        patterns = [
            # Common author patterns
            r'(?:^|\n)\s*(?:author|by|written by|created by):\s*([^\n\r]+?)(?:\n|\r|$)',
            r'(?:^|\n)\s*(?:author|by|written by|created by)\s+([^\n\r]+?)(?:\n|\r|$)',
            r'(?:^|\n)\s*@author\s+([^\n\r]+?)(?:\n|\r|$)',
            r'\\author\{([^}]+)\}',  # LaTeX
            r'<meta\s+name=["\']author["\']\s+content=["\']([^"\']+)["\']',  # HTML meta
            r'<author>([^<]+)</author>',  # XML
            r'(?:^|\n)\s*Author:\s*([^\n\r]+?)(?:\n|\r|$)',
            r'(?:^|\n)\s*By:\s*([^\n\r]+?)(?:\n|\r|$)',
        ]
        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
    
    def _compile_date_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for date extraction."""
        patterns = [
            # ISO date formats
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2})',
            # Common date formats
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\d{4}/\d{1,2}/\d{1,2})',
            # Date with context
            r'(?:date|created|published|modified):\s*(\d{4}-\d{2}-\d{2})',
            r'(?:date|created|published|modified)\s+(\d{4}-\d{2}-\d{2})',
            r'<meta\s+name=["\'](?:date|created|published)["\']\s+content=["\']([^"\']+)["\']',
            # Month day, year format
            r'([A-Za-z]+\s+\d{1,2},\s+\d{4})',
            r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        ]
        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
    
    def _compile_url_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for URL extraction."""
        patterns = [
            r'(?:source|url|link):\s*(https?://[^\s\n\r]+?)(?:\s|\n|\r|$)',
            r'(?:source|url|link)\s+(https?://[^\s\n\r]+?)(?:\s|\n|\r|$)',
            r'<meta\s+name=["\']url["\']\s+content=["\']([^"\']+)["\']',
            r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']',
            r'(https?://[^\s\n\r]+?)(?:\s|\n|\r|$)',  # Any HTTP URL
        ]
        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
    
    def _compile_title_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for title extraction."""
        patterns = [
            # More specific patterns first - require word boundaries
            r'(?:^|\n)\s*(?:title|subject):\s*([^\n\r]+?)(?:\n|\r|$)',
            r'<title>([^<]+)</title>',
            r'<h1[^>]*>([^<]+)</h1>',
            r'<meta\s+name=["\']title["\']\s+content=["\']([^"\']+)["\']',
            r'\\title\{([^}]+)\}',  # LaTeX
            r'^#\s+(.+?)$',  # Markdown H1
            r'^(.+?)\n=+$',  # Markdown H1 underline style
        ]
        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
    
    def extract_metadata(self, document: Document) -> ExtractedMetadata:
        """
        Extract metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            ExtractedMetadata with extracted information
        """
        metadata = ExtractedMetadata()
        
        try:
            # Extract from file system metadata
            file_metadata = self._extract_file_metadata(document.source_path)
            if file_metadata:
                metadata.creation_date = file_metadata.get('creation_date')
                metadata.modification_date = file_metadata.get('modification_date')
                metadata.document_type = file_metadata.get('document_type')
            
            # Extract from content
            content_metadata = self._extract_content_metadata(document.content)
            if content_metadata:
                # Prefer content metadata over file metadata for some fields
                if content_metadata.author:
                    metadata.author = content_metadata.author
                if content_metadata.creation_date:
                    metadata.creation_date = content_metadata.creation_date
                if content_metadata.source_url:
                    metadata.source_url = content_metadata.source_url
                if content_metadata.title:
                    metadata.title = content_metadata.title
                
                # Update confidence based on content extraction success
                metadata.confidence = content_metadata.confidence
                metadata.extraction_method = content_metadata.extraction_method
            
            # Extract from existing document metadata if available
            if document.metadata:
                if document.metadata.author and not metadata.author:
                    metadata.author = document.metadata.author
                if document.metadata.creation_date and not metadata.creation_date:
                    metadata.creation_date = document.metadata.creation_date
                if document.metadata.source_url and not metadata.source_url:
                    metadata.source_url = document.metadata.source_url
                if document.metadata.language and not metadata.language:
                    metadata.language = document.metadata.language
        
        except Exception as e:
            raise ProcessingError(
                stage="metadata_extraction",
                error_type="ExtractionError",
                message=f"Failed to extract metadata: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                document_id=document.id
            )
        
        return metadata
    
    def _extract_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from file system properties.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata or None if extraction fails
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                'creation_date': datetime.fromtimestamp(stat.st_ctime),
                'modification_date': datetime.fromtimestamp(stat.st_mtime),
                'document_type': path_obj.suffix.lower().lstrip('.') or 'unknown'
            }
        
        except Exception:
            return None
    
    def _extract_content_metadata(self, content: str) -> ExtractedMetadata:
        """
        Extract metadata from document content using pattern matching.
        
        Args:
            content: Document content to analyze
            
        Returns:
            ExtractedMetadata with extracted information
        """
        metadata = ExtractedMetadata()
        extraction_count = 0
        
        # Extract author
        author = self._extract_author(content)
        if author:
            metadata.author = author
            extraction_count += 1
        
        # Extract dates
        date = self._extract_date(content)
        if date:
            metadata.creation_date = date
            extraction_count += 1
        
        # Extract URLs
        url = self._extract_url(content)
        if url:
            metadata.source_url = url
            extraction_count += 1
        
        # Extract title
        title = self._extract_title(content)
        if title:
            metadata.title = title
            extraction_count += 1
        
        # Calculate confidence based on successful extractions
        total_possible = 4  # author, date, url, title
        metadata.confidence = extraction_count / total_possible
        metadata.extraction_method = "content_patterns"
        
        return metadata
    
    def _extract_author(self, content: str) -> Optional[str]:
        """Extract author from content using regex patterns."""
        for pattern in self._author_patterns:
            match = pattern.search(content)
            if match:
                author = match.group(1).strip()
                # Clean up common artifacts
                author = re.sub(r'[<>"\']', '', author)
                author = author.strip()
                # Check if it looks like a reasonable author name
                if (len(author) > 2 and len(author) < 100 and 
                    not author.lower() in ('information', 'content', 'text') and
                    not author.lower().endswith(('information', 'content', 'text'))):
                    return author
        return None
    
    def _extract_date(self, content: str) -> Optional[datetime]:
        """Extract date from content using regex patterns."""
        for pattern in self._date_patterns:
            match = pattern.search(content)
            if match:
                date_str = match.group(1).strip()
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date
        return None
    
    def _extract_url(self, content: str) -> Optional[str]:
        """Extract URL from content using regex patterns."""
        for pattern in self._url_patterns:
            match = pattern.search(content)
            if match:
                url = match.group(1).strip()
                # Basic URL validation
                if url.startswith(('http://', 'https://')) and len(url) > 10:
                    return url
        return None
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from content using regex patterns."""
        for i, pattern in enumerate(self._title_patterns):
            match = pattern.search(content)
            if match:
                title = match.group(1).strip()
                # Clean up title
                title = re.sub(r'[<>"\']', '', title)
                title = title.strip()
                # Check if it looks like a reasonable title
                if (len(title) > 3 and len(title) < 200 and 
                    not title.lower() in ('information', 'content', 'text') and
                    not title.lower().endswith(('information', 'content', 'text'))):
                    return title
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string into a datetime object.
        
        Args:
            date_str: String representation of a date
            
        Returns:
            Parsed datetime object or None if parsing fails
        """
        # Common date formats to try
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%Y/%m/%d',
            '%B %d, %Y',
            '%d %B %Y',
            '%b %d, %Y',
            '%d %b %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def update_document_metadata(self, document: Document, 
                               extracted: ExtractedMetadata) -> Document:
        """
        Update document metadata with extracted information.
        
        Args:
            document: Document to update
            extracted: Extracted metadata to apply
            
        Returns:
            Updated document
        """
        if not document.metadata:
            document.metadata = DocumentMetadata(
                file_type=extracted.document_type or "unknown",
                size_bytes=len(document.content.encode('utf-8')),
                language=extracted.language or "unknown"
            )
        
        # Update metadata fields if extracted values are available
        if extracted.author:
            document.metadata.author = extracted.author
        
        if extracted.creation_date:
            document.metadata.creation_date = extracted.creation_date
        
        if extracted.modification_date:
            document.metadata.modification_date = extracted.modification_date
        
        if extracted.source_url:
            document.metadata.source_url = extracted.source_url
        
        if extracted.document_type:
            document.metadata.file_type = extracted.document_type
        
        return document
    
    def get_extraction_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about metadata extraction across multiple documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            'total_documents': len(documents),
            'author_extracted': 0,
            'date_extracted': 0,
            'url_extracted': 0,
            'title_extracted': 0,
            'avg_confidence': 0.0,
            'extraction_methods': {}
        }
        
        total_confidence = 0.0
        
        for doc in documents:
            extracted = self.extract_metadata(doc)
            
            if extracted.author:
                stats['author_extracted'] += 1
            if extracted.creation_date:
                stats['date_extracted'] += 1
            if extracted.source_url:
                stats['url_extracted'] += 1
            if extracted.title:
                stats['title_extracted'] += 1
            
            total_confidence += extracted.confidence
            
            method = extracted.extraction_method
            stats['extraction_methods'][method] = stats['extraction_methods'].get(method, 0) + 1
        
        if len(documents) > 0:
            stats['avg_confidence'] = total_confidence / len(documents)
        
        return stats