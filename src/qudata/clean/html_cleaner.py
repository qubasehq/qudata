"""
HTML cleaning and emoji removal module.

This module provides comprehensive HTML cleaning capabilities including:
- HTML tag removal with content preservation
- Emoji and special character removal
- HTML entity decoding
- Whitespace normalization
"""

import re
import html
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from bs4 import BeautifulSoup
import unicodedata


@dataclass
class HTMLCleaningResult:
    """Result of HTML cleaning operation."""
    cleaned_text: str
    removed_tags: List[str]
    removed_emojis: List[str]
    removed_entities: List[str]
    original_length: int
    cleaned_length: int
    
    @property
    def reduction_ratio(self) -> float:
        """Calculate the reduction ratio after cleaning."""
        if self.original_length == 0:
            return 0.0
        return (self.original_length - self.cleaned_length) / self.original_length


class HTMLCleaner:
    """
    Comprehensive HTML cleaner for removing tags, emojis, and special characters.
    
    This class provides methods to clean HTML content while preserving meaningful text
    and removing unwanted elements like tags, emojis, and special characters.
    """
    
    def __init__(self, 
                 preserve_links: bool = False,
                 preserve_formatting: bool = False,
                 remove_emojis: bool = True,
                 remove_special_chars: bool = True,
                 custom_emoji_patterns: Optional[List[str]] = None):
        """
        Initialize HTMLCleaner with configuration options.
        
        Args:
            preserve_links: Whether to preserve link URLs
            preserve_formatting: Whether to preserve basic formatting (bold, italic)
            remove_emojis: Whether to remove emoji characters
            remove_special_chars: Whether to remove special characters
            custom_emoji_patterns: Additional regex patterns for emoji removal
        """
        self.preserve_links = preserve_links
        self.preserve_formatting = preserve_formatting
        self.remove_emojis = remove_emojis
        self.remove_special_chars = remove_special_chars
        
        # Build emoji pattern
        self._emoji_pattern = self._build_emoji_pattern(custom_emoji_patterns)
        
        # Tags to preserve if formatting is enabled
        self.formatting_tags = {'b', 'strong', 'i', 'em', 'u', 'mark'} if preserve_formatting else set()
        
        # Special characters pattern
        self._special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]')
    
    def _build_emoji_pattern(self, custom_patterns: Optional[List[str]] = None) -> re.Pattern:
        """Build comprehensive emoji regex pattern."""
        # Unicode emoji ranges
        emoji_ranges = [
            r'\U0001F600-\U0001F64F',  # emoticons
            r'\U0001F300-\U0001F5FF',  # symbols & pictographs
            r'\U0001F680-\U0001F6FF',  # transport & map symbols
            r'\U0001F1E0-\U0001F1FF',  # flags (iOS)
            r'\U00002702-\U000027B0',  # dingbats
            r'\U000024C2-\U0001F251',  # enclosed characters
            r'\U0001F900-\U0001F9FF',  # supplemental symbols
            r'\U0001FA70-\U0001FAFF',  # symbols and pictographs extended-A
        ]
        
        # Common emoji patterns
        patterns = [
            f'[{"".join(emoji_ranges)}]',
            r'[\u2600-\u26FF]',  # miscellaneous symbols
            r'[\u2700-\u27BF]',  # dingbats
        ]
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns.extend(custom_patterns)
        
        combined_pattern = '|'.join(patterns)
        return re.compile(combined_pattern, re.UNICODE)
    
    def clean_html(self, html_content: str) -> HTMLCleaningResult:
        """
        Clean HTML content by removing tags, emojis, and special characters.
        
        Args:
            html_content: Raw HTML content to clean
            
        Returns:
            HTMLCleaningResult with cleaned text and metadata
        """
        original_length = len(html_content)
        removed_tags = []
        removed_emojis = []
        removed_entities = []
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract and remove script and style tags
        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            removed_tags.append(tag.name)
            tag.decompose()
        
        # Handle links if preservation is enabled
        if self.preserve_links:
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().strip()
                link_url = link['href']
                if link_text and link_url:
                    link.replace_with(f"{link_text} ({link_url})")
                else:
                    removed_tags.append('a')
                    link.decompose()
        
        # Handle formatting tags
        if not self.preserve_formatting:
            for tag in soup.find_all():
                if tag.name not in ['script', 'style', 'meta', 'link', 'head']:
                    removed_tags.append(tag.name)
        
        # Get text content
        text = soup.get_text()
        
        # Decode HTML entities
        original_entities = re.findall(r'&[a-zA-Z0-9#]+;', text)
        removed_entities.extend(original_entities)
        text = html.unescape(text)
        
        # Remove emojis if enabled
        if self.remove_emojis:
            emoji_matches = self._emoji_pattern.findall(text)
            removed_emojis.extend(emoji_matches)
            text = self._emoji_pattern.sub('', text)
        
        # Remove special characters if enabled
        if self.remove_special_chars:
            text = self._special_chars_pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove empty lines and excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return HTMLCleaningResult(
            cleaned_text=text,
            removed_tags=list(set(removed_tags)),
            removed_emojis=removed_emojis,
            removed_entities=removed_entities,
            original_length=original_length,
            cleaned_length=len(text)
        )
    
    def clean_text_emojis(self, text: str) -> HTMLCleaningResult:
        """
        Clean only emojis and special characters from plain text.
        
        Args:
            text: Plain text content to clean
            
        Returns:
            HTMLCleaningResult with cleaned text and metadata
        """
        original_length = len(text)
        removed_emojis = []
        
        # Remove emojis if enabled
        if self.remove_emojis:
            emoji_matches = self._emoji_pattern.findall(text)
            removed_emojis.extend(emoji_matches)
            text = self._emoji_pattern.sub('', text)
        
        # Remove special characters if enabled
        if self.remove_special_chars:
            text = self._special_chars_pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return HTMLCleaningResult(
            cleaned_text=text,
            removed_tags=[],
            removed_emojis=removed_emojis,
            removed_entities=[],
            original_length=original_length,
            cleaned_length=len(text)
        )
    
    def batch_clean(self, html_contents: List[str]) -> List[HTMLCleaningResult]:
        """
        Clean multiple HTML contents in batch.
        
        Args:
            html_contents: List of HTML content strings to clean
            
        Returns:
            List of HTMLCleaningResult objects
        """
        return [self.clean_html(content) for content in html_contents]
    
    def get_cleaning_stats(self, results: List[HTMLCleaningResult]) -> Dict[str, any]:
        """
        Get statistics from multiple cleaning results.
        
        Args:
            results: List of HTMLCleaningResult objects
            
        Returns:
            Dictionary with cleaning statistics
        """
        if not results:
            return {}
        
        total_original = sum(r.original_length for r in results)
        total_cleaned = sum(r.cleaned_length for r in results)
        total_tags = sum(len(r.removed_tags) for r in results)
        total_emojis = sum(len(r.removed_emojis) for r in results)
        total_entities = sum(len(r.removed_entities) for r in results)
        
        return {
            'total_documents': len(results),
            'total_original_length': total_original,
            'total_cleaned_length': total_cleaned,
            'average_reduction_ratio': (total_original - total_cleaned) / total_original if total_original > 0 else 0,
            'total_tags_removed': total_tags,
            'total_emojis_removed': total_emojis,
            'total_entities_removed': total_entities,
            'average_tags_per_document': total_tags / len(results),
            'average_emojis_per_document': total_emojis / len(results),
        }


def clean_html_content(html_content: str, 
                      preserve_links: bool = False,
                      preserve_formatting: bool = False,
                      remove_emojis: bool = True) -> str:
    """
    Convenience function to clean HTML content with default settings.
    
    Args:
        html_content: HTML content to clean
        preserve_links: Whether to preserve link URLs
        preserve_formatting: Whether to preserve basic formatting
        remove_emojis: Whether to remove emojis
        
    Returns:
        Cleaned text content
    """
    cleaner = HTMLCleaner(
        preserve_links=preserve_links,
        preserve_formatting=preserve_formatting,
        remove_emojis=remove_emojis
    )
    result = cleaner.clean_html(html_content)
    return result.cleaned_text