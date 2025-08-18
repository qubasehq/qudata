"""
Data ingestion module for LLMDataForge.

This module provides comprehensive data ingestion capabilities including:
- File format detection and extraction
- Web scraping with rate limiting and caching
- API integration for REST and GraphQL endpoints
- Database connectivity
- Streaming data processing
"""

from .detector import FileTypeDetector
from .files import PlainTextExtractor
from .pdf import PDFExtractor
from .document import DocumentExtractor
from .web import WebExtractor
from .structured import StructuredExtractor
from .scraper import WebScraper, RateLimiter, CacheManager
from .api import APIClient, AuthConfig, ContentExtractor
from .ocr import OCRProcessor, ImagePreprocessor, ScannedPDFHandler, OCRExtractor, OCRResult
from .stream import (
    StreamProcessor, BaseStreamProcessor, RSSFeedReader, LogParser, KafkaConnector,
    StreamConfig, StreamItem
)

__all__ = [
    # File detection and extraction
    'FileTypeDetector',
    'PlainTextExtractor', 
    'PDFExtractor',
    'DocumentExtractor',
    'WebExtractor',
    'StructuredExtractor',
    
    # Web scraping
    'WebScraper',
    'RateLimiter',
    'CacheManager',
    
    # API integration
    'APIClient',
    'AuthConfig', 
    'ContentExtractor',
    
    # OCR processing
    'OCRProcessor',
    'ImagePreprocessor',
    'ScannedPDFHandler',
    'OCRExtractor',
    'OCRResult',
    
    # Streaming data processing
    'StreamProcessor',
    'BaseStreamProcessor',
    'RSSFeedReader',
    'LogParser', 
    'KafkaConnector',
    'StreamConfig',
    'StreamItem',
]