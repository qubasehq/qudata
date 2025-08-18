"""
Web scraping module for LLMDataForge.

This module provides comprehensive web scraping capabilities using BeautifulSoup4 and requests,
with rate limiting, caching, and configurable extraction rules.
"""

import time
import hashlib
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from pathlib import Path
import logging

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from readability import Document as ReadabilityDocument
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

from ..models import (
    ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, TableData, ImageData
)


class RateLimiter:
    """Rate limiter for web requests."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class CacheManager:
    """Simple file-based cache for web requests."""
    
    def __init__(self, cache_dir: str = ".cache/web", max_age_hours: int = 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cached content in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content for URL.
        
        Args:
            url: URL to get cached content for
            
        Returns:
            Cached content dict or None if not found/expired
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is expired
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > self.max_age_seconds:
                cache_path.unlink()  # Remove expired cache
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None
    
    def set(self, url: str, content: Dict[str, Any]) -> None:
        """
        Cache content for URL.
        
        Args:
            url: URL to cache content for
            content: Content to cache
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logging.warning(f"Failed to cache content for {url}: {e}")


class WebScraper:
    """
    Comprehensive web scraper with rate limiting, caching, and configurable extraction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize web scraper.
        
        Args:
            config: Configuration dictionary with options like:
                - requests_per_minute: Rate limit (default: 60)
                - cache_enabled: Enable caching (default: True)
                - cache_dir: Cache directory (default: .cache/web)
                - cache_max_age_hours: Cache expiration (default: 24)
                - user_agent: User agent string
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Maximum retry attempts (default: 3)
                - respect_robots_txt: Check robots.txt (default: True)
                - follow_redirects: Follow redirects (default: True)
                - max_redirects: Maximum redirects (default: 10)
                - use_readability: Use readability extraction (default: True)
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for web scraping. "
                "Install it with: pip install requests"
            )
        
        if not HAS_BS4:
            raise ImportError(
                "beautifulsoup4 is required for web scraping. "
                "Install it with: pip install beautifulsoup4"
            )
        
        self.config = config or {}
        
        # Rate limiting
        requests_per_minute = self.config.get('requests_per_minute', 60)
        self.rate_limiter = RateLimiter(requests_per_minute)
        
        # Caching
        self.cache_enabled = self.config.get('cache_enabled', True)
        if self.cache_enabled:
            cache_dir = self.config.get('cache_dir', '.cache/web')
            cache_max_age = self.config.get('cache_max_age_hours', 24)
            self.cache = CacheManager(cache_dir, cache_max_age)
        
        # Request configuration
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.respect_robots_txt = self.config.get('respect_robots_txt', True)
        self.follow_redirects = self.config.get('follow_redirects', True)
        self.max_redirects = self.config.get('max_redirects', 10)
        self.use_readability = self.config.get('use_readability', True) and HAS_READABILITY
        
        # Setup session with retry strategy
        self.session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        user_agent = self.config.get('user_agent', 
            'Mozilla/5.0 (compatible; LLMDataForge/1.0; +https://github.com/qubasehq/qudata)')
        self.session.headers.update({'User-Agent': user_agent})
        
        # Robots.txt cache
        self.robots_cache = {}
    
    def scrape_url(self, url: str, extraction_rules: Dict[str, Any] = None) -> ExtractedContent:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            extraction_rules: Optional extraction configuration
            
        Returns:
            ExtractedContent object with scraped content
            
        Raises:
            ProcessingError: If scraping fails
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ProcessingError(
                    stage="scraping",
                    error_type="InvalidURL",
                    message=f"Invalid URL: {url}",
                    severity=ErrorSeverity.HIGH
                )
            
            # Check robots.txt if enabled
            if self.respect_robots_txt and not self._can_fetch(url):
                raise ProcessingError(
                    stage="scraping",
                    error_type="RobotsTxtBlocked",
                    message=f"URL blocked by robots.txt: {url}",
                    severity=ErrorSeverity.MEDIUM
                )
            
            # Check cache first
            if self.cache_enabled:
                cached_content = self.cache.get(url)
                if cached_content:
                    return self._create_extracted_content_from_cache(cached_content, url)
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Make request
            response = self._make_request(url)
            
            # Extract content
            extracted_content = self._extract_content_from_response(response, extraction_rules)
            
            # Cache the result
            if self.cache_enabled:
                # Convert metadata to dict with datetime serialization
                metadata_dict = extracted_content.metadata.to_dict()
                
                cache_data = {
                    'content': extracted_content.content,
                    'metadata': metadata_dict,
                    'structure': extracted_content.structure.__dict__ if extracted_content.structure else None,
                    'tables': [table.__dict__ for table in extracted_content.tables],
                    'images': [image.__dict__ for image in extracted_content.images],
                    'url': url,
                    'timestamp': time.time()
                }
                self.cache.set(url, cache_data)
            
            return extracted_content
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                stage="scraping",
                error_type="ScrapingError",
                message=f"Failed to scrape URL {url}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def scrape_urls(self, urls: List[str], extraction_rules: Dict[str, Any] = None) -> List[ExtractedContent]:
        """
        Scrape content from multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            extraction_rules: Optional extraction configuration
            
        Returns:
            List of ExtractedContent objects
        """
        results = []
        
        for url in urls:
            try:
                content = self.scrape_url(url, extraction_rules)
                results.append(content)
            except ProcessingError as e:
                logging.warning(f"Failed to scrape {url}: {e.message}")
                continue
        
        return results
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if base_url not in self.robots_cache:
                robots_url = urljoin(base_url, '/robots.txt')
                rp = RobotFileParser()
                rp.set_url(robots_url)
                
                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except Exception:
                    # If robots.txt can't be read, assume we can fetch
                    self.robots_cache[base_url] = None
            
            robots_parser = self.robots_cache[base_url]
            if robots_parser is None:
                return True
            
            user_agent = self.session.headers.get('User-Agent', '*')
            return robots_parser.can_fetch(user_agent, url)
            
        except Exception:
            # If anything goes wrong, assume we can fetch
            return True
    
    def _make_request(self, url: str) -> requests.Response:
        """
        Make HTTP request to URL.
        
        Args:
            url: URL to request
            
        Returns:
            Response object
            
        Raises:
            ProcessingError: If request fails
        """
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects
            )
            
            # Check for too many redirects
            if len(response.history) > self.max_redirects:
                raise ProcessingError(
                    stage="scraping",
                    error_type="TooManyRedirects",
                    message=f"Too many redirects for URL: {url}",
                    severity=ErrorSeverity.MEDIUM
                )
            
            # Check response status
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.Timeout:
            raise ProcessingError(
                stage="scraping",
                error_type="RequestTimeout",
                message=f"Request timeout for URL: {url}",
                severity=ErrorSeverity.MEDIUM
            )
        except requests.exceptions.ConnectionError:
            raise ProcessingError(
                stage="scraping",
                error_type="ConnectionError",
                message=f"Connection error for URL: {url}",
                severity=ErrorSeverity.MEDIUM
            )
        except requests.exceptions.HTTPError as e:
            raise ProcessingError(
                stage="scraping",
                error_type="HTTPError",
                message=f"HTTP error {e.response.status_code} for URL: {url}",
                severity=ErrorSeverity.MEDIUM
            )
        except Exception as e:
            raise ProcessingError(
                stage="scraping",
                error_type="RequestError",
                message=f"Request failed for URL {url}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def _extract_content_from_response(self, response: requests.Response, 
                                     extraction_rules: Dict[str, Any] = None) -> ExtractedContent:
        """
        Extract content from HTTP response.
        
        Args:
            response: HTTP response object
            extraction_rules: Optional extraction configuration
            
        Returns:
            ExtractedContent object
        """
        # Get content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'html' not in content_type:
            raise ProcessingError(
                stage="extraction",
                error_type="UnsupportedContentType",
                message=f"Unsupported content type: {content_type}",
                severity=ErrorSeverity.MEDIUM
            )
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Use readability if enabled and available
        if self.use_readability:
            try:
                doc = ReadabilityDocument(response.content)
                main_content = doc.summary()
                title = doc.title()
                
                # Parse the cleaned content
                content_soup = BeautifulSoup(main_content, 'html.parser')
            except Exception as e:
                logging.warning(f"Readability extraction failed, falling back to BeautifulSoup: {e}")
                content_soup = soup
                title = self._extract_title(soup)
        else:
            content_soup = soup
            title = self._extract_title(soup)
        
        # Apply extraction rules if provided
        if extraction_rules:
            content_soup = self._apply_extraction_rules(content_soup, extraction_rules)
        
        # Clean the content
        cleaned_soup = self._clean_html(content_soup)
        
        # Extract content and structure
        content, structure = self._extract_content_and_structure(cleaned_soup, title)
        
        # Extract tables
        tables = self._extract_tables(cleaned_soup)
        
        # Extract images
        images = self._extract_images(soup, response.url)
        
        # Create metadata
        metadata = FileMetadata(
            file_path=response.url,
            file_type='html',
            size_bytes=len(response.content)
        )
        
        # Create extracted content
        extracted = ExtractedContent(content, metadata)
        extracted.structure = structure
        extracted.tables = tables
        extracted.images = images
        
        return extracted
    
    def _apply_extraction_rules(self, soup: BeautifulSoup, rules: Dict[str, Any]) -> BeautifulSoup:
        """
        Apply custom extraction rules to soup.
        
        Args:
            soup: BeautifulSoup object
            rules: Extraction rules configuration
            
        Returns:
            Modified BeautifulSoup object
        """
        # Remove elements by selector
        remove_selectors = rules.get('remove_selectors', [])
        for selector in remove_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Keep only elements matching selector
        keep_selectors = rules.get('keep_selectors', [])
        if keep_selectors:
            kept_elements = []
            for selector in keep_selectors:
                kept_elements.extend(soup.select(selector))
            
            if kept_elements:
                # Create new soup with only kept elements
                new_soup = BeautifulSoup('<div></div>', 'html.parser')
                container = new_soup.div
                for element in kept_elements:
                    container.append(element.extract())
                soup = new_soup
        
        # Remove elements by attribute
        remove_attrs = rules.get('remove_by_attributes', {})
        for attr_name, attr_values in remove_attrs.items():
            if isinstance(attr_values, str):
                attr_values = [attr_values]
            
            for attr_value in attr_values:
                for element in soup.find_all(attrs={attr_name: attr_value}):
                    element.decompose()
        
        return soup
    
    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean HTML by removing unwanted elements."""
        # Remove script and style tags
        for script in soup.find_all(['script', 'style']):
            script.decompose()
        
        # Remove comments
        from bs4 import Comment
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
        
        return soup
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML document."""
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
        """Extract main content and analyze document structure."""
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
        """Extract tables from HTML."""
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
                    if first_row and first_row.find('th'):
                        headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
                
                # Extract data rows
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
                logging.warning(f"Error extracting table {table_idx}: {e}")
                continue
        
        return tables
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[ImageData]:
        """Extract image metadata from HTML."""
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
                    width = int(width) if width else None
                except (ValueError, TypeError):
                    width = None
                
                try:
                    height = int(height) if height else None
                except (ValueError, TypeError):
                    height = None
                
                # Resolve relative URLs
                if src and not src.startswith(('http://', 'https://', 'data:')):
                    src = urljoin(base_url, src)
                
                image_data = ImageData(
                    path=src or f"image_{img_idx}",
                    caption=title or alt or f"Image {img_idx}",
                    alt_text=alt,
                    width=width,
                    height=height
                )
                images.append(image_data)
                
            except Exception as e:
                logging.warning(f"Error extracting image {img_idx}: {e}")
                continue
        
        return images
    
    def _create_extracted_content_from_cache(self, cache_data: Dict[str, Any], url: str) -> ExtractedContent:
        """Create ExtractedContent object from cached data."""
        # Reconstruct metadata
        metadata_dict = cache_data['metadata']
        metadata = FileMetadata(
            file_path=metadata_dict.get('file_path', url),
            file_type=metadata_dict.get('file_type', 'html'),
            size_bytes=metadata_dict.get('size_bytes', 0)
        )
        
        # Create extracted content
        extracted = ExtractedContent(cache_data['content'], metadata)
        
        # Reconstruct structure if available
        if cache_data.get('structure'):
            structure_dict = cache_data['structure']
            extracted.structure = DocumentStructure(
                headings=structure_dict.get('headings', []),
                paragraphs=structure_dict.get('paragraphs', 0),
                tables=structure_dict.get('tables', 0),
                images=structure_dict.get('images', 0),
                code_blocks=structure_dict.get('code_blocks', 0),
                lists=structure_dict.get('lists', 0),
                links=structure_dict.get('links', 0)
            )
        
        # Reconstruct tables
        extracted.tables = []
        for table_dict in cache_data.get('tables', []):
            table = TableData(
                headers=table_dict.get('headers', []),
                rows=table_dict.get('rows', []),
                caption=table_dict.get('caption', '')
            )
            extracted.tables.append(table)
        
        # Reconstruct images
        extracted.images = []
        for image_dict in cache_data.get('images', []):
            image = ImageData(
                path=image_dict.get('path', ''),
                caption=image_dict.get('caption', ''),
                alt_text=image_dict.get('alt_text', ''),
                width=image_dict.get('width'),
                height=image_dict.get('height')
            )
            extracted.images.append(image)
        
        return extracted