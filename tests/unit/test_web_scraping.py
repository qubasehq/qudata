"""
Unit tests for web scraping functionality.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.qudata.ingest.scraper import WebScraper, RateLimiter, CacheManager
from src.qudata.models import ExtractedContent, FileMetadata, ProcessingError


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.requests_per_minute == 60
        assert limiter.min_interval == 1.0
    
    def test_rate_limiter_wait(self):
        """Test rate limiter wait functionality."""
        limiter = RateLimiter(requests_per_minute=120)  # 0.5 second interval
        
        start_time = time.time()
        limiter.wait_if_needed()
        first_wait = time.time() - start_time
        
        # First call should not wait
        assert first_wait < 0.1
        
        # Second call should wait
        start_time = time.time()
        limiter.wait_if_needed()
        second_wait = time.time() - start_time
        
        assert second_wait >= 0.4  # Should wait approximately 0.5 seconds


class TestCacheManager:
    """Test caching functionality."""
    
    def test_cache_manager_initialization(self, tmp_path):
        """Test cache manager initialization."""
        cache_dir = tmp_path / "test_cache"
        cache = CacheManager(str(cache_dir), max_age_hours=1)
        
        assert cache.cache_dir == cache_dir
        assert cache.max_age_seconds == 3600
        assert cache_dir.exists()
    
    def test_cache_set_and_get(self, tmp_path):
        """Test cache set and get operations."""
        cache_dir = tmp_path / "test_cache"
        cache = CacheManager(str(cache_dir), max_age_hours=24)
        
        test_url = "https://example.com/test"
        test_data = {
            "content": "Test content",
            "metadata": {"title": "Test Page"},
            "timestamp": time.time()
        }
        
        # Set cache
        cache.set(test_url, test_data)
        
        # Get cache
        cached_data = cache.get(test_url)
        assert cached_data is not None
        assert cached_data["content"] == "Test content"
        assert cached_data["metadata"]["title"] == "Test Page"
    
    def test_cache_expiration(self, tmp_path):
        """Test cache expiration."""
        cache_dir = tmp_path / "test_cache"
        cache = CacheManager(str(cache_dir), max_age_hours=0)  # Immediate expiration
        
        test_url = "https://example.com/test"
        test_data = {"content": "Test content"}
        
        # Set cache
        cache.set(test_url, test_data)
        
        # Wait a bit to ensure expiration
        time.sleep(0.1)
        
        # Should return None due to expiration
        cached_data = cache.get(test_url)
        assert cached_data is None
    
    def test_cache_nonexistent_url(self, tmp_path):
        """Test cache get for non-existent URL."""
        cache_dir = tmp_path / "test_cache"
        cache = CacheManager(str(cache_dir))
        
        cached_data = cache.get("https://nonexistent.com")
        assert cached_data is None


class TestWebScraper:
    """Test web scraping functionality."""
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = Mock()
        response.status_code = 200
        response.headers = {
            'content-type': 'text/html; charset=utf-8',
            'last-modified': 'Wed, 21 Oct 2015 07:28:00 GMT'
        }
        response.content = b"""
        <html>
        <head>
            <title>Test Page</title>
            <meta property="og:title" content="Test Page OG">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a test paragraph with <a href="https://example.com">a link</a>.</p>
            <h2>Subheading</h2>
            <ul>
                <li>List item 1</li>
                <li>List item 2</li>
            </ul>
            <table>
                <thead>
                    <tr><th>Header 1</th><th>Header 2</th></tr>
                </thead>
                <tbody>
                    <tr><td>Cell 1</td><td>Cell 2</td></tr>
                </tbody>
            </table>
            <img src="test.jpg" alt="Test image" width="100" height="200">
            <script>console.log('test');</script>
            <style>.test { color: red; }</style>
        </body>
        </html>
        """
        response.url = "https://example.com/test"
        response.encoding = 'utf-8'
        response.history = []
        response.raise_for_status = Mock()
        return response
    
    @pytest.fixture
    def scraper_config(self, tmp_path):
        """Create scraper configuration."""
        return {
            'requests_per_minute': 120,
            'cache_enabled': True,
            'cache_dir': str(tmp_path / "cache"),
            'timeout': 10,
            'max_retries': 2,
            'respect_robots_txt': False,  # Disable for testing
            'use_readability': False  # Disable for consistent testing
        }
    
    def test_scraper_initialization(self, scraper_config):
        """Test web scraper initialization."""
        scraper = WebScraper(scraper_config)
        
        assert scraper.timeout == 10
        assert scraper.max_retries == 2
        assert scraper.respect_robots_txt is False
        assert scraper.use_readability is False
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_url_success(self, mock_session_class, scraper_config, mock_response):
        """Test successful URL scraping."""
        # Setup mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        scraper = WebScraper(scraper_config)
        
        # Mock the session to use our mock
        scraper.session = mock_session
        
        result = scraper.scrape_url("https://example.com/test")
        
        assert isinstance(result, ExtractedContent)
        assert "Main Heading" in result.content
        assert "test paragraph" in result.content
        assert result.metadata.file_path == "https://example.com/test"
        assert result.metadata.file_type == "html"
        assert result.metadata.size_bytes > 0
        assert len(result.tables) == 1
        assert len(result.images) == 1
        assert result.structure.paragraphs > 0
        assert result.structure.links > 0
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_url_with_extraction_rules(self, mock_session_class, scraper_config, mock_response):
        """Test URL scraping with custom extraction rules."""
        # Setup mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        extraction_rules = {
            'remove_selectors': ['script', 'style'],
            'keep_selectors': ['h1', 'p'],
            'remove_by_attributes': {
                'class': ['ads', 'sidebar']
            }
        }
        
        result = scraper.scrape_url("https://example.com/test", extraction_rules)
        
        assert isinstance(result, ExtractedContent)
        assert "Main Heading" in result.content
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_url_invalid_url(self, mock_session_class, scraper_config):
        """Test scraping with invalid URL."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            scraper.scrape_url("invalid-url")
        
        assert exc_info.value.error_type == "InvalidURL"
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_url_http_error(self, mock_session_class, scraper_config):
        """Test scraping with HTTP error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock HTTP error - the error should be raised by session.get, not raise_for_status
        from requests.exceptions import HTTPError
        mock_response = Mock()
        mock_response.status_code = 404
        
        # Create HTTPError with response attribute
        http_error = HTTPError()
        http_error.response = mock_response
        
        # Make session.get raise the HTTPError directly
        mock_session.get.side_effect = http_error
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            scraper.scrape_url("https://example.com/notfound")
        
        assert exc_info.value.error_type == "HTTPError"
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_url_timeout(self, mock_session_class, scraper_config):
        """Test scraping with timeout error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock timeout error
        from requests.exceptions import Timeout
        mock_session.get.side_effect = Timeout()
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            scraper.scrape_url("https://example.com/timeout")
        
        assert exc_info.value.error_type == "RequestTimeout"
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_urls_multiple(self, mock_session_class, scraper_config, mock_response):
        """Test scraping multiple URLs."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]
        
        results = scraper.scrape_urls(urls)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExtractedContent)
            assert "Main Heading" in result.content
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_scrape_with_caching(self, mock_session_class, scraper_config, mock_response, tmp_path):
        """Test scraping with caching enabled."""
        scraper_config['cache_dir'] = str(tmp_path / "cache")
        
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        scraper = WebScraper(scraper_config)
        scraper.session = mock_session
        
        url = "https://example.com/cached"
        
        # First request should hit the network
        result1 = scraper.scrape_url(url)
        assert mock_session.get.call_count == 1
        
        # Second request should use cache
        result2 = scraper.scrape_url(url)
        assert mock_session.get.call_count == 1  # No additional network call
        
        # Results should be the same
        assert result1.content == result2.content
        assert result1.metadata.file_path == result2.metadata.file_path
    
    def test_robots_txt_checking(self, scraper_config):
        """Test robots.txt checking functionality."""
        scraper_config['respect_robots_txt'] = True
        scraper = WebScraper(scraper_config)
        
        # Mock robots.txt parser
        with patch('src.qudata.ingest.scraper.RobotFileParser') as mock_rp_class:
            mock_rp = Mock()
            mock_rp_class.return_value = mock_rp
            mock_rp.can_fetch.return_value = False
            
            scraper.robots_cache['https://example.com'] = mock_rp
            
            # Should return False when robots.txt blocks
            can_fetch = scraper._can_fetch("https://example.com/blocked")
            assert can_fetch is False
    
    def test_content_extraction_with_readability(self, scraper_config):
        """Test content extraction with readability."""
        scraper_config['use_readability'] = True
        
        # Test that readability configuration is set correctly
        scraper = WebScraper(scraper_config)
        
        # Since readability is not installed, use_readability should be False
        # But the config should still be set to True
        assert scraper.config.get('use_readability') is True


class TestContentExtraction:
    """Test content extraction functionality."""
    
    def test_title_extraction(self):
        """Test title extraction from HTML."""
        from src.qudata.ingest.scraper import WebScraper
        from bs4 import BeautifulSoup
        
        scraper = WebScraper()
        
        # Test title tag
        html = "<html><head><title>Page Title</title></head></html>"
        soup = BeautifulSoup(html, 'html.parser')
        title = scraper._extract_title(soup)
        assert title == "Page Title"
        
        # Test h1 fallback
        html = "<html><body><h1>Main Heading</h1></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        title = scraper._extract_title(soup)
        assert title == "Main Heading"
        
        # Test og:title fallback
        html = '<html><head><meta property="og:title" content="OG Title"></head></html>'
        soup = BeautifulSoup(html, 'html.parser')
        title = scraper._extract_title(soup)
        assert title == "OG Title"
    
    def test_table_extraction(self):
        """Test table extraction from HTML."""
        from src.qudata.ingest.scraper import WebScraper
        from bs4 import BeautifulSoup
        
        scraper = WebScraper()
        
        html = """
        <table>
            <thead>
                <tr><th>Name</th><th>Age</th></tr>
            </thead>
            <tbody>
                <tr><td>John</td><td>30</td></tr>
                <tr><td>Jane</td><td>25</td></tr>
            </tbody>
            <caption>User Data</caption>
        </table>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        tables = scraper._extract_tables(soup)
        
        assert len(tables) == 1
        table = tables[0]
        assert table.headers == ["Name", "Age"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["John", "30"]
        assert table.rows[1] == ["Jane", "25"]
        assert table.caption == "User Data"
    
    def test_image_extraction(self):
        """Test image extraction from HTML."""
        from src.qudata.ingest.scraper import WebScraper
        from bs4 import BeautifulSoup
        
        scraper = WebScraper()
        
        html = """
        <img src="image1.jpg" alt="First image" title="Image 1" width="100" height="200">
        <img src="https://example.com/image2.png" alt="Second image">
        <img src="relative/image3.gif">
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        images = scraper._extract_images(soup, "https://example.com/page")
        
        assert len(images) == 3
        
        # First image (should be resolved to absolute URL)
        assert images[0].path == "https://example.com/image1.jpg"
        assert images[0].alt_text == "First image"
        assert images[0].caption == "Image 1"
        assert images[0].width == 100
        assert images[0].height == 200
        
        # Second image (absolute URL)
        assert images[1].path == "https://example.com/image2.png"
        assert images[1].alt_text == "Second image"
        
        # Third image (relative URL resolved)
        assert images[2].path == "https://example.com/relative/image3.gif"
    
    def test_html_cleaning(self):
        """Test HTML cleaning functionality."""
        from src.qudata.ingest.scraper import WebScraper
        from bs4 import BeautifulSoup
        
        scraper = WebScraper()
        
        html = """
        <div>
            <script>alert('test');</script>
            <style>.test { color: red; }</style>
            <!-- This is a comment -->
            <nav>Navigation</nav>
            <aside class="sidebar">Sidebar</aside>
            <div class="ads">Advertisement</div>
            <p>Main content</p>
        </div>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        cleaned_soup = scraper._clean_html(soup)
        
        # Check that unwanted elements are removed
        assert not cleaned_soup.find('script')
        assert not cleaned_soup.find('style')
        assert not cleaned_soup.find('nav')
        assert not cleaned_soup.find('aside')
        assert not cleaned_soup.find(class_='ads')
        
        # Check that main content is preserved
        assert cleaned_soup.find('p')
        assert "Main content" in cleaned_soup.get_text()


if __name__ == "__main__":
    pytest.main([__file__])