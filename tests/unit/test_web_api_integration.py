"""
Integration tests for web scraping and API functionality.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.qudata.ingest.scraper import WebScraper
from src.qudata.ingest.api import APIClient, AuthConfig, ContentExtractor
from src.qudata.models import ExtractedContent, ProcessingError


class TestWebScrapingIntegration:
    """Integration tests for web scraping functionality."""
    
    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample News Article</title>
            <meta property="og:title" content="Breaking News: AI Advances">
            <meta name="author" content="Jane Reporter">
        </head>
        <body>
            <header>
                <nav>Navigation menu</nav>
            </header>
            
            <main>
                <article>
                    <h1>AI Technology Breakthrough</h1>
                    <p class="byline">By Jane Reporter | Published: 2023-01-15</p>
                    
                    <p>Scientists have announced a major breakthrough in artificial intelligence 
                    technology that could revolutionize how we interact with computers.</p>
                    
                    <h2>Key Findings</h2>
                    <ul>
                        <li>Improved natural language processing</li>
                        <li>Better reasoning capabilities</li>
                        <li>Enhanced multimodal understanding</li>
                    </ul>
                    
                    <blockquote>
                        "This represents a significant step forward in AI development," 
                        said Dr. Smith, lead researcher.
                    </blockquote>
                    
                    <h2>Technical Details</h2>
                    <p>The new system uses advanced neural networks to process information 
                    more efficiently than previous models.</p>
                    
                    <table>
                        <caption>Performance Comparison</caption>
                        <thead>
                            <tr><th>Model</th><th>Accuracy</th><th>Speed</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Previous</td><td>85%</td><td>100ms</td></tr>
                            <tr><td>New</td><td>95%</td><td>50ms</td></tr>
                        </tbody>
                    </table>
                    
                    <p>For more information, visit our 
                    <a href="https://example.com/research">research page</a>.</p>
                    
                    <img src="ai-diagram.png" alt="AI Architecture Diagram" 
                         width="600" height="400" title="System Architecture">
                </article>
            </main>
            
            <aside class="sidebar">
                <div class="ads">Advertisement content</div>
                <div class="social">Social media links</div>
            </aside>
            
            <footer>
                <p>Copyright 2023 News Site</p>
            </footer>
            
            <script>
                // Analytics code
                console.log('Page loaded');
            </script>
        </body>
        </html>
        """
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_complete_web_scraping_workflow(self, mock_session_class, sample_html_content, tmp_path):
        """Test complete web scraping workflow with all features."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'content-type': 'text/html; charset=utf-8',
            'last-modified': 'Mon, 15 Jan 2023 10:00:00 GMT'
        }
        mock_response.content = sample_html_content.encode('utf-8')
        mock_response.url = "https://news.example.com/ai-breakthrough"
        mock_response.encoding = 'utf-8'
        mock_response.history = []
        mock_response.raise_for_status = Mock()
        
        # Setup mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        # Configure scraper
        config = {
            'requests_per_minute': 120,
            'cache_enabled': True,
            'cache_dir': str(tmp_path / "cache"),
            'timeout': 30,
            'max_retries': 3,
            'respect_robots_txt': False,
            'use_readability': False,  # Disable for consistent testing
            'extract_tables': True,
            'extract_images': True,
            'preserve_links': True
        }
        
        scraper = WebScraper(config)
        scraper.session = mock_session
        
        # Test scraping
        result = scraper.scrape_url("https://news.example.com/ai-breakthrough")
        
        # Verify extracted content
        assert isinstance(result, ExtractedContent)
        
        # Check content extraction
        assert "AI Technology Breakthrough" in result.content
        assert "Scientists have announced" in result.content
        assert "Key Findings" in result.content
        assert "Technical Details" in result.content
        
        # Check metadata
        assert result.metadata.file_path == "https://news.example.com/ai-breakthrough"
        assert result.metadata.file_type == "html"
        assert result.metadata.size_bytes > 0
        
        # Check structure analysis
        assert result.structure is not None
        assert len(result.structure.headings) >= 2  # h1 and h2 elements
        assert result.structure.paragraphs > 0
        assert result.structure.lists > 0  # ul element
        assert result.structure.links > 0  # a element
        
        # Check table extraction
        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.caption == "Performance Comparison"
        assert "Model" in table.headers
        assert "Accuracy" in table.headers
        assert len(table.rows) == 2
        
        # Check image extraction
        assert len(result.images) == 1
        image = result.images[0]
        assert image.path == "https://news.example.com/ai-diagram.png"  # Should be resolved to absolute URL
        assert image.alt_text == "AI Architecture Diagram"
        assert image.width == 600
        assert image.height == 400
        
        # Verify noise removal (ads, navigation, etc. should be removed)
        assert "Advertisement content" not in result.content
        assert "Navigation menu" not in result.content
        assert "Social media links" not in result.content
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_web_scraping_with_extraction_rules(self, mock_session_class, sample_html_content):
        """Test web scraping with custom extraction rules."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.content = sample_html_content.encode('utf-8')
        mock_response.url = "https://news.example.com/test"
        mock_response.encoding = 'utf-8'
        mock_response.history = []
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        
        scraper = WebScraper({'respect_robots_txt': False})
        scraper.session = mock_session
        
        # Define extraction rules
        extraction_rules = {
            'keep_selectors': ['article'],  # Only keep article content
            'remove_selectors': ['.byline', 'blockquote'],  # Remove byline and quotes
            'remove_by_attributes': {
                'class': ['ads', 'social']
            }
        }
        
        result = scraper.scrape_url("https://news.example.com/test", extraction_rules)
        
        # Should contain main article content
        assert "AI Technology Breakthrough" in result.content
        assert "Scientists have announced" in result.content
        
        # Should not contain removed elements
        assert "Jane Reporter" not in result.content  # byline removed
        assert "Dr. Smith" not in result.content  # blockquote removed
    
    @patch('src.qudata.ingest.scraper.requests.Session')
    def test_web_scraping_error_handling(self, mock_session_class):
        """Test web scraping error handling."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        scraper = WebScraper({'respect_robots_txt': False})
        scraper.session = mock_session
        
        # Test HTTP 404 error
        from requests.exceptions import HTTPError
        mock_response = Mock()
        mock_response.status_code = 404
        
        # Create HTTPError with response attribute
        http_error = HTTPError()
        http_error.response = mock_response
        
        # Make session.get raise the HTTPError directly
        mock_session.get.side_effect = http_error
        
        with pytest.raises(ProcessingError) as exc_info:
            scraper.scrape_url("https://example.com/notfound")
        
        assert exc_info.value.error_type == "HTTPError"
        
        # Test connection error
        from requests.exceptions import ConnectionError
        mock_session.get.side_effect = ConnectionError()
        
        with pytest.raises(ProcessingError) as exc_info:
            scraper.scrape_url("https://unreachable.com")
        
        assert exc_info.value.error_type == "ConnectionError"


class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    @pytest.fixture
    def sample_api_responses(self):
        """Sample API responses for testing."""
        return {
            'articles_list': {
                'data': [
                    {
                        'id': 1,
                        'title': 'First Article',
                        'content': 'Content of the first article about technology trends.',
                        'author': 'Tech Writer',
                        'created_at': '2023-01-01T00:00:00Z',
                        'tags': ['technology', 'trends']
                    },
                    {
                        'id': 2,
                        'title': 'Second Article', 
                        'content': 'Content of the second article about AI developments.',
                        'author': 'AI Researcher',
                        'created_at': '2023-01-02T00:00:00Z',
                        'tags': ['ai', 'research']
                    }
                ],
                'pagination': {
                    'page': 1,
                    'per_page': 2,
                    'total': 5,
                    'has_more': True
                }
            },
            'single_article': {
                'id': 1,
                'title': 'Detailed Article',
                'content': 'This is a comprehensive article about machine learning algorithms and their applications in modern software development.',
                'author': {
                    'name': 'Dr. Jane Smith',
                    'username': 'jsmith',
                    'bio': 'AI researcher and professor'
                },
                'created_at': '2023-01-15T10:30:00Z',
                'updated_at': '2023-01-16T14:20:00Z',
                'metadata': {
                    'word_count': 1500,
                    'reading_time': 6,
                    'category': 'technology'
                },
                'sections': [
                    {
                        'title': 'Introduction',
                        'content': 'Machine learning has revolutionized software development.'
                    },
                    {
                        'title': 'Applications',
                        'content': 'Various applications include recommendation systems and natural language processing.'
                    }
                ]
            },
            'graphql_response': {
                'data': {
                    'posts': [
                        {
                            'id': '1',
                            'title': 'GraphQL Article',
                            'body': 'This article explains GraphQL concepts and best practices.',
                            'author': {
                                'name': 'GraphQL Expert',
                                'email': 'expert@example.com'
                            },
                            'publishedAt': '2023-02-01T00:00:00Z'
                        }
                    ]
                }
            }
        }
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_rest_api_integration(self, mock_session_class, sample_api_responses):
        """Test REST API integration workflow."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Configure API client
        config = {
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'requests_per_minute': 60,
            'auth': {
                'auth_type': 'bearer',
                'token': 'test-api-token'
            },
            'headers': {
                'Accept': 'application/json',
                'User-Agent': 'LLMDataForge/1.0'
            }
        }
        
        client = APIClient(config)
        client.session = mock_session
        
        # Test GET request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_responses['single_article']
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        
        result = client.get('/articles/1')
        
        # Verify request was made correctly
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'GET'
        assert call_args[1]['url'] == 'https://api.example.com/articles/1'
        
        # Verify response
        assert result['title'] == 'Detailed Article'
        assert result['author']['name'] == 'Dr. Jane Smith'
        
        # Test POST request
        mock_session.request.reset_mock()
        post_data = {
            'title': 'New Article',
            'content': 'New article content',
            'author_id': 1
        }
        
        result = client.post('/articles', json_data=post_data)
        
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['json'] == post_data
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_graphql_api_integration(self, mock_session_class, sample_api_responses):
        """Test GraphQL API integration."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        client = APIClient({'base_url': 'https://api.example.com'})
        client.session = mock_session
        
        # Mock GraphQL response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_responses['graphql_response']
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        
        # Test GraphQL query
        query = """
        query GetPosts($limit: Int!) {
            posts(limit: $limit) {
                id
                title
                body
                author {
                    name
                    email
                }
                publishedAt
            }
        }
        """
        variables = {'limit': 10}
        
        result = client.graphql_query(query, variables)
        
        # Verify request
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['url'] == 'https://api.example.com/graphql'
        
        sent_data = call_args[1]['json']
        assert sent_data['query'] == query
        assert sent_data['variables'] == variables
        
        # Verify response
        assert 'posts' in result
        assert len(result['posts']) == 1
        assert result['posts'][0]['title'] == 'GraphQL Article'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_paginated_data_fetching(self, mock_session_class, sample_api_responses):
        """Test fetching paginated data from API."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        client = APIClient({'base_url': 'https://api.example.com'})
        client.session = mock_session
        
        # Mock paginated responses
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            'data': sample_api_responses['articles_list']['data'],
            'has_more': True
        }
        page1_response.raise_for_status = Mock()
        
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            'data': [
                {
                    'id': 3,
                    'title': 'Third Article',
                    'content': 'Content of the third article.',
                    'author': 'Another Writer',
                    'created_at': '2023-01-03T00:00:00Z'
                }
            ],
            'has_more': False
        }
        page2_response.raise_for_status = Mock()
        
        mock_session.request.side_effect = [page1_response, page2_response]
        
        # Fetch all pages
        all_items = client.fetch_paginated_data('/articles', params={'limit': 2})
        
        # Verify results
        assert len(all_items) == 3
        assert all_items[0]['title'] == 'First Article'
        assert all_items[1]['title'] == 'Second Article'
        assert all_items[2]['title'] == 'Third Article'
        
        # Verify pagination requests
        assert mock_session.request.call_count == 2
    
    def test_content_extraction_from_api_response(self, sample_api_responses):
        """Test extracting content from API responses."""
        extractor = ContentExtractor()
        
        # Test simple article extraction
        article_data = sample_api_responses['single_article']
        result = extractor.extract_from_api_response(
            article_data,
            source_url='https://api.example.com/articles/1'
        )
        
        assert isinstance(result, ExtractedContent)
        assert result.content == article_data['content']
        assert result.metadata.file_path == 'https://api.example.com/articles/1'
        assert result.metadata.file_type == 'api_response'
        assert result.metadata.size_bytes > 0
        
        # Verify structure extraction
        assert result.structure is not None
        assert result.structure.paragraphs > 0
        assert result.structure.lists > 0  # sections array
    
    def test_content_extraction_with_custom_rules(self, sample_api_responses):
        """Test content extraction with custom field mapping."""
        config = {
            'extraction_rules': {
                'content_fields': ['body', 'text', 'description'],
                'title_fields': ['headline', 'subject', 'name'],
                'author_fields': ['creator', 'writer', 'user'],
                'date_fields': ['publishedAt', 'timestamp', 'date']
            }
        }
        
        extractor = ContentExtractor(config)
        
        # Test GraphQL response extraction
        graphql_data = sample_api_responses['graphql_response']['data']['posts'][0]
        result = extractor.extract_from_api_response(graphql_data)
        
        # Should use 'body' field for content
        assert result.content == 'This article explains GraphQL concepts and best practices.'
        assert result.metadata.file_type == 'api_response'
        assert result.metadata.size_bytes > 0
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_api_authentication_methods(self, mock_session_class):
        """Test different API authentication methods."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Test Bearer token auth
        config = {
            'base_url': 'https://api.example.com',
            'auth': {
                'auth_type': 'bearer',
                'token': 'bearer-token-123'
            }
        }
        
        client = APIClient(config)
        assert client.session.headers['Authorization'] == 'Bearer bearer-token-123'
        
        # Test API key auth
        config = {
            'base_url': 'https://api.example.com',
            'auth': {
                'auth_type': 'api_key',
                'api_key': 'api-key-456',
                'header_name': 'X-API-Key'
            }
        }
        
        client = APIClient(config)
        assert client.session.headers['X-API-Key'] == 'api-key-456'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_api_error_handling_and_retries(self, mock_session_class):
        """Test API error handling and retry logic."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        client = APIClient({
            'base_url': 'https://api.example.com',
            'max_retries': 2
        })
        client.session = mock_session
        
        # Test server error with successful retry
        from requests.exceptions import HTTPError
        
        error_response = Mock()
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = HTTPError(response=error_response)
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'success': True}
        success_response.raise_for_status = Mock()
        
        mock_session.request.side_effect = [error_response, success_response]
        
        # Should succeed after retry
        result = client.get('/test')
        assert result['success'] is True
        assert mock_session.request.call_count == 2
        
        # Test client error (should not retry)
        mock_session.request.reset_mock()
        
        # Create HTTPError for 404 response
        client_error_response = Mock()
        client_error_response.status_code = 404
        client_error_response.text = 'Not Found'
        
        http_error = HTTPError()
        http_error.response = client_error_response
        
        # Make session.request raise the HTTPError directly
        mock_session.request.side_effect = http_error
        
        with pytest.raises(ProcessingError) as exc_info:
            client.get('/notfound')
        
        assert exc_info.value.error_type == "HTTPClientError"
        assert mock_session.request.call_count == 1  # No retry for 4xx errors


if __name__ == "__main__":
    pytest.main([__file__])