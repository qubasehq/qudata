"""
Unit tests for API client functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.qudata.ingest.api import APIClient, AuthConfig, ContentExtractor
from src.qudata.models import ExtractedContent, FileMetadata, ProcessingError


class TestAuthConfig:
    """Test authentication configuration."""
    
    def test_basic_auth_config(self):
        """Test basic authentication configuration."""
        auth = AuthConfig('basic', username='user', password='pass')
        assert auth.auth_type == 'basic'
        assert auth.params['username'] == 'user'
        assert auth.params['password'] == 'pass'
    
    def test_bearer_auth_config(self):
        """Test bearer token authentication configuration."""
        auth = AuthConfig('bearer', token='abc123')
        assert auth.auth_type == 'bearer'
        assert auth.params['token'] == 'abc123'
    
    def test_api_key_auth_config(self):
        """Test API key authentication configuration."""
        auth = AuthConfig('api_key', api_key='key123', header_name='X-Custom-Key')
        assert auth.auth_type == 'api_key'
        assert auth.params['api_key'] == 'key123'
        assert auth.params['header_name'] == 'X-Custom-Key'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_apply_basic_auth_to_session(self, mock_session_class):
        """Test applying basic auth to session."""
        mock_session = Mock()
        mock_session.headers = {}
        
        auth = AuthConfig('basic', username='user', password='pass')
        auth.apply_to_session(mock_session)
        
        # Check that auth was set (we can't easily test the exact auth object)
        assert mock_session.auth is not None
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_apply_bearer_auth_to_session(self, mock_session_class):
        """Test applying bearer auth to session."""
        mock_session = Mock()
        mock_session.headers = {}
        
        auth = AuthConfig('bearer', token='abc123')
        auth.apply_to_session(mock_session)
        
        assert mock_session.headers['Authorization'] == 'Bearer abc123'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_apply_api_key_auth_to_session(self, mock_session_class):
        """Test applying API key auth to session."""
        mock_session = Mock()
        mock_session.headers = {}
        
        auth = AuthConfig('api_key', api_key='key123', header_name='X-API-Key')
        auth.apply_to_session(mock_session)
        
        assert mock_session.headers['X-API-Key'] == 'key123'


class TestAPIClient:
    """Test API client functionality."""
    
    @pytest.fixture
    def client_config(self):
        """Create API client configuration."""
        return {
            'base_url': 'https://api.example.com',
            'timeout': 10,
            'max_retries': 2,
            'requests_per_minute': 120,
            'headers': {'User-Agent': 'TestClient/1.0'},
            'verify_ssl': True
        }
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            'id': 1,
            'title': 'Test Article',
            'content': 'This is test content',
            'author': 'Test Author',
            'created_at': '2023-01-01T00:00:00Z'
        }
        response.raise_for_status = Mock()
        return response
    
    def test_client_initialization(self, client_config):
        """Test API client initialization."""
        client = APIClient(client_config)
        
        assert client.base_url == 'https://api.example.com'
        assert client.timeout == 10
        assert client.max_retries == 2
        assert client.verify_ssl is True
    
    def test_client_initialization_with_auth(self):
        """Test API client initialization with authentication."""
        config = {
            'base_url': 'https://api.example.com',
            'auth': {
                'auth_type': 'bearer',
                'token': 'test-token'
            }
        }
        
        client = APIClient(config)
        assert client.session.headers.get('Authorization') == 'Bearer test-token'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_get_request(self, mock_session_class, client_config, mock_response):
        """Test GET request."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        
        client = APIClient(client_config)
        client.session = mock_session
        
        result = client.get('/articles/1', params={'include': 'author'})
        
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'GET'
        assert call_args[1]['url'] == 'https://api.example.com/articles/1'
        assert call_args[1]['params'] == {'include': 'author'}
        
        assert result['title'] == 'Test Article'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_post_request(self, mock_session_class, client_config, mock_response):
        """Test POST request."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        
        client = APIClient(client_config)
        client.session = mock_session
        
        post_data = {'title': 'New Article', 'content': 'New content'}
        result = client.post('/articles', json_data=post_data)
        
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['json'] == post_data
        
        assert result['title'] == 'Test Article'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_graphql_query(self, mock_session_class, client_config):
        """Test GraphQL query."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock GraphQL response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'user': {
                    'id': '1',
                    'name': 'John Doe',
                    'email': 'john@example.com'
                }
            }
        }
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        
        client = APIClient(client_config)
        client.session = mock_session
        
        query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
            }
        }
        """
        variables = {'id': '1'}
        
        result = client.graphql_query(query, variables)
        
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['url'] == 'https://api.example.com/graphql'
        
        # Check that query and variables were sent correctly
        sent_data = call_args[1]['json']
        assert sent_data['query'] == query
        assert sent_data['variables'] == variables
        
        assert result['user']['name'] == 'John Doe'
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_graphql_query_with_errors(self, mock_session_class, client_config):
        """Test GraphQL query with errors."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock GraphQL error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'errors': [
                {'message': 'User not found'},
                {'message': 'Invalid permissions'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        
        client = APIClient(client_config)
        client.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            client.graphql_query("query { user { id } }")
        
        assert exc_info.value.error_type == "GraphQLError"
        assert "User not found" in exc_info.value.message
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_fetch_paginated_data(self, mock_session_class, client_config):
        """Test fetching paginated data."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock paginated responses
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            'data': [
                {'id': 1, 'title': 'Article 1'},
                {'id': 2, 'title': 'Article 2'}
            ],
            'has_more': True
        }
        page1_response.raise_for_status = Mock()
        
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            'data': [
                {'id': 3, 'title': 'Article 3'}
            ],
            'has_more': False
        }
        page2_response.raise_for_status = Mock()
        
        mock_session.request.side_effect = [page1_response, page2_response]
        
        client = APIClient(client_config)
        client.session = mock_session
        
        result = client.fetch_paginated_data('/articles', params={'limit': 2})
        
        assert len(result) == 3
        assert result[0]['title'] == 'Article 1'
        assert result[1]['title'] == 'Article 2'
        assert result[2]['title'] == 'Article 3'
        
        # Check that two requests were made
        assert mock_session.request.call_count == 2
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_http_error_handling(self, mock_session_class, client_config):
        """Test HTTP error handling."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock HTTP error
        from requests.exceptions import HTTPError
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = 'Not Found'
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_session.request.return_value = mock_response
        
        client = APIClient(client_config)
        client.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            client.get('/nonexistent')
        
        assert exc_info.value.error_type == "HTTPClientError"
        assert "404" in exc_info.value.message
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_timeout_error_handling(self, mock_session_class, client_config):
        """Test timeout error handling."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock timeout error
        from requests.exceptions import Timeout
        mock_session.request.side_effect = Timeout()
        
        client = APIClient(client_config)
        client.session = mock_session
        
        with pytest.raises(ProcessingError) as exc_info:
            client.get('/timeout')
        
        assert exc_info.value.error_type == "RequestTimeout"
    
    @patch('src.qudata.ingest.api.requests.Session')
    def test_retry_logic(self, mock_session_class, client_config):
        """Test retry logic for server errors."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.headers = {}
        
        # Mock server error followed by success
        from requests.exceptions import HTTPError
        
        error_response = Mock()
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = HTTPError(response=error_response)
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'success': True}
        success_response.raise_for_status = Mock()
        
        mock_session.request.side_effect = [error_response, success_response]
        
        client = APIClient(client_config)
        client.session = mock_session
        
        # Should succeed after retry
        result = client.get('/retry-test')
        assert result['success'] is True
        assert mock_session.request.call_count == 2
    
    def test_build_url(self, client_config):
        """Test URL building."""
        client = APIClient(client_config)
        
        # Test relative URL
        url = client._build_url('/articles/1')
        assert url == 'https://api.example.com/articles/1'
        
        # Test absolute URL
        url = client._build_url('https://other-api.com/data')
        assert url == 'https://other-api.com/data'
        
        # Test URL without leading slash
        url = client._build_url('articles/1')
        assert url == 'https://api.example.com/articles/1'


class TestContentExtractor:
    """Test content extraction from API responses."""
    
    def test_extract_from_simple_response(self):
        """Test extracting content from simple API response."""
        extractor = ContentExtractor()
        
        response_data = {
            'id': 1,
            'title': 'Test Article',
            'content': 'This is the main content of the article.',
            'author': 'John Doe',
            'created_at': '2023-01-01T00:00:00Z',
            'tags': ['tech', 'api']
        }
        
        result = extractor.extract_from_api_response(
            response_data, 
            source_url='https://api.example.com/articles/1'
        )
        
        assert isinstance(result, ExtractedContent)
        assert result.content == 'This is the main content of the article.'
        assert result.metadata.file_path == 'https://api.example.com/articles/1'
        assert result.metadata.file_type == 'api_response'
        assert result.metadata.size_bytes > 0
    
    def test_extract_with_custom_rules(self):
        """Test extracting content with custom extraction rules."""
        config = {
            'extraction_rules': {
                'content_fields': ['body', 'text'],
                'title_fields': ['name', 'headline'],
                'author_fields': ['creator', 'user'],
                'date_fields': ['published_at', 'timestamp']
            }
        }
        
        extractor = ContentExtractor(config)
        
        response_data = {
            'id': 1,
            'name': 'Custom Title',
            'body': 'Custom content field',
            'creator': 'Jane Smith',
            'published_at': '2023-02-01T00:00:00Z'
        }
        
        result = extractor.extract_from_api_response(response_data)
        
        assert result.content == 'Custom content field'
        assert result.metadata.file_type == 'api_response'
        assert result.metadata.size_bytes > 0
    
    def test_extract_from_nested_response(self):
        """Test extracting content from nested API response."""
        extractor = ContentExtractor()
        
        response_data = {
            'data': {
                'article': {
                    'title': 'Nested Article',
                    'content': 'Nested content',
                    'metadata': {
                        'author': {
                            'name': 'Author Name',
                            'username': 'author123'
                        }
                    }
                }
            }
        }
        
        result = extractor.extract_from_api_response(response_data)
        
        # Should extract content recursively
        assert 'Nested Article' in result.content or 'Nested content' in result.content
        assert isinstance(result, ExtractedContent)
    
    def test_extract_from_list_response(self):
        """Test extracting content from list API response."""
        extractor = ContentExtractor()
        
        response_data = [
            {
                'title': 'Article 1',
                'content': 'Content 1'
            },
            {
                'title': 'Article 2', 
                'content': 'Content 2'
            }
        ]
        
        result = extractor.extract_from_api_response(response_data)
        
        # Should handle list responses
        assert isinstance(result, ExtractedContent)
        assert result.content is not None
    
    def test_extract_fallback_to_json(self):
        """Test fallback to JSON when no content fields found."""
        extractor = ContentExtractor()
        
        response_data = {
            'id': 1,
            'status': 'active',
            'metadata': {
                'version': '1.0',
                'type': 'document'
            }
        }
        
        result = extractor.extract_from_api_response(response_data)
        
        # Should fallback to JSON representation
        assert isinstance(result, ExtractedContent)
        assert '"id": 1' in result.content
        assert '"status": "active"' in result.content
    
    def test_structure_extraction(self):
        """Test document structure extraction from API response."""
        extractor = ContentExtractor()
        
        response_data = {
            'title': 'Test Document',
            'content': 'This is a paragraph.\n\nThis is another paragraph with http://example.com link.',
            'sections': [
                {'name': 'Section 1', 'content': 'Section content'},
                {'name': 'Section 2', 'content': 'More content'}
            ],
            'links': ['http://example.com', 'https://test.com']
        }
        
        result = extractor.extract_from_api_response(response_data)
        
        assert result.structure is not None
        assert result.structure.paragraphs > 0
        assert result.structure.lists > 0  # sections array counts as list
        assert result.structure.links > 0


if __name__ == "__main__":
    pytest.main([__file__])