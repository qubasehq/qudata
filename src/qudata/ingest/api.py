"""
API client module for LLMDataForge.

This module provides API integration capabilities for REST and GraphQL endpoints
with authentication, rate limiting, and error handling.
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
import logging

try:
    import requests
    from requests.auth import HTTPBasicAuth, HTTPDigestAuth
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ..models import (
    ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity
)


class AuthConfig:
    """Authentication configuration for API requests."""
    
    def __init__(self, auth_type: str, **kwargs):
        """
        Initialize authentication configuration.
        
        Args:
            auth_type: Type of authentication ('none', 'basic', 'digest', 'bearer', 'api_key', 'oauth2')
            **kwargs: Authentication parameters
        """
        self.auth_type = auth_type.lower()
        self.params = kwargs
    
    def apply_to_session(self, session: requests.Session) -> None:
        """Apply authentication to requests session."""
        if self.auth_type == 'basic':
            username = self.params.get('username')
            password = self.params.get('password')
            if username and password:
                session.auth = HTTPBasicAuth(username, password)
        
        elif self.auth_type == 'digest':
            username = self.params.get('username')
            password = self.params.get('password')
            if username and password:
                session.auth = HTTPDigestAuth(username, password)
        
        elif self.auth_type == 'bearer':
            token = self.params.get('token')
            if token:
                session.headers['Authorization'] = f'Bearer {token}'
        
        elif self.auth_type == 'api_key':
            api_key = self.params.get('api_key')
            header_name = self.params.get('header_name', 'X-API-Key')
            if api_key:
                session.headers[header_name] = api_key
        
        elif self.auth_type == 'oauth2':
            access_token = self.params.get('access_token')
            if access_token:
                session.headers['Authorization'] = f'Bearer {access_token}'


class APIClient:
    """
    Generic API client for REST and GraphQL endpoints.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize API client.
        
        Args:
            config: Configuration dictionary with options like:
                - base_url: Base URL for API endpoints
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Maximum retry attempts (default: 3)
                - requests_per_minute: Rate limit (default: 60)
                - auth: Authentication configuration
                - headers: Default headers to include
                - verify_ssl: Verify SSL certificates (default: True)
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for API integration. "
                "Install it with: pip install requests"
            )
        
        self.config = config or {}
        self.base_url = self.config.get('base_url', '')
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.verify_ssl = self.config.get('verify_ssl', True)
        
        # Rate limiting
        requests_per_minute = self.config.get('requests_per_minute', 60)
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        
        # Setup session
        self.session = requests.Session()
        
        # Apply default headers
        default_headers = self.config.get('headers', {})
        self.session.headers.update(default_headers)
        
        # Apply authentication
        auth_config = self.config.get('auth')
        if auth_config:
            if isinstance(auth_config, dict):
                auth = AuthConfig(**auth_config)
            else:
                auth = auth_config
            auth.apply_to_session(self.session)
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make GET request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            ProcessingError: If request fails
        """
        url = self._build_url(endpoint)
        return self._make_request('GET', url, params=params)
    
    def post(self, endpoint: str, data: Dict[str, Any] = None, json_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make POST request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            data: Form data
            json_data: JSON data
            
        Returns:
            Response data as dictionary
            
        Raises:
            ProcessingError: If request fails
        """
        url = self._build_url(endpoint)
        return self._make_request('POST', url, data=data, json=json_data)
    
    def put(self, endpoint: str, data: Dict[str, Any] = None, json_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make PUT request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            data: Form data
            json_data: JSON data
            
        Returns:
            Response data as dictionary
            
        Raises:
            ProcessingError: If request fails
        """
        url = self._build_url(endpoint)
        return self._make_request('PUT', url, data=data, json=json_data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """
        Make DELETE request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Response data as dictionary
            
        Raises:
            ProcessingError: If request fails
        """
        url = self._build_url(endpoint)
        return self._make_request('DELETE', url)
    
    def graphql_query(self, query: str, variables: Dict[str, Any] = None, endpoint: str = '/graphql') -> Dict[str, Any]:
        """
        Execute GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            endpoint: GraphQL endpoint path (default: /graphql)
            
        Returns:
            GraphQL response data
            
        Raises:
            ProcessingError: If query fails
        """
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        try:
            response_data = self.post(endpoint, json_data=payload)
            
            # Check for GraphQL errors
            if 'errors' in response_data:
                error_messages = [error.get('message', 'Unknown error') for error in response_data['errors']]
                raise ProcessingError(
                    stage="api_request",
                    error_type="GraphQLError",
                    message=f"GraphQL errors: {'; '.join(error_messages)}",
                    severity=ErrorSeverity.HIGH
                )
            
            return response_data.get('data', {})
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                stage="api_request",
                error_type="GraphQLRequestError",
                message=f"GraphQL request failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def fetch_paginated_data(self, endpoint: str, params: Dict[str, Any] = None, 
                           page_param: str = 'page', limit_param: str = 'limit',
                           max_pages: int = None) -> List[Dict[str, Any]]:
        """
        Fetch paginated data from API endpoint.
        
        Args:
            endpoint: API endpoint path
            params: Base query parameters
            page_param: Parameter name for page number
            limit_param: Parameter name for page size
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of all items from all pages
            
        Raises:
            ProcessingError: If request fails
        """
        all_items = []
        page = 1
        params = params or {}
        
        while True:
            # Add pagination parameters
            page_params = params.copy()
            page_params[page_param] = page
            
            try:
                response_data = self.get(endpoint, page_params)
                
                # Extract items (try common response structures)
                items = []
                if isinstance(response_data, list):
                    items = response_data
                elif isinstance(response_data, dict):
                    # Try common pagination response formats
                    for key in ['data', 'items', 'results', 'records']:
                        if key in response_data:
                            items = response_data[key]
                            break
                    else:
                        # If no common key found, use the entire response
                        items = [response_data]
                
                if not items:
                    break
                
                all_items.extend(items)
                
                # Check if we should continue
                if max_pages and page >= max_pages:
                    break
                
                # Check if there are more pages (try common pagination indicators)
                has_more = False
                if isinstance(response_data, dict):
                    # Check common pagination metadata
                    pagination_keys = ['has_more', 'hasMore', 'has_next', 'hasNext']
                    for key in pagination_keys:
                        if response_data.get(key):
                            has_more = True
                            break
                    
                    # Check if current page has fewer items than expected
                    if not has_more and limit_param in page_params:
                        expected_count = page_params[limit_param]
                        if len(items) < expected_count:
                            has_more = False
                        else:
                            has_more = True
                    elif not has_more:
                        # If no pagination metadata, assume more pages if we got items
                        has_more = len(items) > 0
                
                if not has_more:
                    break
                
                page += 1
                
            except ProcessingError as e:
                if page == 1:
                    # If first page fails, re-raise the error
                    raise
                else:
                    # If later page fails, log warning and stop
                    logging.warning(f"Failed to fetch page {page}: {e.message}")
                    break
        
        return all_items
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith(('http://', 'https://')):
            return endpoint
        
        if self.base_url:
            return urljoin(self.base_url.rstrip('/') + '/', endpoint.lstrip('/'))
        else:
            return endpoint
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            ProcessingError: If request fails
        """
        # Rate limiting
        self._wait_for_rate_limit()
        
        # Retry logic
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    **kwargs
                )
                
                # Update rate limit tracking
                self.last_request_time = time.time()
                
                # Check response status
                response.raise_for_status()
                
                # Parse JSON response
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # If response is not JSON, return text content
                    return {'content': response.text, 'status_code': response.status_code}
                
            except requests.exceptions.Timeout as e:
                last_exception = ProcessingError(
                    stage="api_request",
                    error_type="RequestTimeout",
                    message=f"Request timeout for {method} {url}",
                    severity=ErrorSeverity.MEDIUM
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = ProcessingError(
                    stage="api_request",
                    error_type="ConnectionError",
                    message=f"Connection error for {method} {url}",
                    severity=ErrorSeverity.MEDIUM
                )
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                
                # Don't retry client errors (4xx)
                if 400 <= status_code < 500:
                    raise ProcessingError(
                        stage="api_request",
                        error_type="HTTPClientError",
                        message=f"HTTP {status_code} error for {method} {url}: {e.response.text}",
                        severity=ErrorSeverity.HIGH
                    )
                
                last_exception = ProcessingError(
                    stage="api_request",
                    error_type="HTTPServerError",
                    message=f"HTTP {status_code} error for {method} {url}",
                    severity=ErrorSeverity.MEDIUM
                )
            except Exception as e:
                last_exception = ProcessingError(
                    stage="api_request",
                    error_type="RequestError",
                    message=f"Request failed for {method} {url}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    stack_trace=str(e)
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)


class ContentExtractor:
    """
    Extract content from API responses and convert to ExtractedContent format.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize content extractor.
        
        Args:
            config: Configuration dictionary with extraction rules
        """
        self.config = config or {}
    
    def extract_from_api_response(self, response_data: Dict[str, Any], 
                                source_url: str = None, 
                                extraction_rules: Dict[str, Any] = None) -> ExtractedContent:
        """
        Extract content from API response data.
        
        Args:
            response_data: API response data
            source_url: Source URL for metadata
            extraction_rules: Optional extraction configuration
            
        Returns:
            ExtractedContent object
            
        Raises:
            ProcessingError: If extraction fails
        """
        try:
            rules = extraction_rules or self.config.get('extraction_rules', {})
            
            # Extract content based on rules
            content = self._extract_content_from_data(response_data, rules)
            
            # Extract metadata
            metadata = self._extract_metadata_from_data(response_data, source_url, rules)
            
            # Create extracted content
            extracted = ExtractedContent(content, metadata)
            
            # Extract structure if possible
            structure = self._extract_structure_from_data(response_data, rules)
            if structure:
                extracted.structure = structure
            
            return extracted
            
        except Exception as e:
            raise ProcessingError(
                stage="content_extraction",
                error_type="APIContentExtractionError",
                message=f"Failed to extract content from API response: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def _extract_content_from_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> str:
        """Extract main content from API response data."""
        content_fields = rules.get('content_fields', ['content', 'text', 'body', 'description'])
        
        # Try to find content in specified fields
        for field in content_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, (list, dict)):
                    return json.dumps(value, indent=2)
        
        # If no content field found, try to extract from nested objects
        content_parts = []
        
        def extract_text_recursively(obj, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            
            if isinstance(obj, str):
                if len(obj.strip()) > 10:  # Only include meaningful text
                    content_parts.append(obj.strip())
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if any(keyword in key.lower() for keyword in ['text', 'content', 'body', 'description', 'title']):
                        extract_text_recursively(value, max_depth, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursively(item, max_depth, current_depth + 1)
        
        extract_text_recursively(data)
        
        if content_parts:
            return '\n\n'.join(content_parts)
        
        # Fallback: return JSON representation
        return json.dumps(data, indent=2)
    
    def _extract_metadata_from_data(self, data: Dict[str, Any], source_url: str, rules: Dict[str, Any]) -> FileMetadata:
        """Extract metadata from API response data."""
        title_fields = rules.get('title_fields', ['title', 'name', 'subject', 'headline'])
        author_fields = rules.get('author_fields', ['author', 'creator', 'user', 'username'])
        date_fields = rules.get('date_fields', ['created_at', 'published_at', 'date', 'timestamp'])
        
        # Extract title
        title = None
        for field in title_fields:
            if field in data and data[field]:
                title = str(data[field]).strip()
                break
        
        # Extract author
        author = None
        for field in author_fields:
            if field in data and data[field]:
                author_data = data[field]
                if isinstance(author_data, dict):
                    author = author_data.get('name') or author_data.get('username') or str(author_data)
                else:
                    author = str(author_data).strip()
                break
        
        # Extract date
        creation_date = None
        for field in date_fields:
            if field in data and data[field]:
                creation_date = str(data[field])
                break
        
        # Calculate content size
        content_size = len(json.dumps(data).encode('utf-8'))
        
        return FileMetadata(
            file_path=source_url or 'api_response',
            file_type='api_response',
            size_bytes=content_size
        )
    
    def _extract_structure_from_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Optional[DocumentStructure]:
        """Extract document structure from API response data."""
        try:
            # Count different types of content
            paragraphs = 0
            lists = 0
            links = 0
            
            def count_structure_recursively(obj):
                nonlocal paragraphs, lists, links
                
                if isinstance(obj, str):
                    # Count paragraphs (rough estimate)
                    if len(obj.strip()) > 50:
                        paragraphs += obj.count('\n\n') + 1
                    
                    # Count links (rough estimate)
                    links += obj.count('http://') + obj.count('https://')
                
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        count_structure_recursively(value)
                
                elif isinstance(obj, list):
                    lists += 1
                    for item in obj:
                        count_structure_recursively(item)
            
            count_structure_recursively(data)
            
            return DocumentStructure(
                headings=[],  # Hard to extract from API data
                paragraphs=paragraphs,
                tables=0,  # Not applicable for API responses
                images=0,  # Not applicable for API responses
                code_blocks=0,  # Not applicable for API responses
                lists=lists,
                links=links
            )
            
        except Exception:
            return None