"""
Integration tests for streaming data processing.

These tests demonstrate real-world usage scenarios for:
- RSS feed processing
- Log file monitoring
- Stream coordination and processing
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
import pytest

from src.qudata.ingest.stream import (
    StreamProcessor, RSSFeedReader, LogParser,
    StreamConfig, StreamItem
)


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""
    
    def test_rss_feed_processing_workflow(self):
        """Test complete RSS feed processing workflow."""
        # Create a mock RSS feed XML
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <link>https://example.com</link>
        <description>Test RSS feed</description>
        <item>
            <title>First Article</title>
            <link>https://example.com/article1</link>
            <description>This is the first test article</description>
            <pubDate>Mon, 01 Jan 2023 12:00:00 GMT</pubDate>
            <guid>article-1</guid>
        </item>
        <item>
            <title>Second Article</title>
            <link>https://example.com/article2</link>
            <description>This is the second test article</description>
            <pubDate>Mon, 01 Jan 2023 13:00:00 GMT</pubDate>
            <guid>article-2</guid>
        </item>
    </channel>
</rss>"""
        
        # Note: This test would require a mock HTTP server in a real scenario
        # For now, we'll test the configuration and setup
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml',
            update_interval=60,
            rss_user_agent='QuData Test Agent'
        )
        
        reader = RSSFeedReader(config)
        
        # Test configuration
        assert reader.config.source_url == 'https://example.com/feed.xml'
        assert reader.config.update_interval == 60
        assert reader.config.rss_user_agent == 'QuData Test Agent'
        
    def test_log_file_monitoring_workflow(self):
        """Test complete log file monitoring workflow."""
        # Create a temporary log file
        log_entries = [
            '192.168.1.1 - - [01/Jan/2023:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234',
            '192.168.1.2 - - [01/Jan/2023:12:01:00 +0000] "POST /api/users HTTP/1.1" 201 567',
            '192.168.1.3 - - [01/Jan/2023:12:02:00 +0000] "GET /about.html HTTP/1.1" 200 890',
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            for entry in log_entries:
                f.write(entry + '\n')
            temp_log_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_log_path,
                log_format='common',
                update_interval=5
            )
            
            parser = LogParser(config)
            
            # Test connection
            assert parser.connect() is True
            
            # Reset position to read from beginning
            parser.file_position = 0
            
            # Fetch items
            items = list(parser.fetch_items())
            
            # Verify parsed items
            assert len(items) == 3
            
            # Check first item
            first_item = items[0]
            assert '192.168.1.1' in first_item.metadata['ip']
            assert first_item.metadata['status'] == 200
            assert 'GET /index.html' in first_item.content
            
            # Check second item
            second_item = items[1]
            assert '192.168.1.2' in second_item.metadata['ip']
            assert second_item.metadata['status'] == 201
            assert 'POST /api/users' in second_item.content
            
            parser.disconnect()
            
        finally:
            Path(temp_log_path).unlink()
            
    def test_json_log_processing_workflow(self):
        """Test JSON log processing workflow."""
        json_logs = [
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "level": "INFO",
                "message": "User login successful",
                "user_id": "user123",
                "ip": "192.168.1.1"
            },
            {
                "timestamp": "2023-01-01T12:01:00Z",
                "level": "ERROR",
                "message": "Database connection failed",
                "error_code": "DB001",
                "retry_count": 3
            },
            {
                "timestamp": "2023-01-01T12:02:00Z",
                "level": "WARN",
                "message": "High memory usage detected",
                "memory_usage": "85%",
                "threshold": "80%"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            for log_entry in json_logs:
                f.write(json.dumps(log_entry) + '\n')
            temp_log_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_log_path,
                log_format='json',
                update_interval=5
            )
            
            parser = LogParser(config)
            
            # Test connection
            assert parser.connect() is True
            
            # Reset position to read from beginning
            parser.file_position = 0
            
            # Fetch items
            items = list(parser.fetch_items())
            
            # Verify parsed items
            assert len(items) == 3
            
            # Check INFO log
            info_item = items[0]
            assert info_item.content == "User login successful"
            assert info_item.metadata['level'] == 'INFO'
            assert info_item.metadata['user_id'] == 'user123'
            
            # Check ERROR log
            error_item = items[1]
            assert error_item.content == "Database connection failed"
            assert error_item.metadata['level'] == 'ERROR'
            assert error_item.metadata['error_code'] == 'DB001'
            
            # Check WARN log
            warn_item = items[2]
            assert warn_item.content == "High memory usage detected"
            assert warn_item.metadata['level'] == 'WARN'
            assert warn_item.metadata['memory_usage'] == '85%'
            
            parser.disconnect()
            
        finally:
            Path(temp_log_path).unlink()
            
    def test_stream_processor_coordination(self):
        """Test StreamProcessor coordinating multiple streams."""
        processor = StreamProcessor()
        
        # Create test log files
        access_log_entries = [
            '192.168.1.1 - - [01/Jan/2023:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234',
            '192.168.1.2 - - [01/Jan/2023:12:01:00 +0000] "POST /api/data HTTP/1.1" 201 567',
        ]
        
        error_log_entries = [
            '{"timestamp": "2023-01-01T12:00:30Z", "level": "ERROR", "message": "Connection timeout"}',
            '{"timestamp": "2023-01-01T12:01:30Z", "level": "WARN", "message": "Slow query detected"}',
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as access_file:
            for entry in access_log_entries:
                access_file.write(entry + '\n')
            access_log_path = access_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as error_file:
            for entry in error_log_entries:
                error_file.write(entry + '\n')
            error_log_path = error_file.name
            
        try:
            # Add access log stream
            access_config = StreamConfig(
                source_type='log',
                source_path=access_log_path,
                log_format='common'
            )
            processor.add_stream('access_logs', access_config)
            
            # Add error log stream
            error_config = StreamConfig(
                source_type='log',
                source_path=error_log_path,
                log_format='json'
            )
            processor.add_stream('error_logs', error_config)
            
            # Verify streams were added
            assert 'access_logs' in processor.processors
            assert 'error_logs' in processor.processors
            assert isinstance(processor.processors['access_logs'], LogParser)
            assert isinstance(processor.processors['error_logs'], LogParser)
            
            # Test manual item processing (simulating what would happen in real streaming)
            access_parser = processor.processors['access_logs']
            error_parser = processor.processors['error_logs']
            
            # Connect and process access logs
            access_parser.connect()
            access_parser.file_position = 0
            access_items = list(access_parser.fetch_items())
            access_parser.disconnect()
            
            # Connect and process error logs
            error_parser.connect()
            error_parser.file_position = 0
            error_items = list(error_parser.fetch_items())
            error_parser.disconnect()
            
            # Verify items were processed
            assert len(access_items) == 2
            assert len(error_items) == 2
            
            # Simulate adding items to processor
            for item in access_items + error_items:
                processor.processed_items.append(item)
                
            # Test retrieval
            all_items = processor.get_processed_items()
            assert len(all_items) == 4
            
            # Test filtering by timestamp (use timezone-aware datetime)
            from datetime import timezone
            recent_time = datetime(2023, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
            recent_items = processor.get_processed_items(since=recent_time)
            assert len(recent_items) >= 2  # At least the items from 12:01 onwards
            
        finally:
            Path(access_log_path).unlink()
            Path(error_log_path).unlink()
            
    def test_stream_error_handling(self):
        """Test error handling in streaming scenarios."""
        # Test with non-existent log file
        config = StreamConfig(
            source_type='log',
            source_path='/nonexistent/file.log',
            log_format='common'
        )
        
        parser = LogParser(config)
        
        # Should fail to connect
        assert parser.connect() is False
        assert parser.file_handle is None
        
        # Test with invalid JSON log
        invalid_json_log = '{"invalid": json, "missing": quote}'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            f.write(invalid_json_log + '\n')
            f.write('{"valid": "json"}\n')
            temp_log_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_log_path,
                log_format='json'
            )
            
            parser = LogParser(config)
            parser.connect()
            parser.file_position = 0
            
            items = list(parser.fetch_items())
            
            # Should only get the valid JSON entry
            assert len(items) == 1
            assert items[0].metadata['valid'] == 'json'
            
            parser.disconnect()
            
        finally:
            Path(temp_log_path).unlink()
            
    def test_stream_configuration_validation(self):
        """Test stream configuration validation."""
        processor = StreamProcessor()
        
        # Test unsupported stream type
        with pytest.raises(ValueError, match="Unsupported stream type"):
            config = StreamConfig(source_type='unsupported_type')
            processor.add_stream('test', config)
            
        # Test Kafka configuration validation
        kafka_config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topic='test-topic',
            kafka_group_id='test-group'
        )
        
        # Should not raise an error
        processor.add_stream('kafka_test', kafka_config)
        assert 'kafka_test' in processor.processors
        
    def test_stream_item_metadata_extraction(self):
        """Test metadata extraction from different stream types."""
        # Test common log format metadata
        common_log = '192.168.1.100 - admin [01/Jan/2023:15:30:45 +0000] "DELETE /api/users/123 HTTP/1.1" 204 0'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            f.write(common_log + '\n')
            temp_log_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_log_path,
                log_format='common'
            )
            
            parser = LogParser(config)
            parser.connect()
            parser.file_position = 0
            
            items = list(parser.fetch_items())
            
            assert len(items) == 1
            item = items[0]
            
            # Verify metadata extraction
            assert item.metadata['ip'] == '192.168.1.100'
            assert item.metadata['status'] == 204
            assert item.metadata['size'] == 0
            assert 'DELETE /api/users/123' in item.metadata['request']
            assert 'DELETE /api/users/123' in item.content and '204' in item.content
            
            parser.disconnect()
            
        finally:
            Path(temp_log_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__])