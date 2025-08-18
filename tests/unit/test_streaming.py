"""
Unit tests for streaming data processing components.

Tests cover:
- RSS feed reading with various feed formats
- Log file parsing for different log formats
- Kafka stream processing
- Stream processor coordination
- Error handling and recovery
"""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.qudata.ingest.stream import (
    StreamProcessor, BaseStreamProcessor, RSSFeedReader, LogParser, KafkaConnector,
    StreamConfig, StreamItem
)


class TestStreamConfig:
    """Test StreamConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig(source_type='rss')
        
        assert config.source_type == 'rss'
        assert config.update_interval == 300
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.encoding == 'utf-8'
        
    def test_rss_config(self):
        """Test RSS-specific configuration."""
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml',
            update_interval=600,
            rss_user_agent='Custom Agent'
        )
        
        assert config.source_url == 'https://example.com/feed.xml'
        assert config.update_interval == 600
        assert config.rss_user_agent == 'Custom Agent'
        
    def test_kafka_config(self):
        """Test Kafka-specific configuration."""
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topic='test-topic',
            kafka_group_id='test-group'
        )
        
        assert config.kafka_bootstrap_servers == ['localhost:9092']
        assert config.kafka_topic == 'test-topic'
        assert config.kafka_group_id == 'test-group'
        
    def test_log_config(self):
        """Test log-specific configuration."""
        config = StreamConfig(
            source_type='log',
            source_path='/var/log/access.log',
            log_format='combined',
            log_pattern=r'^(\S+) (.+)'
        )
        
        assert config.source_path == '/var/log/access.log'
        assert config.log_format == 'combined'
        assert config.log_pattern == r'^(\S+) (.+)'


class TestStreamItem:
    """Test StreamItem dataclass."""
    
    def test_stream_item_creation(self):
        """Test creating a stream item."""
        timestamp = datetime.now()
        metadata = {'key': 'value'}
        
        item = StreamItem(
            content='test content',
            metadata=metadata,
            timestamp=timestamp,
            source='test-source',
            item_id='item-123'
        )
        
        assert item.content == 'test content'
        assert item.metadata == metadata
        assert item.timestamp == timestamp
        assert item.source == 'test-source'
        assert item.item_id == 'item-123'


class TestStreamProcessor:
    """Test main StreamProcessor class."""
    
    def test_add_rss_stream(self):
        """Test adding RSS stream processor."""
        processor = StreamProcessor()
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        
        processor.add_stream('test-rss', config)
        
        assert 'test-rss' in processor.processors
        assert isinstance(processor.processors['test-rss'], RSSFeedReader)
        
    def test_add_log_stream(self):
        """Test adding log stream processor."""
        processor = StreamProcessor()
        config = StreamConfig(
            source_type='log',
            source_path='/var/log/test.log'
        )
        
        processor.add_stream('test-log', config)
        
        assert 'test-log' in processor.processors
        assert isinstance(processor.processors['test-log'], LogParser)
        
    def test_add_kafka_stream(self):
        """Test adding Kafka stream processor."""
        processor = StreamProcessor()
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topic='test-topic'
        )
        
        processor.add_stream('test-kafka', config)
        
        assert 'test-kafka' in processor.processors
        assert isinstance(processor.processors['test-kafka'], KafkaConnector)
        
    def test_unsupported_stream_type(self):
        """Test error handling for unsupported stream type."""
        processor = StreamProcessor()
        config = StreamConfig(source_type='unsupported')
        
        with pytest.raises(ValueError, match="Unsupported stream type"):
            processor.add_stream('test', config)
            
    def test_get_processed_items(self):
        """Test retrieving processed items."""
        processor = StreamProcessor()
        
        # Add some test items
        item1 = StreamItem('content1', {}, datetime.now(), 'source1')
        item2 = StreamItem('content2', {}, datetime.now(), 'source2')
        processor.processed_items = [item1, item2]
        
        items = processor.get_processed_items()
        assert len(items) == 2
        assert items[0] == item1
        assert items[1] == item2
        
    def test_get_processed_items_since(self):
        """Test retrieving processed items since a timestamp."""
        processor = StreamProcessor()
        
        now = datetime.now()
        old_time = now - timedelta(hours=1)
        
        item1 = StreamItem('content1', {}, old_time, 'source1')
        item2 = StreamItem('content2', {}, now, 'source2')
        processor.processed_items = [item1, item2]
        
        items = processor.get_processed_items(since=now - timedelta(minutes=30))
        assert len(items) == 1
        assert items[0] == item2
        
    def test_clear_processed_items(self):
        """Test clearing processed items."""
        processor = StreamProcessor()
        processor.processed_items = [
            StreamItem('content', {}, datetime.now(), 'source')
        ]
        
        processor.clear_processed_items()
        assert len(processor.processed_items) == 0


class TestRSSFeedReader:
    """Test RSS feed reader functionality."""
    
    def test_rss_config_validation(self):
        """Test RSS configuration validation."""
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        
        reader = RSSFeedReader(config)
        assert reader.config.source_url == 'https://example.com/feed.xml'
        
    @patch('src.qudata.ingest.stream.requests.Session')
    def test_connect_success(self, mock_session_class):
        """Test successful RSS connection."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        reader = RSSFeedReader(config)
        
        result = reader.connect()
        
        assert result is True
        assert reader.session == mock_session
        mock_session.headers.update.assert_called_once()
        
    @patch('src.qudata.ingest.stream.requests.Session')
    def test_connect_failure(self, mock_session_class):
        """Test RSS connection failure."""
        mock_session_class.side_effect = Exception("Connection failed")
        
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        reader = RSSFeedReader(config)
        
        result = reader.connect()
        
        assert result is False
        assert reader.session is None
        
    def test_disconnect(self):
        """Test RSS disconnection."""
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        reader = RSSFeedReader(config)
        mock_session = Mock()
        reader.session = mock_session
        
        reader.disconnect()
        
        mock_session.close.assert_called_once()
        assert reader.session is None
        
    @patch('src.qudata.ingest.stream.feedparser.parse')
    def test_fetch_items_success(self, mock_feedparser):
        """Test successful RSS item fetching."""
        # Mock RSS feed data
        mock_feed = Mock()
        mock_feed.bozo = False
        mock_feed.feed.title = 'Test Feed'
        mock_feed.feed.link = 'https://example.com'
        
        mock_entry = Mock()
        mock_entry.title = 'Test Entry'
        mock_entry.link = 'https://example.com/entry1'
        mock_entry.summary = 'Test summary'
        mock_entry.published_parsed = time.struct_time((2023, 1, 1, 12, 0, 0, 0, 1, 0))
        mock_entry.id = 'entry-1'
        mock_entry.author = 'Test Author'
        mock_entry.published = '2023-01-01T12:00:00Z'
        mock_entry.tags = []
        
        # Mock content attribute properly
        mock_content = Mock()
        mock_content.value = 'Test content'
        mock_entry.content = [mock_content]
        
        mock_feed.entries = [mock_entry]
        mock_feedparser.return_value = mock_feed
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<rss>...</rss>'
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        reader = RSSFeedReader(config)
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        reader.session = mock_session
        
        items = list(reader.fetch_items())
        
        assert len(items) == 1
        item = items[0]
        assert item.content == 'Test content'
        assert item.metadata['title'] == 'Test Entry'
        assert item.metadata['link'] == 'https://example.com/entry1'
        assert item.source == 'https://example.com/feed.xml'
        assert item.item_id == 'entry-1'
        
    @patch('src.qudata.ingest.stream.feedparser.parse')
    def test_fetch_items_not_modified(self, mock_feedparser):
        """Test RSS fetch with 304 Not Modified response."""
        mock_response = Mock()
        mock_response.status_code = 304
        
        config = StreamConfig(
            source_type='rss',
            source_url='https://example.com/feed.xml'
        )
        reader = RSSFeedReader(config)
        reader.session = Mock()
        reader.session.get.return_value = mock_response
        
        items = list(reader.fetch_items())
        
        assert len(items) == 0
        mock_feedparser.assert_not_called()
        
    def test_fetch_items_no_url(self):
        """Test RSS fetch with no URL configured."""
        config = StreamConfig(source_type='rss')
        reader = RSSFeedReader(config)
        reader.session = Mock()
        
        items = list(reader.fetch_items())
        
        assert len(items) == 0


class TestLogParser:
    """Test log file parser functionality."""
    
    def test_log_patterns(self):
        """Test log pattern compilation."""
        config = StreamConfig(
            source_type='log',
            log_format='common'
        )
        parser = LogParser(config)
        
        assert parser.pattern is not None
        
    def test_custom_pattern(self):
        """Test custom log pattern."""
        custom_pattern = r'^(\d+) (.+)'
        config = StreamConfig(
            source_type='log',
            log_pattern=custom_pattern
        )
        parser = LogParser(config)
        
        assert parser.pattern.pattern == custom_pattern
        
    def test_connect_success(self):
        """Test successful log file connection."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test log line\n')
            temp_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_path
            )
            parser = LogParser(config)
            
            result = parser.connect()
            
            assert result is True
            assert parser.file_handle is not None
            
            parser.disconnect()
            
        finally:
            Path(temp_path).unlink()
            
    def test_connect_file_not_found(self):
        """Test log file connection with missing file."""
        config = StreamConfig(
            source_type='log',
            source_path='/nonexistent/file.log'
        )
        parser = LogParser(config)
        
        result = parser.connect()
        
        assert result is False
        assert parser.file_handle is None
        
    def test_connect_no_path(self):
        """Test log file connection with no path."""
        config = StreamConfig(source_type='log')
        parser = LogParser(config)
        
        result = parser.connect()
        
        assert result is False
        
    def test_disconnect(self):
        """Test log file disconnection."""
        config = StreamConfig(source_type='log')
        parser = LogParser(config)
        mock_file = Mock()
        parser.file_handle = mock_file
        
        parser.disconnect()
        
        mock_file.close.assert_called_once()
        assert parser.file_handle is None
        
    def test_fetch_items_new_content(self):
        """Test fetching new log entries."""
        log_content = """192.168.1.1 - - [01/Jan/2023:12:00:00 +0000] "GET /test HTTP/1.1" 200 1234
192.168.1.2 - - [01/Jan/2023:12:01:00 +0000] "POST /api HTTP/1.1" 201 567
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(log_content)
            temp_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_path,
                log_format='common'
            )
            parser = LogParser(config)
            parser.connect()
            
            # Reset position to beginning to simulate new content
            parser.file_position = 0
            
            items = list(parser.fetch_items())
            
            assert len(items) == 2
            assert '192.168.1.1' in items[0].metadata['ip']
            assert '192.168.1.2' in items[1].metadata['ip']
            
            parser.disconnect()
            
        finally:
            Path(temp_path).unlink()
            
    def test_parse_json_log(self):
        """Test parsing JSON formatted logs."""
        json_log = '{"timestamp": "2023-01-01T12:00:00Z", "message": "Test message", "level": "INFO"}'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(json_log + '\n')
            temp_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_path,
                log_format='json'
            )
            parser = LogParser(config)
            parser.connect()
            parser.file_position = 0
            
            items = list(parser.fetch_items())
            
            assert len(items) == 1
            item = items[0]
            assert item.content == 'Test message'
            assert item.metadata['level'] == 'INFO'
            
            parser.disconnect()
            
        finally:
            Path(temp_path).unlink()
            
    def test_parse_invalid_json_log(self):
        """Test handling invalid JSON logs."""
        invalid_json = '{"invalid": json}'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(invalid_json + '\n')
            temp_path = f.name
            
        try:
            config = StreamConfig(
                source_type='log',
                source_path=temp_path,
                log_format='json'
            )
            parser = LogParser(config)
            parser.connect()
            parser.file_position = 0
            
            items = list(parser.fetch_items())
            
            # Should skip invalid JSON lines
            assert len(items) == 0
            
            parser.disconnect()
            
        finally:
            Path(temp_path).unlink()


class TestKafkaConnector:
    """Test Kafka connector functionality."""
    
    @patch('src.qudata.ingest.stream.KafkaConsumer')
    def test_connect_success(self, mock_consumer_class):
        """Test successful Kafka connection."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer
        
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topic='test-topic'
        )
        connector = KafkaConnector(config)
        
        result = connector.connect()
        
        assert result is True
        assert connector.consumer == mock_consumer
        mock_consumer_class.assert_called_once()
        
    def test_connect_no_servers(self):
        """Test Kafka connection with no bootstrap servers."""
        config = StreamConfig(source_type='kafka')
        connector = KafkaConnector(config)
        
        result = connector.connect()
        
        assert result is False
        
    def test_connect_no_topic(self):
        """Test Kafka connection with no topic."""
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092']
        )
        connector = KafkaConnector(config)
        
        result = connector.connect()
        
        assert result is False
        
    @patch('src.qudata.ingest.stream.KafkaConsumer')
    def test_connect_failure(self, mock_consumer_class):
        """Test Kafka connection failure."""
        mock_consumer_class.side_effect = Exception("Connection failed")
        
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topic='test-topic'
        )
        connector = KafkaConnector(config)
        
        result = connector.connect()
        
        assert result is False
        
    def test_disconnect(self):
        """Test Kafka disconnection."""
        config = StreamConfig(source_type='kafka')
        connector = KafkaConnector(config)
        mock_consumer = Mock()
        mock_producer = Mock()
        connector.consumer = mock_consumer
        connector.producer = mock_producer
        
        connector.disconnect()
        
        mock_consumer.close.assert_called_once()
        mock_producer.close.assert_called_once()
        assert connector.consumer is None
        assert connector.producer is None
        
    def test_fetch_items_success(self):
        """Test successful Kafka message fetching."""
        # Mock Kafka message
        mock_message = Mock()
        mock_message.value = 'test message'
        mock_message.topic = 'test-topic'
        mock_message.partition = 0
        mock_message.offset = 123
        mock_message.key = b'test-key'
        mock_message.headers = [('header1', b'value1')]
        mock_message.timestamp = 1672574400000  # 2023-01-01 12:00:00 UTC
        
        # Mock topic partition and message batch
        mock_tp = Mock()
        message_batch = {mock_tp: [mock_message]}
        
        config = StreamConfig(source_type='kafka')
        connector = KafkaConnector(config)
        connector.consumer = Mock()
        connector.consumer.poll.return_value = message_batch
        
        items = list(connector.fetch_items())
        
        assert len(items) == 1
        item = items[0]
        assert item.content == 'test message'
        assert item.metadata['topic'] == 'test-topic'
        assert item.metadata['partition'] == 0
        assert item.metadata['offset'] == 123
        assert item.metadata['key'] == 'test-key'
        assert item.source == 'kafka://test-topic'
        
    def test_fetch_items_no_consumer(self):
        """Test Kafka fetch with no consumer."""
        config = StreamConfig(source_type='kafka')
        connector = KafkaConnector(config)
        
        items = list(connector.fetch_items())
        
        assert len(items) == 0
        
    @patch('src.qudata.ingest.stream.KafkaProducer')
    def test_send_message_success(self, mock_producer_class):
        """Test successful Kafka message sending."""
        mock_producer = Mock()
        mock_future = Mock()
        mock_producer.send.return_value = mock_future
        mock_producer_class.return_value = mock_producer
        
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092']
        )
        connector = KafkaConnector(config)
        
        result = connector.send_message('test-topic', 'test message', 'test-key')
        
        assert result is True
        mock_producer.send.assert_called_once()
        mock_future.get.assert_called_once()
        
    @patch('src.qudata.ingest.stream.KafkaProducer')
    def test_send_message_failure(self, mock_producer_class):
        """Test Kafka message sending failure."""
        mock_producer_class.side_effect = Exception("Send failed")
        
        config = StreamConfig(
            source_type='kafka',
            kafka_bootstrap_servers=['localhost:9092']
        )
        connector = KafkaConnector(config)
        
        result = connector.send_message('test-topic', 'test message')
        
        assert result is False


class TestBaseStreamProcessor:
    """Test base stream processor functionality."""
    
    def test_initialization(self):
        """Test base processor initialization."""
        config = StreamConfig(source_type='test')
        
        class TestProcessor(BaseStreamProcessor):
            def connect(self):
                return True
            def disconnect(self):
                pass
            def fetch_items(self):
                return []
                
        processor = TestProcessor(config)
        
        assert processor.config == config
        assert processor.is_running is False
        assert processor.last_update is None
        assert processor.error_count == 0
        
    def test_stop(self):
        """Test stopping stream processor."""
        config = StreamConfig(source_type='test')
        
        class TestProcessor(BaseStreamProcessor):
            def connect(self):
                return True
            def disconnect(self):
                pass
            def fetch_items(self):
                return []
                
        processor = TestProcessor(config)
        processor.is_running = True
        
        processor.stop()
        
        assert processor.is_running is False


if __name__ == '__main__':
    pytest.main([__file__])