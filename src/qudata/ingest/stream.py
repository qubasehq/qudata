"""
Streaming data processing module for RSS feeds, log files, and Kafka streams.

This module provides components for processing streaming data sources including:
- RSS feed processing with configurable update intervals
- Log file parsing for various formats
- Kafka stream processing for real-time data
- Base streaming processor interface
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import feedparser
import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from ..models import Document, DocumentMetadata, ProcessingResult, ProcessingError


logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming data sources."""
    source_type: str  # 'rss', 'log', 'kafka'
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    update_interval: int = 300  # seconds
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 30
    encoding: str = 'utf-8'
    
    # RSS specific
    rss_user_agent: str = 'QuData RSS Reader 1.0'
    
    # Kafka specific
    kafka_bootstrap_servers: List[str] = None
    kafka_topic: str = None
    kafka_group_id: str = 'qudata-consumer'
    
    # Log specific
    log_format: str = 'common'  # 'common', 'combined', 'json', 'custom'
    log_pattern: Optional[str] = None


@dataclass
class StreamItem:
    """Represents a single item from a stream."""
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
    item_id: Optional[str] = None


class BaseStreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_running = False
        self.last_update = None
        self.error_count = 0
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the stream source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the stream source."""
        pass
    
    @abstractmethod
    def fetch_items(self) -> Iterator[StreamItem]:
        """Fetch new items from the stream."""
        pass
    
    def process_stream(self, callback: Callable[[StreamItem], None]) -> None:
        """Process stream items with a callback function."""
        if not self.connect():
            logger.error(f"Failed to connect to stream: {self.config.source_url}")
            return
            
        try:
            self.is_running = True
            while self.is_running:
                try:
                    items = list(self.fetch_items())
                    for item in items:
                        if not self.is_running:
                            break
                        callback(item)
                    
                    self.error_count = 0  # Reset error count on success
                    time.sleep(self.config.update_interval)
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Stream processing error: {e}")
                    
                    if self.error_count >= self.config.max_retries:
                        logger.error("Max retries exceeded, stopping stream")
                        break
                    
                    time.sleep(min(60, self.config.update_interval))
                    
        finally:
            self.disconnect()
    
    def stop(self) -> None:
        """Stop the stream processing."""
        self.is_running = False


class StreamProcessor:
    """Main stream processor that coordinates different stream types."""
    
    def __init__(self):
        self.processors: Dict[str, BaseStreamProcessor] = {}
        self.processed_items: List[StreamItem] = []
        
    def add_stream(self, name: str, config: StreamConfig) -> None:
        """Add a new stream processor."""
        if config.source_type == 'rss':
            processor = RSSFeedReader(config)
        elif config.source_type == 'log':
            processor = LogParser(config)
        elif config.source_type == 'kafka':
            processor = KafkaConnector(config)
        else:
            raise ValueError(f"Unsupported stream type: {config.source_type}")
            
        self.processors[name] = processor
        
    def start_stream(self, name: str) -> None:
        """Start processing a specific stream."""
        if name not in self.processors:
            raise ValueError(f"Stream '{name}' not found")
            
        processor = self.processors[name]
        
        def item_callback(item: StreamItem):
            self.processed_items.append(item)
            logger.info(f"Processed stream item from {item.source}: {item.item_id}")
            
        processor.process_stream(item_callback)
        
    def stop_stream(self, name: str) -> None:
        """Stop processing a specific stream."""
        if name in self.processors:
            self.processors[name].stop()
            
    def stop_all_streams(self) -> None:
        """Stop all stream processors."""
        for processor in self.processors.values():
            processor.stop()
            
    def get_processed_items(self, since: Optional[datetime] = None) -> List[StreamItem]:
        """Get processed items, optionally filtered by timestamp."""
        if since is None:
            return self.processed_items.copy()
        
        return [item for item in self.processed_items if item.timestamp >= since]
        
    def clear_processed_items(self) -> None:
        """Clear the processed items cache."""
        self.processed_items.clear()


class RSSFeedReader(BaseStreamProcessor):
    """RSS feed reader with configurable update intervals."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.session = None
        self.last_etag = None
        self.last_modified = None
        
    def connect(self) -> bool:
        """Establish HTTP session for RSS feed."""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': self.config.rss_user_agent
            })
            return True
        except Exception as e:
            logger.error(f"Failed to create RSS session: {e}")
            return False
            
    def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()
            self.session = None
            
    def fetch_items(self) -> Iterator[StreamItem]:
        """Fetch new RSS feed items."""
        if not self.config.source_url:
            return
            
        try:
            # Use conditional requests to avoid unnecessary downloads
            headers = {}
            if self.last_etag:
                headers['If-None-Match'] = self.last_etag
            if self.last_modified:
                headers['If-Modified-Since'] = self.last_modified
                
            response = self.session.get(
                self.config.source_url,
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 304:
                # Not modified, no new items
                return
                
            response.raise_for_status()
            
            # Update conditional request headers
            self.last_etag = response.headers.get('ETag')
            self.last_modified = response.headers.get('Last-Modified')
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS feed parsing warning: {feed.bozo_exception}")
                
            # Process feed entries
            for entry in feed.entries:
                # Skip items we've already processed
                published = getattr(entry, 'published_parsed', None)
                if published:
                    pub_date = datetime(*published[:6])
                    if self.last_update and pub_date <= self.last_update:
                        continue
                        
                # Extract content
                content = ""
                if hasattr(entry, 'content'):
                    content = entry.content[0].value if entry.content else ""
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description
                    
                # Create metadata
                metadata = {
                    'title': getattr(entry, 'title', ''),
                    'link': getattr(entry, 'link', ''),
                    'author': getattr(entry, 'author', ''),
                    'published': getattr(entry, 'published', ''),
                    'tags': [tag.term for tag in getattr(entry, 'tags', [])],
                    'feed_title': getattr(feed.feed, 'title', ''),
                    'feed_link': getattr(feed.feed, 'link', ''),
                }
                
                yield StreamItem(
                    content=content,
                    metadata=metadata,
                    timestamp=pub_date if published else datetime.now(),
                    source=self.config.source_url,
                    item_id=getattr(entry, 'id', getattr(entry, 'link', ''))
                )
                
            self.last_update = datetime.now()
            
        except requests.RequestException as e:
            logger.error(f"RSS feed request error: {e}")
        except Exception as e:
            logger.error(f"RSS feed processing error: {e}")


class LogParser(BaseStreamProcessor):
    """Log file parser for various log formats."""
    
    # Common log format patterns
    LOG_PATTERNS = {
        'common': r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(.+?)" (\d{3}) (\d+|-)',
        'combined': r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(.+?)" (\d{3}) (\d+|-) "([^"]*)" "([^"]*)"',
        'json': None,  # Special handling for JSON logs
        'nginx_error': r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (\d+)#(\d+): (.+)',
        'apache_error': r'^\[([^\]]+)\] \[([^\]]+)\] (.+)',
    }
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.file_handle = None
        self.file_position = 0
        self.pattern = None
        
        # Compile regex pattern
        if config.log_pattern:
            self.pattern = re.compile(config.log_pattern)
        elif config.log_format in self.LOG_PATTERNS:
            pattern_str = self.LOG_PATTERNS[config.log_format]
            if pattern_str:
                self.pattern = re.compile(pattern_str)
                
    def connect(self) -> bool:
        """Open log file for reading."""
        try:
            if not self.config.source_path:
                logger.error("Log file path not specified")
                return False
                
            log_path = Path(self.config.source_path)
            if not log_path.exists():
                logger.error(f"Log file not found: {log_path}")
                return False
                
            self.file_handle = open(log_path, 'r', encoding=self.config.encoding)
            
            # Seek to end of file to only read new entries
            self.file_handle.seek(0, 2)  # Seek to end
            self.file_position = self.file_handle.tell()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to open log file: {e}")
            return False
            
    def disconnect(self) -> None:
        """Close log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
    def fetch_items(self) -> Iterator[StreamItem]:
        """Fetch new log entries."""
        if not self.file_handle:
            return
            
        try:
            # Check if file has new content
            self.file_handle.seek(0, 2)  # Seek to end
            current_size = self.file_handle.tell()
            
            if current_size < self.file_position:
                # File was truncated/rotated, start from beginning
                self.file_position = 0
                
            if current_size == self.file_position:
                # No new content
                return
                
            # Seek to last position and read new lines
            self.file_handle.seek(self.file_position)
            
            for line in self.file_handle:
                line = line.strip()
                if not line:
                    continue
                    
                item = self._parse_log_line(line)
                if item:
                    yield item
                    
            # Update position
            self.file_position = self.file_handle.tell()
            
        except Exception as e:
            logger.error(f"Log parsing error: {e}")
            
    def _parse_log_line(self, line: str) -> Optional[StreamItem]:
        """Parse a single log line."""
        try:
            if self.config.log_format == 'json':
                return self._parse_json_log(line)
            elif self.pattern:
                return self._parse_regex_log(line)
            else:
                # Fallback: treat as plain text
                return StreamItem(
                    content=line,
                    metadata={'raw_line': line},
                    timestamp=datetime.now(),
                    source=self.config.source_path,
                    item_id=None
                )
                
        except Exception as e:
            logger.warning(f"Failed to parse log line: {e}")
            return None
            
    def _parse_json_log(self, line: str) -> Optional[StreamItem]:
        """Parse JSON formatted log line."""
        import json
        
        try:
            log_data = json.loads(line)
            
            # Extract timestamp
            timestamp = datetime.now()
            for ts_field in ['timestamp', 'time', '@timestamp', 'datetime']:
                if ts_field in log_data:
                    try:
                        timestamp = datetime.fromisoformat(log_data[ts_field].replace('Z', '+00:00'))
                        break
                    except:
                        continue
                        
            # Extract message content
            content = ""
            for msg_field in ['message', 'msg', 'text', 'content']:
                if msg_field in log_data:
                    content = str(log_data[msg_field])
                    break
                    
            if not content:
                content = line  # Use full JSON as content
                
            return StreamItem(
                content=content,
                metadata=log_data,
                timestamp=timestamp,
                source=self.config.source_path,
                item_id=log_data.get('id')
            )
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON log line: {line[:100]}...")
            return None
            
    def _parse_regex_log(self, line: str) -> Optional[StreamItem]:
        """Parse log line using regex pattern."""
        match = self.pattern.match(line)
        if not match:
            return None
            
        groups = match.groups()
        
        # Common log format parsing
        if self.config.log_format == 'common':
            ip, timestamp_str, request, status, size = groups
            metadata = {
                'ip': ip,
                'request': request,
                'status': int(status),
                'size': int(size) if size != '-' else 0,
            }
            content = f"{request} -> {status}"
            
        elif self.config.log_format == 'combined':
            ip, timestamp_str, request, status, size, referer, user_agent = groups
            metadata = {
                'ip': ip,
                'request': request,
                'status': int(status),
                'size': int(size) if size != '-' else 0,
                'referer': referer,
                'user_agent': user_agent,
            }
            content = f"{request} -> {status}"
            
        else:
            # Generic handling
            metadata = {'groups': groups}
            content = line
            timestamp_str = groups[0] if groups else None
            
        # Parse timestamp
        timestamp = datetime.now()
        if timestamp_str:
            try:
                # Try common timestamp formats
                for fmt in ['%d/%b/%Y:%H:%M:%S %z', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
            except:
                pass
                
        return StreamItem(
            content=content,
            metadata=metadata,
            timestamp=timestamp,
            source=self.config.source_path,
            item_id=None
        )


class KafkaConnector(BaseStreamProcessor):
    """Kafka stream processor for real-time data processing."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.consumer = None
        self.producer = None
        
    def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            if not self.config.kafka_bootstrap_servers:
                logger.error("Kafka bootstrap servers not specified")
                return False
                
            if not self.config.kafka_topic:
                logger.error("Kafka topic not specified")
                return False
                
            self.consumer = KafkaConsumer(
                self.config.kafka_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                value_deserializer=lambda x: x.decode(self.config.encoding),
                consumer_timeout_ms=self.config.timeout * 1000,
                max_poll_records=self.config.batch_size,
                enable_auto_commit=True,
                auto_offset_reset='latest'  # Only read new messages
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
            
        if self.producer:
            self.producer.close()
            self.producer = None
            
    def fetch_items(self) -> Iterator[StreamItem]:
        """Fetch messages from Kafka topic."""
        if not self.consumer:
            return
            
        try:
            # Poll for messages
            message_batch = self.consumer.poll(timeout_ms=self.config.timeout * 1000)
            
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    # Extract message content
                    content = message.value
                    
                    # Create metadata
                    metadata = {
                        'topic': message.topic,
                        'partition': message.partition,
                        'offset': message.offset,
                        'key': message.key.decode(self.config.encoding) if message.key else None,
                        'headers': dict(message.headers) if message.headers else {},
                    }
                    
                    # Convert timestamp
                    timestamp = datetime.fromtimestamp(message.timestamp / 1000) if message.timestamp else datetime.now()
                    
                    yield StreamItem(
                        content=content,
                        metadata=metadata,
                        timestamp=timestamp,
                        source=f"kafka://{message.topic}",
                        item_id=f"{message.topic}-{message.partition}-{message.offset}"
                    )
                    
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
        except Exception as e:
            logger.error(f"Kafka message processing error: {e}")
            
    def send_message(self, topic: str, message: str, key: Optional[str] = None) -> bool:
        """Send a message to Kafka topic (for testing/integration)."""
        try:
            if not self.producer:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    value_serializer=lambda x: x.encode(self.config.encoding)
                )
                
            future = self.producer.send(
                topic,
                value=message,
                key=key.encode(self.config.encoding) if key else None
            )
            
            # Wait for send to complete
            future.get(timeout=self.config.timeout)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Kafka message: {e}")
            return False