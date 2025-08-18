#!/usr/bin/env python3
"""
Streaming Data Processing Demo

This script demonstrates the streaming data processing capabilities of QuData,
including RSS feed reading, log file monitoring, and Kafka stream processing.

Usage:
    python examples/streaming_demo.py
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qudata.ingest.stream import (
    StreamProcessor, RSSFeedReader, LogParser, KafkaConnector,
    StreamConfig, StreamItem
)


def demo_rss_feed_processing():
    """Demonstrate RSS feed processing configuration."""
    print("=== RSS Feed Processing Demo ===")
    
    # Configure RSS feed reader
    config = StreamConfig(
        source_type='rss',
        source_url='https://feeds.feedburner.com/oreilly/radar',  # O'Reilly Radar feed
        update_interval=300,  # Check every 5 minutes
        rss_user_agent='QuData RSS Reader Demo 1.0'
    )
    
    reader = RSSFeedReader(config)
    
    print(f"RSS Feed URL: {config.source_url}")
    print(f"Update Interval: {config.update_interval} seconds")
    print(f"User Agent: {config.rss_user_agent}")
    
    # Note: In a real scenario, you would call reader.process_stream() 
    # with a callback function to handle incoming items
    print("RSS reader configured successfully!")
    print("To start processing: reader.process_stream(callback_function)")
    print()


def demo_log_file_monitoring():
    """Demonstrate log file monitoring with different formats."""
    print("=== Log File Monitoring Demo ===")
    
    # Create sample log files
    sample_logs = {
        'access.log': [
            '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234',
            '192.168.1.2 - - [01/Jan/2024:12:01:00 +0000] "POST /api/users HTTP/1.1" 201 567',
            '192.168.1.3 - - [01/Jan/2024:12:02:00 +0000] "GET /about.html HTTP/1.1" 200 890',
        ],
        'error.log': [
            '{"timestamp": "2024-01-01T12:00:30Z", "level": "ERROR", "message": "Database connection failed", "error_code": "DB001"}',
            '{"timestamp": "2024-01-01T12:01:30Z", "level": "WARN", "message": "High memory usage detected", "memory_usage": "85%"}',
            '{"timestamp": "2024-01-01T12:02:30Z", "level": "INFO", "message": "Backup completed successfully", "backup_size": "2.5GB"}',
        ]
    }
    
    temp_files = {}
    
    try:
        # Create temporary log files
        for log_name, entries in sample_logs.items():
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_{log_name}')
            for entry in entries:
                temp_file.write(entry + '\n')
            temp_file.close()
            temp_files[log_name] = temp_file.name
            
        # Configure access log parser (Common Log Format)
        access_config = StreamConfig(
            source_type='log',
            source_path=temp_files['access.log'],
            log_format='common',
            update_interval=5
        )
        
        access_parser = LogParser(access_config)
        
        # Configure error log parser (JSON format)
        error_config = StreamConfig(
            source_type='log',
            source_path=temp_files['error.log'],
            log_format='json',
            update_interval=5
        )
        
        error_parser = LogParser(error_config)
        
        print("Processing access logs (Common Log Format):")
        access_parser.connect()
        access_parser.file_position = 0  # Start from beginning for demo
        
        for item in access_parser.fetch_items():
            print(f"  IP: {item.metadata.get('ip', 'N/A')}")
            print(f"  Request: {item.metadata.get('request', 'N/A')}")
            print(f"  Status: {item.metadata.get('status', 'N/A')}")
            print(f"  Content: {item.content}")
            print(f"  Timestamp: {item.timestamp}")
            print()
            
        access_parser.disconnect()
        
        print("Processing error logs (JSON format):")
        error_parser.connect()
        error_parser.file_position = 0  # Start from beginning for demo
        
        for item in error_parser.fetch_items():
            print(f"  Level: {item.metadata.get('level', 'N/A')}")
            print(f"  Message: {item.content}")
            print(f"  Error Code: {item.metadata.get('error_code', 'N/A')}")
            print(f"  Memory Usage: {item.metadata.get('memory_usage', 'N/A')}")
            print(f"  Timestamp: {item.timestamp}")
            print()
            
        error_parser.disconnect()
        
    finally:
        # Clean up temporary files
        for temp_path in temp_files.values():
            try:
                Path(temp_path).unlink()
            except:
                pass  # Ignore cleanup errors
                
    print("Log file monitoring demo completed!")
    print()


def demo_kafka_stream_processing():
    """Demonstrate Kafka stream processing configuration."""
    print("=== Kafka Stream Processing Demo ===")
    
    # Configure Kafka connector
    config = StreamConfig(
        source_type='kafka',
        kafka_bootstrap_servers=['localhost:9092'],
        kafka_topic='qudata-events',
        kafka_group_id='qudata-consumer-group',
        batch_size=50,
        timeout=30
    )
    
    connector = KafkaConnector(config)
    
    print(f"Kafka Bootstrap Servers: {config.kafka_bootstrap_servers}")
    print(f"Topic: {config.kafka_topic}")
    print(f"Consumer Group: {config.kafka_group_id}")
    print(f"Batch Size: {config.batch_size}")
    
    # Note: This would require a running Kafka instance
    print("Kafka connector configured successfully!")
    print("Note: Requires running Kafka instance to actually connect")
    print("To start processing: connector.process_stream(callback_function)")
    print()


def demo_stream_processor_coordination():
    """Demonstrate coordinating multiple streams with StreamProcessor."""
    print("=== Stream Processor Coordination Demo ===")
    
    processor = StreamProcessor()
    
    # Create sample data
    sample_data = {
        'web_access.log': [
            '10.0.0.1 - - [01/Jan/2024:14:30:00 +0000] "GET /api/health HTTP/1.1" 200 25',
            '10.0.0.2 - - [01/Jan/2024:14:31:00 +0000] "POST /api/login HTTP/1.1" 200 156',
        ],
        'app_events.log': [
            '{"timestamp": "2024-01-01T14:30:15Z", "event": "user_signup", "user_id": "user123", "source": "web"}',
            '{"timestamp": "2024-01-01T14:31:15Z", "event": "purchase", "user_id": "user456", "amount": 29.99}',
        ]
    }
    
    temp_files = {}
    
    try:
        # Create temporary files
        for log_name, entries in sample_data.items():
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_{log_name}')
            for entry in entries:
                temp_file.write(entry + '\n')
            temp_file.close()
            temp_files[log_name] = temp_file.name
            
        # Add web access log stream
        web_config = StreamConfig(
            source_type='log',
            source_path=temp_files['web_access.log'],
            log_format='common'
        )
        processor.add_stream('web_access', web_config)
        
        # Add application events stream
        app_config = StreamConfig(
            source_type='log',
            source_path=temp_files['app_events.log'],
            log_format='json'
        )
        processor.add_stream('app_events', app_config)
        
        print(f"Added {len(processor.processors)} streams:")
        for name in processor.processors.keys():
            print(f"  - {name}")
        print()
        
        # Simulate processing items from both streams
        all_items = []
        
        for stream_name, stream_processor in processor.processors.items():
            print(f"Processing {stream_name}:")
            stream_processor.connect()
            stream_processor.file_position = 0  # Start from beginning for demo
            
            items = list(stream_processor.fetch_items())
            all_items.extend(items)
            
            for item in items:
                print(f"  Source: {stream_name}")
                print(f"  Content: {item.content}")
                print(f"  Metadata keys: {list(item.metadata.keys())}")
                print(f"  Timestamp: {item.timestamp}")
                print()
                
            stream_processor.disconnect()
            
        # Add items to processor for tracking
        processor.processed_items.extend(all_items)
        
        print(f"Total items processed: {len(processor.get_processed_items())}")
        
        # Demonstrate filtering by timestamp
        from datetime import timezone
        filter_time = datetime(2024, 1, 1, 14, 31, 0, tzinfo=timezone.utc)
        recent_items = processor.get_processed_items(since=filter_time)
        print(f"Items since {filter_time}: {len(recent_items)}")
        
    finally:
        # Clean up
        for temp_path in temp_files.values():
            try:
                Path(temp_path).unlink()
            except:
                pass
                
    print("Stream coordination demo completed!")
    print()


def demo_custom_log_patterns():
    """Demonstrate custom log pattern parsing."""
    print("=== Custom Log Pattern Demo ===")
    
    # Create sample nginx error log
    nginx_error_log = [
        '2024/01/01 14:30:45 [error] 12345#0: *1 connect() failed (111: Connection refused)',
        '2024/01/01 14:31:45 [warn] 12345#0: *2 upstream server temporarily disabled',
        '2024/01/01 14:32:45 [info] 12345#0: *3 client disconnected, bytes from/to client:0/0',
    ]
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_nginx_error.log')
    for entry in nginx_error_log:
        temp_file.write(entry + '\n')
    temp_file.close()
    
    try:
        # Configure with custom pattern for nginx error logs
        config = StreamConfig(
            source_type='log',
            source_path=temp_file.name,
            log_pattern=r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (\d+)#(\d+): (.+)'
        )
        
        parser = LogParser(config)
        
        print("Processing nginx error logs with custom pattern:")
        parser.connect()
        parser.file_position = 0
        
        for item in parser.fetch_items():
            groups = item.metadata.get('groups', [])
            if len(groups) >= 5:
                timestamp_str, level, pid, worker_id, message = groups[:5]
                print(f"  Timestamp: {timestamp_str}")
                print(f"  Level: {level}")
                print(f"  PID: {pid}")
                print(f"  Worker ID: {worker_id}")
                print(f"  Message: {message}")
                print(f"  Content: {item.content}")
                print()
                
        parser.disconnect()
        
    finally:
        try:
            Path(temp_file.name).unlink()
        except:
            pass
            
    print("Custom log pattern demo completed!")
    print()


def main():
    """Run all streaming demos."""
    print("QuData Streaming Data Processing Demo")
    print("=" * 50)
    print()
    
    # Run individual demos
    demo_rss_feed_processing()
    demo_log_file_monitoring()
    demo_kafka_stream_processing()
    demo_stream_processor_coordination()
    demo_custom_log_patterns()
    
    print("All streaming demos completed!")
    print()
    print("Key Features Demonstrated:")
    print("- RSS feed processing with configurable intervals")
    print("- Log file monitoring for multiple formats (Common, JSON, Custom)")
    print("- Kafka stream processing configuration")
    print("- Multi-stream coordination and management")
    print("- Custom log pattern parsing")
    print("- Metadata extraction and content processing")
    print("- Error handling and recovery mechanisms")


if __name__ == '__main__':
    main()