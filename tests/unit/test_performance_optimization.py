"""
Unit tests for performance optimization and scaling components.

Tests parallel processing, memory management, caching, load balancing,
and streaming processing capabilities.
"""

import pytest
import threading
import time
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.qudata.performance.parallel import (
    ParallelProcessor, ParallelConfig, ProcessingTask, ProcessingMode,
    BatchProcessor, create_parallel_processor
)
from src.qudata.performance.memory import (
    MemoryManager, MemoryConfig, MemoryPressure, MemoryPool,
    create_memory_manager
)
from src.qudata.performance.cache import (
    CacheLayer, CacheConfig, LRUCache, DiskCache, cached,
    get_cache, cache_manager
)
from src.qudata.performance.load_balancer import (
    LoadBalancer, LoadBalancerConfig, WorkerNode, ProcessingRequest,
    LoadBalancingStrategy, create_load_balancer
)
from src.qudata.performance.streaming import (
    StreamingProcessor, StreamingConfig, StreamingMode,
    create_streaming_processor
)


class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    def test_parallel_processor_creation(self):
        """Test parallel processor creation."""
        config = ParallelConfig(max_workers=4, mode=ProcessingMode.THREAD)
        processor = ParallelProcessor(config)
        
        assert processor.config.max_workers == 4
        assert processor.config.mode == ProcessingMode.THREAD
    
    def test_map_function_parallel(self):
        """Test parallel map function."""
        def square(x):
            return x * x
        
        processor = create_parallel_processor(mode="thread", max_workers=2)
        
        with processor:
            items = [1, 2, 3, 4, 5]
            results = processor.map_function(square, items)
            
            assert len(results) == 5
            assert all(result.success for result in results)
            assert [result.result for result in results] == [1, 4, 9, 16, 25]
    
    def test_task_submission(self):
        """Test task submission and execution."""
        def add_numbers(a, b):
            return a + b
        
        processor = create_parallel_processor()
        
        with processor:
            task = ProcessingTask(
                id="test_task",
                function=add_numbers,
                args=(5, 3),
                kwargs={}
            )
            
            task_id = processor.submit_task(task)
            assert task_id == "test_task"
            
            # Wait for completion
            processor.wait_for_completion(timeout=5.0)
            
            progress = processor.get_progress()
            assert progress['completed_tasks'] == 1
    
    def test_batch_processing(self):
        """Test batch processing."""
        def process_item(item):
            return item * 2
        
        processor = create_parallel_processor()
        batch_processor = BatchProcessor(processor, batch_size=2)
        
        items = [1, 2, 3, 4, 5]
        results = batch_processor.process_items(items, process_item)
        
        assert len(results) == 5
        assert all(result.success for result in results)
        assert [result.result for result in results] == [2, 4, 6, 8, 10]
    
    def test_error_handling(self):
        """Test error handling in parallel processing."""
        def failing_function(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2
        
        processor = create_parallel_processor()
        
        with processor:
            items = [1, 2, 3, 4, 5]
            results = processor.map_function(failing_function, items)
            
            assert len(results) == 5
            assert sum(1 for r in results if r.success) == 4
            assert sum(1 for r in results if not r.success) == 1
            
            # Check that the error is captured
            failed_result = next(r for r in results if not r.success)
            assert isinstance(failed_result.error, ValueError)


class TestMemoryManager:
    """Test memory management functionality."""
    
    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        config = MemoryConfig(max_memory_percent=75.0, enable_monitoring=False)
        manager = MemoryManager(config)
        
        assert manager.config.max_memory_percent == 75.0
        assert not manager.config.enable_monitoring
    
    def test_memory_stats(self):
        """Test memory statistics retrieval."""
        manager = create_memory_manager(enable_monitoring=False)
        
        stats = manager.get_memory_stats()
        
        assert stats.total_memory > 0
        assert stats.available_memory > 0
        assert stats.memory_percent >= 0
        assert stats.process_memory > 0
        assert isinstance(stats.pressure_level, MemoryPressure)
    
    def test_optimal_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        manager = create_memory_manager(enable_monitoring=False)
        
        chunk_size = manager.get_optimal_chunk_size(
            item_size_bytes=1024,  # 1KB per item
            total_items=10000
        )
        
        assert chunk_size > 0
        assert chunk_size <= 10000
    
    def test_memory_availability_check(self):
        """Test memory availability checking."""
        manager = create_memory_manager(enable_monitoring=False)
        
        # Should have at least 1MB available
        assert manager.check_memory_availability(1.0)
        
        # Unlikely to have 1TB available
        assert not manager.check_memory_availability(1024 * 1024)
    
    def test_garbage_collection(self):
        """Test forced garbage collection."""
        manager = create_memory_manager(enable_monitoring=False)
        
        gc_stats = manager.force_garbage_collection()
        
        assert 'gen_0' in gc_stats
        assert 'gen_1' in gc_stats
        assert 'gen_2' in gc_stats
        assert 'memory_freed_bytes' in gc_stats
    
    def test_memory_pool(self):
        """Test memory pool functionality."""
        def create_list():
            return []
        
        pool = MemoryPool(create_list, max_size=5)
        
        # Get objects from pool
        obj1 = pool.get()
        obj2 = pool.get()
        
        assert isinstance(obj1, list)
        assert isinstance(obj2, list)
        
        # Return objects to pool
        pool.put(obj1)
        pool.put(obj2)
        
        assert pool.size() == 2
        
        # Get object back from pool
        obj3 = pool.get()
        assert obj3 is obj2  # Should be the last one returned


class TestCacheLayer:
    """Test caching functionality."""
    
    def test_cache_creation(self):
        """Test cache creation."""
        config = CacheConfig(max_entries=1000, default_ttl=3600, enable_disk_cache=False)
        cache = CacheLayer(config)
        
        assert cache.config.max_entries == 1000
        assert cache.config.default_ttl == 3600
    
    def test_basic_caching(self):
        """Test basic cache operations."""
        config = CacheConfig(enable_disk_cache=False)
        cache = CacheLayer(config)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test removal
        assert cache.remove("key1")
        assert cache.get("key1") is None
    
    def test_lru_cache(self):
        """Test LRU cache functionality."""
        lru = LRUCache(max_size=3)
        
        # Fill cache
        lru.put("key1", "value1")
        lru.put("key2", "value2")
        lru.put("key3", "value3")
        
        assert lru.size() == 3
        
        # Access key1 to make it most recent
        assert lru.get("key1") == "value1"
        
        # Add new key, should evict key2 (least recent)
        lru.put("key4", "value4")
        
        assert lru.get("key2") is None  # Evicted
        assert lru.get("key1") == "value1"  # Still there
        assert lru.get("key4") == "value4"  # New key
    
    def test_disk_cache(self):
        """Test disk cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = DiskCache(temp_dir)
            
            try:
                # Test put and get
                disk_cache.put("test_key", {"data": "test_value"})
                result = disk_cache.get("test_key")
                
                assert result == {"data": "test_value"}
                
                # Test removal
                assert disk_cache.remove("test_key")
                assert disk_cache.get("test_key") is None
            finally:
                disk_cache.close()
    
    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        config = CacheConfig(enable_disk_cache=False)
        cache = CacheLayer(config)
        
        call_count = 0
        
        @cache.cache_function(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * x
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Not incremented
        
        # Different argument should execute function
        result3 = expensive_function(6)
        assert result3 == 36
        assert call_count == 2
    
    def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig(enable_disk_cache=False)
        cache = CacheLayer(config)
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5


class TestLoadBalancer:
    """Test load balancing functionality."""
    
    def test_load_balancer_creation(self):
        """Test load balancer creation."""
        config = LoadBalancerConfig(
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            max_retries=3
        )
        lb = LoadBalancer(config)
        
        assert lb.config.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert lb.config.max_retries == 3
    
    def test_worker_management(self):
        """Test adding and removing workers."""
        lb = create_load_balancer()
        
        def worker1(data):
            return f"worker1: {data}"
        
        def worker2(data):
            return f"worker2: {data}"
        
        # Add workers
        assert lb.add_worker("worker1", worker1, weight=1.0)
        assert lb.add_worker("worker2", worker2, weight=2.0)
        
        # Check worker stats
        stats = lb.get_worker_stats()
        assert "worker1" in stats
        assert "worker2" in stats
        assert stats["worker1"]["weight"] == 1.0
        assert stats["worker2"]["weight"] == 2.0
        
        # Remove worker
        assert lb.remove_worker("worker1")
        stats = lb.get_worker_stats()
        assert "worker1" not in stats
    
    def test_round_robin_balancing(self):
        """Test round-robin load balancing."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        lb = LoadBalancer(config)
        
        def worker1(data):
            return f"worker1: {data}"
        
        def worker2(data):
            return f"worker2: {data}"
        
        lb.add_worker("worker1", worker1)
        lb.add_worker("worker2", worker2)
        
        # Process requests
        requests = [
            ProcessingRequest(id="req1", data="test1"),
            ProcessingRequest(id="req2", data="test2"),
            ProcessingRequest(id="req3", data="test3"),
            ProcessingRequest(id="req4", data="test4")
        ]
        
        responses = []
        for req in requests:
            response = lb.process_request(req)
            responses.append(response)
        
        # Check that both workers were used
        worker_ids = [r.worker_id for r in responses]
        assert "worker1" in worker_ids
        assert "worker2" in worker_ids
        
        # Check results
        assert all(r.success for r in responses)
    
    def test_least_connections_balancing(self):
        """Test least connections load balancing."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        lb = LoadBalancer(config)
        
        def fast_worker(data):
            return f"fast: {data}"
        
        def slow_worker(data):
            time.sleep(0.1)  # Simulate slower processing
            return f"slow: {data}"
        
        lb.add_worker("fast", fast_worker)
        lb.add_worker("slow", slow_worker)
        
        # Process multiple requests
        requests = [ProcessingRequest(id=f"req{i}", data=f"test{i}") for i in range(5)]
        
        responses = []
        for req in requests:
            response = lb.process_request(req)
            responses.append(response)
        
        assert all(r.success for r in responses)
    
    def test_error_handling_and_retries(self):
        """Test error handling and retry logic."""
        config = LoadBalancerConfig(max_retries=2)
        lb = LoadBalancer(config)
        
        call_count = 0
        
        def failing_worker(data):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Simulated failure")
            return f"success: {data}"
        
        lb.add_worker("failing", failing_worker)
        
        request = ProcessingRequest(id="test", data="test_data")
        response = lb.process_request(request)
        
        assert response.success
        assert response.result == "success: test_data"
        assert call_count == 3  # Initial call + 2 retries


class TestStreamingProcessor:
    """Test streaming processing functionality."""
    
    def test_streaming_processor_creation(self):
        """Test streaming processor creation."""
        config = StreamingConfig(
            mode=StreamingMode.CHUNK_BY_SIZE,
            chunk_size=1024
        )
        processor = StreamingProcessor(config)
        
        assert processor.config.mode == StreamingMode.CHUNK_BY_SIZE
        assert processor.config.chunk_size == 1024
    
    def test_text_stream_processing(self):
        """Test text stream processing."""
        processor = create_streaming_processor(mode="line_by_line")
        
        text_data = "line1\nline2\nline3\nline4\n"
        processed_lines = []
        
        def line_processor(line):
            processed_lines.append(line.upper())
            return line.upper()
        
        stats = processor.process_text_stream(text_data, line_processor)
        
        assert len(processed_lines) == 4
        assert processed_lines == ["LINE1", "LINE2", "LINE3", "LINE4"]
        assert stats.total_chunks_processed == 4
    
    def test_file_streaming(self):
        """Test file streaming processing."""
        processor = create_streaming_processor(mode="line_by_line")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line1\nline2\nline3\n")
            temp_file = f.name
        
        try:
            processed_lines = []
            
            def line_processor(line):
                processed_lines.append(line)
                return line
            
            stats = processor.process_file(temp_file, line_processor)
            
            assert len(processed_lines) == 3
            assert processed_lines == ["line1", "line2", "line3"]
            assert stats.total_chunks_processed == 3
            
        finally:
            os.unlink(temp_file)
    
    def test_json_lines_processing(self):
        """Test JSONL processing."""
        processor = create_streaming_processor()
        
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"id": 1, "name": "test1"}\n')
            f.write('{"id": 2, "name": "test2"}\n')
            f.write('{"id": 3, "name": "test3"}\n')
            temp_file = f.name
        
        try:
            processed_objects = []
            
            def json_processor(obj):
                processed_objects.append(obj)
                return obj
            
            stats = processor.process_json_lines(temp_file, json_processor)
            
            assert len(processed_objects) == 3
            assert processed_objects[0]["id"] == 1
            assert processed_objects[1]["name"] == "test2"
            
        finally:
            os.unlink(temp_file)
    
    def test_csv_streaming(self):
        """Test CSV streaming processing."""
        processor = create_streaming_processor()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,value\n")
            f.write("1,test1,100\n")
            f.write("2,test2,200\n")
            f.write("3,test3,300\n")
            temp_file = f.name
        
        try:
            processed_rows = []
            
            def row_processor(row):
                processed_rows.append(row)
                return row
            
            stats = processor.process_csv_stream(temp_file, row_processor)
            
            assert len(processed_rows) == 3
            assert processed_rows[0]["id"] == "1"
            assert processed_rows[1]["name"] == "test2"
            assert processed_rows[2]["value"] == "300"
            
        finally:
            os.unlink(temp_file)
    
    def test_chunk_processing(self):
        """Test chunk-based processing."""
        config = StreamingConfig(
            mode=StreamingMode.CHUNK_BY_SIZE,
            chunk_size=10  # Small chunks for testing
        )
        processor = StreamingProcessor(config)
        
        text_data = "This is a longer text that should be split into chunks"
        processed_chunks = []
        
        def chunk_processor(chunk):
            processed_chunks.append(len(chunk))
            return len(chunk)
        
        stats = processor.process_text_stream(text_data, chunk_processor)
        
        assert len(processed_chunks) > 1  # Should be split into multiple chunks
        assert sum(processed_chunks) == len(text_data)
    
    def test_streaming_iterator(self):
        """Test streaming iterator functionality."""
        processor = create_streaming_processor(mode="line_by_line")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line1\nline2\nline3\n")
            temp_file = f.name
        
        try:
            lines = list(processor.create_streaming_iterator(temp_file))
            assert len(lines) == 3
            assert lines == ["line1", "line2", "line3"]
            
        finally:
            os.unlink(temp_file)
    
    def test_progress_tracking(self):
        """Test progress tracking during streaming."""
        processor = create_streaming_processor()
        
        progress_updates = []
        
        def progress_callback(stats):
            progress_updates.append(stats.progress_percent)
        
        processor.add_progress_callback(progress_callback)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
            temp_file = f.name
        
        try:
            def line_processor(line):
                return line
            
            stats = processor.process_file(temp_file, line_processor)
            
            # Should have received progress updates
            assert len(progress_updates) > 0
            
        finally:
            os.unlink(temp_file)


class TestPerformanceIntegration:
    """Test integration between performance components."""
    
    def test_parallel_with_caching(self):
        """Test parallel processing with caching."""
        config = CacheConfig(enable_disk_cache=False)
        cache = CacheLayer(config)
        processor = create_parallel_processor()
        
        expensive_call_count = 0
        call_count_lock = threading.Lock()
        
        def expensive_operation(x):
            nonlocal expensive_call_count
            with call_count_lock:
                expensive_call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * x
        
        @cache.cache_function(ttl=60)
        def cached_operation(x):
            return expensive_operation(x)
        
        # Pre-populate cache with sequential calls to ensure caching works
        for i in [1, 2, 3, 4, 5]:
            cached_operation(i)
        
        # Reset counter
        with call_count_lock:
            expensive_call_count = 0
        
        with processor:
            # Process same values multiple times - should all be cache hits
            items = [1, 2, 3, 1, 2, 3, 4, 5]
            results = processor.map_function(cached_operation, items)
            
            assert all(r.success for r in results)
            # Should have no new expensive calls due to caching
            assert expensive_call_count == 0
    
    def test_streaming_with_parallel_processing(self):
        """Test streaming with parallel processing."""
        streaming_processor = create_streaming_processor(mode="line_by_line")
        parallel_processor = create_parallel_processor()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(100):
                f.write(f"line_{i}\n")
            temp_file = f.name
        
        try:
            processed_lines = []
            
            def process_batch(lines):
                def process_line(line):
                    return line.upper()
                
                with parallel_processor:
                    results = parallel_processor.map_function(process_line, lines)
                    return [r.result for r in results if r.success]
            
            # Process in batches
            batch = []
            batch_size = 10
            
            def line_processor(line):
                nonlocal batch
                batch.append(line)
                
                if len(batch) >= batch_size:
                    results = process_batch(batch)
                    processed_lines.extend(results)
                    batch = []
                
                return line
            
            stats = streaming_processor.process_file(temp_file, line_processor)
            
            # Process remaining batch
            if batch:
                results = process_batch(batch)
                processed_lines.extend(results)
            
            assert len(processed_lines) == 100
            assert all(line.startswith("LINE_") for line in processed_lines)
            
        finally:
            os.unlink(temp_file)
    
    def test_memory_aware_processing(self):
        """Test memory-aware processing."""
        memory_manager = create_memory_manager(enable_monitoring=False)
        processor = create_parallel_processor()
        
        # Get optimal batch size based on memory
        batch_size = memory_manager.get_optimal_batch_size(
            estimated_item_memory=1024  # 1KB per item
        )
        
        assert batch_size > 0
        
        # Use batch size for processing
        items = list(range(1000))
        
        def process_item(x):
            return x * 2
        
        batch_processor = BatchProcessor(processor, batch_size=batch_size)
        results = batch_processor.process_items(items, process_item)
        
        assert len(results) == 1000
        assert all(r.success for r in results)


if __name__ == "__main__":
    pytest.main([__file__])