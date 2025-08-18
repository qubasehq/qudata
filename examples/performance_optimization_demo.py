#!/usr/bin/env python3
"""
Performance optimization and scaling demonstration.

This example demonstrates the performance optimization components:
- Parallel processing for multi-threaded execution
- Memory management for efficient resource usage
- Caching layer for expensive operations
- Load balancing for distributed processing
- Streaming processing for large files
"""

import time
import tempfile
import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.qudata.performance.parallel import create_parallel_processor
from src.qudata.performance.memory import create_memory_manager
from src.qudata.performance.cache import CacheLayer, CacheConfig
from src.qudata.performance.load_balancer import create_load_balancer, ProcessingRequest
from src.qudata.performance.streaming import create_streaming_processor


def demo_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    print("=== Parallel Processing Demo ===")
    
    def cpu_intensive_task(n):
        """Simulate CPU-intensive work."""
        result = 0
        for i in range(n * 10000):
            result += i * i
        return result
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [cpu_intensive_task(i) for i in range(1, 11)]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    processor = create_parallel_processor(mode="thread", max_workers=4)
    
    start_time = time.time()
    with processor:
        parallel_results = processor.map_function(cpu_intensive_task, range(1, 11))
        parallel_results = [r.result for r in parallel_results if r.success]
    parallel_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    print(f"Results match: {sequential_results == parallel_results}")
    print()


def demo_memory_management():
    """Demonstrate memory management capabilities."""
    print("=== Memory Management Demo ===")
    
    manager = create_memory_manager(enable_monitoring=False)
    
    # Get memory statistics
    stats = manager.get_memory_stats()
    print(f"Total memory: {stats.total_memory / 1024 / 1024 / 1024:.1f} GB")
    print(f"Available memory: {stats.available_memory / 1024 / 1024 / 1024:.1f} GB")
    print(f"Memory usage: {stats.memory_percent:.1f}%")
    print(f"Memory pressure: {stats.pressure_level.value}")
    
    # Calculate optimal chunk size
    chunk_size = manager.get_optimal_chunk_size(
        item_size_bytes=1024,  # 1KB per item
        total_items=100000
    )
    print(f"Optimal chunk size for 1KB items: {chunk_size:,} items")
    
    # Check memory availability
    available_100mb = manager.check_memory_availability(100.0)  # 100MB
    print(f"100MB available: {available_100mb}")
    
    # Force garbage collection
    gc_stats = manager.force_garbage_collection()
    print(f"Memory freed by GC: {gc_stats.get('memory_freed_mb', 0):.1f} MB")
    print()


def demo_caching():
    """Demonstrate caching capabilities."""
    print("=== Caching Demo ===")
    
    config = CacheConfig(enable_disk_cache=False)  # Use memory-only cache for demo
    cache = CacheLayer(config)
    
    expensive_call_count = 0
    
    def expensive_operation(x):
        """Simulate expensive operation."""
        nonlocal expensive_call_count
        expensive_call_count += 1
        time.sleep(0.1)  # Simulate 100ms operation
        return x * x
    
    @cache.cache_function(ttl=60)
    def cached_operation(x):
        return expensive_operation(x)
    
    # First calls - should execute expensive operation
    print("First calls (cache misses):")
    start_time = time.time()
    results1 = [cached_operation(i) for i in range(1, 6)]
    first_time = time.time() - start_time
    
    # Second calls - should use cache
    print("Second calls (cache hits):")
    start_time = time.time()
    results2 = [cached_operation(i) for i in range(1, 6)]
    second_time = time.time() - start_time
    
    print(f"First calls time: {first_time:.2f}s")
    print(f"Second calls time: {second_time:.2f}s")
    speedup = first_time / second_time if second_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")
    print(f"Expensive operations called: {expensive_call_count} times")
    
    # Cache statistics
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats.hit_rate:.1%}")
    print()


def demo_load_balancing():
    """Demonstrate load balancing capabilities."""
    print("=== Load Balancing Demo ===")
    
    lb = create_load_balancer(strategy="round_robin")
    
    # Add workers with different processing times
    def fast_worker(data):
        time.sleep(0.01)  # 10ms
        return f"fast: {data}"
    
    def medium_worker(data):
        time.sleep(0.02)  # 20ms
        return f"medium: {data}"
    
    def slow_worker(data):
        time.sleep(0.03)  # 30ms
        return f"slow: {data}"
    
    lb.add_worker("fast", fast_worker, weight=3.0)
    lb.add_worker("medium", medium_worker, weight=2.0)
    lb.add_worker("slow", slow_worker, weight=1.0)
    
    # Process requests
    requests = [ProcessingRequest(id=f"req_{i}", data=f"task_{i}") for i in range(20)]
    
    start_time = time.time()
    responses = [lb.process_request(req) for req in requests]
    total_time = time.time() - start_time
    
    # Analyze results
    success_count = sum(1 for r in responses if r.success)
    worker_distribution = {}
    
    for response in responses:
        if response.success:
            worker_id = response.worker_id
            worker_distribution[worker_id] = worker_distribution.get(worker_id, 0) + 1
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Success rate: {success_count}/{len(requests)}")
    print(f"Throughput: {len(requests) / total_time:.1f} req/sec")
    print("Worker distribution:")
    for worker_id, count in worker_distribution.items():
        print(f"  {worker_id}: {count} requests")
    
    lb.stop_health_checking()
    print()


def demo_streaming_processing():
    """Demonstrate streaming processing capabilities."""
    print("=== Streaming Processing Demo ===")
    
    # Create a large test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        for i in range(10000):
            data = {"id": i, "value": i * 2, "text": f"Item {i}"}
            f.write(json.dumps(data) + '\n')
        temp_file = f.name
    
    try:
        processor = create_streaming_processor(mode="line_by_line")
        
        processed_count = 0
        total_value = 0
        
        def json_processor(json_obj):
            nonlocal processed_count, total_value
            processed_count += 1
            total_value += json_obj.get('value', 0)
            return json_obj
        
        # Process the file
        start_time = time.time()
        stats = processor.process_json_lines(temp_file, json_processor)
        processing_time = time.time() - start_time
        
        file_size = Path(temp_file).stat().st_size
        
        print(f"File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Throughput: {stats.throughput_mbps:.1f} MB/s")
        print(f"Items processed: {processed_count:,}")
        print(f"Total value: {total_value:,}")
        print(f"Items per second: {processed_count / processing_time:.0f}")
        
    finally:
        Path(temp_file).unlink()
    
    print()


def demo_integrated_performance():
    """Demonstrate integrated performance optimization."""
    print("=== Integrated Performance Demo ===")
    
    # Create components
    memory_manager = create_memory_manager(enable_monitoring=False)
    cache_config = CacheConfig(enable_disk_cache=False)
    cache = CacheLayer(cache_config)
    processor = create_parallel_processor(max_workers=4)
    
    # Get optimal batch size based on memory
    batch_size = memory_manager.get_optimal_batch_size(
        estimated_item_memory=1024  # 1KB per item
    )
    print(f"Optimal batch size: {batch_size:,} items")
    
    # Expensive cached operation
    call_count = 0
    
    def expensive_nlp_operation(text):
        """Simulate expensive NLP operation."""
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate processing time
        return len(text.split())  # Word count
    
    @cache.cache_function(ttl=300)
    def cached_nlp_operation(text):
        return expensive_nlp_operation(text)
    
    # Test data
    texts = [f"This is sample text number {i} with some words." for i in range(100)]
    
    # Process with integrated optimization
    start_time = time.time()
    
    with processor:
        # Process in optimal batches
        results = []
        for i in range(0, len(texts), min(batch_size, 20)):  # Limit batch size for demo
            batch = texts[i:i + min(batch_size, 20)]
            batch_results = processor.map_function(cached_nlp_operation, batch)
            results.extend([r.result for r in batch_results if r.success])
    
    total_time = time.time() - start_time
    
    print(f"Processing time: {total_time:.2f}s")
    print(f"Items processed: {len(results)}")
    print(f"Expensive operations: {call_count}")
    print(f"Cache efficiency: {(len(texts) - call_count) / len(texts):.1%}")
    print(f"Throughput: {len(results) / total_time:.1f} items/sec")
    print(f"Total word count: {sum(results)}")
    
    # Memory stats after processing
    final_stats = memory_manager.get_memory_stats()
    print(f"Final memory usage: {final_stats.memory_percent:.1f}%")
    print()


def main():
    """Run all performance optimization demos."""
    print("QuData Performance Optimization Demo")
    print("=" * 50)
    print()
    
    demo_parallel_processing()
    demo_memory_management()
    demo_caching()
    demo_load_balancing()
    demo_streaming_processing()
    demo_integrated_performance()
    
    print("All demos completed successfully!")


if __name__ == "__main__":
    main()