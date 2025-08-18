"""
Performance benchmarks for optimization and scaling components.

Provides comprehensive benchmarks for parallel processing, memory management,
caching, load balancing, and streaming processing performance.
"""

import time
import tempfile
import json
import csv
import random
import string
import statistics
from pathlib import Path
from typing import List, Dict, Any
import logging

from src.qudata.performance.parallel import create_parallel_processor, ProcessingTask
from src.qudata.performance.memory import create_memory_manager
from src.qudata.performance.cache import CacheLayer, get_cache
from src.qudata.performance.load_balancer import create_load_balancer, ProcessingRequest
from src.qudata.performance.streaming import create_streaming_processor, StreamingMode

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        logger.info(f"Running benchmark: {self.name}")
        start_time = time.time()
        
        try:
            self.results = self._run_benchmark()
            self.results['total_time'] = time.time() - start_time
            self.results['success'] = True
        except Exception as e:
            self.results = {
                'error': str(e),
                'total_time': time.time() - start_time,
                'success': False
            }
            logger.error(f"Benchmark {self.name} failed: {e}")
        
        logger.info(f"Benchmark {self.name} completed in {self.results['total_time']:.2f}s")
        return self.results
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Override this method to implement the benchmark."""
        raise NotImplementedError


class ParallelProcessingBenchmark(PerformanceBenchmark):
    """Benchmark parallel processing performance."""
    
    def __init__(self):
        super().__init__("Parallel Processing")
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Run parallel processing benchmarks."""
        results = {}
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        task_counts = [100, 500, 1000]
        
        for worker_count in worker_counts:
            for task_count in task_counts:
                key = f"workers_{worker_count}_tasks_{task_count}"
                results[key] = self._benchmark_parallel_execution(worker_count, task_count)
        
        # Test different processing modes
        modes = ["thread", "process"]
        for mode in modes:
            key = f"mode_{mode}"
            results[key] = self._benchmark_processing_mode(mode)
        
        return results
    
    def _benchmark_parallel_execution(self, worker_count: int, task_count: int) -> Dict[str, Any]:
        """Benchmark parallel execution with specific parameters."""
        def cpu_intensive_task(n):
            # Simulate CPU-intensive work
            result = 0
            for i in range(n * 1000):
                result += i * i
            return result
        
        processor = create_parallel_processor(mode="thread", max_workers=worker_count)
        
        start_time = time.time()
        
        with processor:
            items = [random.randint(10, 100) for _ in range(task_count)]
            results = processor.map_function(cpu_intensive_task, items)
            
            success_count = sum(1 for r in results if r.success)
            processing_times = [r.processing_time for r in results if r.success]
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'success_rate': success_count / task_count,
            'avg_task_time': statistics.mean(processing_times) if processing_times else 0,
            'throughput_tasks_per_second': task_count / total_time,
            'worker_count': worker_count,
            'task_count': task_count
        }
    
    def _benchmark_processing_mode(self, mode: str) -> Dict[str, Any]:
        """Benchmark different processing modes."""
        def simple_task(x):
            return x * x
        
        processor = create_parallel_processor(mode=mode, max_workers=4)
        task_count = 1000
        
        start_time = time.time()
        
        with processor:
            items = list(range(task_count))
            results = processor.map_function(simple_task, items)
            
            success_count = sum(1 for r in results if r.success)
        
        total_time = time.time() - start_time
        
        return {
            'mode': mode,
            'total_time': total_time,
            'success_rate': success_count / task_count,
            'throughput_tasks_per_second': task_count / total_time
        }


class MemoryManagementBenchmark(PerformanceBenchmark):
    """Benchmark memory management performance."""
    
    def __init__(self):
        super().__init__("Memory Management")
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Run memory management benchmarks."""
        results = {}
        
        # Test memory monitoring overhead
        results['monitoring_overhead'] = self._benchmark_monitoring_overhead()
        
        # Test garbage collection performance
        results['gc_performance'] = self._benchmark_gc_performance()
        
        # Test memory optimization
        results['optimization'] = self._benchmark_memory_optimization()
        
        return results
    
    def _benchmark_monitoring_overhead(self) -> Dict[str, Any]:
        """Benchmark memory monitoring overhead."""
        # Test with monitoring disabled
        manager_no_monitoring = create_memory_manager(enable_monitoring=False)
        
        start_time = time.time()
        for _ in range(1000):
            stats = manager_no_monitoring.get_memory_stats()
        no_monitoring_time = time.time() - start_time
        
        # Test with monitoring enabled
        manager_with_monitoring = create_memory_manager(enable_monitoring=True)
        
        start_time = time.time()
        for _ in range(1000):
            stats = manager_with_monitoring.get_memory_stats()
        with_monitoring_time = time.time() - start_time
        
        manager_with_monitoring.stop_monitoring()
        
        return {
            'no_monitoring_time': no_monitoring_time,
            'with_monitoring_time': with_monitoring_time,
            'overhead_percent': ((with_monitoring_time - no_monitoring_time) / no_monitoring_time) * 100
        }
    
    def _benchmark_gc_performance(self) -> Dict[str, Any]:
        """Benchmark garbage collection performance."""
        manager = create_memory_manager(enable_monitoring=False)
        
        # Create some objects to garbage collect
        large_objects = []
        for _ in range(1000):
            large_objects.append([random.random() for _ in range(1000)])
        
        # Clear references
        large_objects.clear()
        
        # Benchmark garbage collection
        start_time = time.time()
        gc_stats = manager.force_garbage_collection()
        gc_time = time.time() - start_time
        
        return {
            'gc_time': gc_time,
            'memory_freed_mb': gc_stats.get('memory_freed_mb', 0),
            'objects_collected': sum(gc_stats.get(f'gen_{i}', 0) for i in range(3))
        }
    
    def _benchmark_memory_optimization(self) -> Dict[str, Any]:
        """Benchmark memory optimization features."""
        manager = create_memory_manager(enable_monitoring=False)
        
        # Test optimal chunk size calculation
        start_time = time.time()
        for _ in range(1000):
            chunk_size = manager.get_optimal_chunk_size(1024, 10000)
        chunk_calc_time = time.time() - start_time
        
        # Test memory availability checking
        start_time = time.time()
        for _ in range(1000):
            available = manager.check_memory_availability(10.0)  # 10MB
        availability_check_time = time.time() - start_time
        
        return {
            'chunk_calc_time': chunk_calc_time,
            'availability_check_time': availability_check_time,
            'chunk_calc_ops_per_second': 1000 / chunk_calc_time,
            'availability_ops_per_second': 1000 / availability_check_time
        }


class CachingBenchmark(PerformanceBenchmark):
    """Benchmark caching performance."""
    
    def __init__(self):
        super().__init__("Caching")
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Run caching benchmarks."""
        results = {}
        
        # Test cache hit/miss performance
        results['hit_miss_performance'] = self._benchmark_hit_miss_performance()
        
        # Test different cache sizes
        results['cache_size_impact'] = self._benchmark_cache_size_impact()
        
        # Test disk vs memory cache
        results['storage_comparison'] = self._benchmark_storage_comparison()
        
        return results
    
    def _benchmark_hit_miss_performance(self) -> Dict[str, Any]:
        """Benchmark cache hit and miss performance."""
        cache = CacheLayer()
        
        # Populate cache
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Benchmark cache hits
        start_time = time.time()
        for i in range(1000):
            value = cache.get(f"key_{i}")
        hit_time = time.time() - start_time
        
        # Benchmark cache misses
        start_time = time.time()
        for i in range(1000, 2000):
            value = cache.get(f"key_{i}")
        miss_time = time.time() - start_time
        
        stats = cache.get_stats()
        
        return {
            'hit_time': hit_time,
            'miss_time': miss_time,
            'hit_ops_per_second': 1000 / hit_time,
            'miss_ops_per_second': 1000 / miss_time,
            'hit_rate': stats.hit_rate
        }
    
    def _benchmark_cache_size_impact(self) -> Dict[str, Any]:
        """Benchmark impact of different cache sizes."""
        cache_sizes = [100, 500, 1000, 5000]
        results = {}
        
        for cache_size in cache_sizes:
            cache = get_cache(f"size_test_{cache_size}")
            
            # Populate cache beyond its size to test eviction
            start_time = time.time()
            for i in range(cache_size * 2):
                cache.put(f"key_{i}", f"value_{i}")
            populate_time = time.time() - start_time
            
            # Test access performance
            start_time = time.time()
            for i in range(cache_size):
                value = cache.get(f"key_{i}")
            access_time = time.time() - start_time
            
            results[f"size_{cache_size}"] = {
                'populate_time': populate_time,
                'access_time': access_time,
                'populate_ops_per_second': (cache_size * 2) / populate_time,
                'access_ops_per_second': cache_size / access_time
            }
        
        return results
    
    def _benchmark_storage_comparison(self) -> Dict[str, Any]:
        """Compare memory vs disk cache performance."""
        # Memory cache
        memory_cache = CacheLayer()
        
        # Populate and test memory cache
        start_time = time.time()
        for i in range(1000):
            memory_cache.put(f"key_{i}", {"data": f"value_{i}", "number": i})
        memory_write_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            value = memory_cache.get(f"key_{i}")
        memory_read_time = time.time() - start_time
        
        return {
            'memory_write_time': memory_write_time,
            'memory_read_time': memory_read_time,
            'memory_write_ops_per_second': 1000 / memory_write_time,
            'memory_read_ops_per_second': 1000 / memory_read_time
        }


class LoadBalancingBenchmark(PerformanceBenchmark):
    """Benchmark load balancing performance."""
    
    def __init__(self):
        super().__init__("Load Balancing")
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Run load balancing benchmarks."""
        results = {}
        
        # Test different strategies
        strategies = ["round_robin", "least_connections", "random"]
        for strategy in strategies:
            results[f"strategy_{strategy}"] = self._benchmark_strategy(strategy)
        
        # Test worker scaling
        results['worker_scaling'] = self._benchmark_worker_scaling()
        
        # Test error handling performance
        results['error_handling'] = self._benchmark_error_handling()
        
        return results
    
    def _benchmark_strategy(self, strategy: str) -> Dict[str, Any]:
        """Benchmark specific load balancing strategy."""
        lb = create_load_balancer(strategy=strategy)
        
        # Add workers
        def worker1(data):
            time.sleep(0.001)  # Simulate work
            return f"worker1: {data}"
        
        def worker2(data):
            time.sleep(0.002)  # Simulate different processing time
            return f"worker2: {data}"
        
        def worker3(data):
            time.sleep(0.001)
            return f"worker3: {data}"
        
        lb.add_worker("worker1", worker1, weight=1.0)
        lb.add_worker("worker2", worker2, weight=2.0)
        lb.add_worker("worker3", worker3, weight=1.0)
        
        # Process requests
        requests = [ProcessingRequest(id=f"req_{i}", data=f"data_{i}") for i in range(100)]
        
        start_time = time.time()
        responses = []
        for req in requests:
            response = lb.process_request(req)
            responses.append(response)
        total_time = time.time() - start_time
        
        # Analyze results
        success_count = sum(1 for r in responses if r.success)
        worker_distribution = {}
        response_times = []
        
        for response in responses:
            if response.success:
                worker_id = response.worker_id
                worker_distribution[worker_id] = worker_distribution.get(worker_id, 0) + 1
                response_times.append(response.response_time)
        
        lb.stop_health_checking()
        
        return {
            'strategy': strategy,
            'total_time': total_time,
            'success_rate': success_count / len(requests),
            'throughput_requests_per_second': len(requests) / total_time,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'worker_distribution': worker_distribution
        }
    
    def _benchmark_worker_scaling(self) -> Dict[str, Any]:
        """Benchmark performance with different numbers of workers."""
        worker_counts = [1, 2, 4, 8]
        results = {}
        
        for worker_count in worker_counts:
            lb = create_load_balancer()
            
            # Add workers
            for i in range(worker_count):
                def worker(data, worker_id=i):
                    time.sleep(0.001)  # Simulate work
                    return f"worker{worker_id}: {data}"
                
                lb.add_worker(f"worker{i}", worker)
            
            # Process requests
            requests = [ProcessingRequest(id=f"req_{i}", data=f"data_{i}") for i in range(100)]
            
            start_time = time.time()
            responses = [lb.process_request(req) for req in requests]
            total_time = time.time() - start_time
            
            success_count = sum(1 for r in responses if r.success)
            
            results[f"workers_{worker_count}"] = {
                'worker_count': worker_count,
                'total_time': total_time,
                'success_rate': success_count / len(requests),
                'throughput_requests_per_second': len(requests) / total_time
            }
            
            lb.stop_health_checking()
        
        return results
    
    def _benchmark_error_handling(self) -> Dict[str, Any]:
        """Benchmark error handling and retry performance."""
        lb = create_load_balancer()
        
        failure_count = 0
        
        def unreliable_worker(data):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:  # Fail every 3rd request
                raise Exception("Simulated failure")
            return f"success: {data}"
        
        lb.add_worker("unreliable", unreliable_worker)
        
        requests = [ProcessingRequest(id=f"req_{i}", data=f"data_{i}") for i in range(50)]
        
        start_time = time.time()
        responses = [lb.process_request(req) for req in requests]
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in responses if r.success)
        
        lb.stop_health_checking()
        
        return {
            'total_time': total_time,
            'success_rate': success_count / len(requests),
            'failure_rate': failure_count / (failure_count + success_count),
            'throughput_requests_per_second': len(requests) / total_time
        }


class StreamingBenchmark(PerformanceBenchmark):
    """Benchmark streaming processing performance."""
    
    def __init__(self):
        super().__init__("Streaming Processing")
    
    def _run_benchmark(self) -> Dict[str, Any]:
        """Run streaming processing benchmarks."""
        results = {}
        
        # Test different streaming modes
        modes = ["line_by_line", "chunk_by_size", "memory_mapped"]
        for mode in modes:
            results[f"mode_{mode}"] = self._benchmark_streaming_mode(mode)
        
        # Test different file sizes
        results['file_size_scaling'] = self._benchmark_file_size_scaling()
        
        # Test different chunk sizes
        results['chunk_size_impact'] = self._benchmark_chunk_size_impact()
        
        return results
    
    def _benchmark_streaming_mode(self, mode: str) -> Dict[str, Any]:
        """Benchmark specific streaming mode."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(10000):
                f.write(f"This is line {i} with some content to process\n")
            temp_file = f.name
        
        try:
            processor = create_streaming_processor(mode=mode, chunk_size=1024*1024)
            
            processed_count = 0
            
            def line_processor(line):
                nonlocal processed_count
                processed_count += 1
                return line.upper()
            
            start_time = time.time()
            stats = processor.process_file(temp_file, line_processor)
            total_time = time.time() - start_time
            
            file_size = Path(temp_file).stat().st_size
            
            return {
                'mode': mode,
                'total_time': total_time,
                'file_size_mb': file_size / 1024 / 1024,
                'throughput_mbps': stats.throughput_mbps,
                'processed_count': processed_count,
                'chunks_processed': stats.total_chunks_processed
            }
            
        finally:
            Path(temp_file).unlink()
    
    def _benchmark_file_size_scaling(self) -> Dict[str, Any]:
        """Benchmark performance with different file sizes."""
        file_sizes = [1000, 5000, 10000, 50000]  # Number of lines
        results = {}
        
        for line_count in file_sizes:
            # Create test file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                for i in range(line_count):
                    f.write(f"Line {i}: " + "x" * 50 + "\n")  # ~60 chars per line
                temp_file = f.name
            
            try:
                processor = create_streaming_processor(mode="line_by_line")
                
                def line_processor(line):
                    return len(line)
                
                start_time = time.time()
                stats = processor.process_file(temp_file, line_processor)
                total_time = time.time() - start_time
                
                file_size = Path(temp_file).stat().st_size
                
                results[f"lines_{line_count}"] = {
                    'line_count': line_count,
                    'file_size_mb': file_size / 1024 / 1024,
                    'total_time': total_time,
                    'throughput_mbps': stats.throughput_mbps,
                    'lines_per_second': line_count / total_time
                }
                
            finally:
                Path(temp_file).unlink()
        
        return results
    
    def _benchmark_chunk_size_impact(self) -> Dict[str, Any]:
        """Benchmark impact of different chunk sizes."""
        chunk_sizes = [1024, 8192, 65536, 1024*1024]  # 1KB to 1MB
        results = {}
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(10000):
                f.write(f"Line {i}: " + "x" * 100 + "\n")
            temp_file = f.name
        
        try:
            for chunk_size in chunk_sizes:
                processor = create_streaming_processor(
                    mode="chunk_by_size", 
                    chunk_size=chunk_size
                )
                
                def chunk_processor(chunk):
                    return len(chunk)
                
                start_time = time.time()
                stats = processor.process_file(temp_file, chunk_processor)
                total_time = time.time() - start_time
                
                results[f"chunk_{chunk_size}"] = {
                    'chunk_size': chunk_size,
                    'total_time': total_time,
                    'throughput_mbps': stats.throughput_mbps,
                    'chunks_processed': stats.total_chunks_processed
                }
                
        finally:
            Path(temp_file).unlink()
        
        return results


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmark suite for all performance components."""
    
    def __init__(self):
        self.benchmarks = [
            ParallelProcessingBenchmark(),
            MemoryManagementBenchmark(),
            CachingBenchmark(),
            LoadBalancingBenchmark(),
            StreamingBenchmark()
        ]
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        logger.info("Starting comprehensive performance benchmark suite")
        
        suite_start_time = time.time()
        results = {}
        
        for benchmark in self.benchmarks:
            try:
                benchmark_results = benchmark.run()
                results[benchmark.name] = benchmark_results
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
                results[benchmark.name] = {
                    'error': str(e),
                    'success': False
                }
        
        suite_total_time = time.time() - suite_start_time
        
        # Generate summary
        successful_benchmarks = sum(1 for r in results.values() if r.get('success', False))
        total_benchmarks = len(self.benchmarks)
        
        results['_summary'] = {
            'total_time': suite_total_time,
            'successful_benchmarks': successful_benchmarks,
            'total_benchmarks': total_benchmarks,
            'success_rate': successful_benchmarks / total_benchmarks
        }
        
        logger.info(f"Benchmark suite completed in {suite_total_time:.2f}s")
        logger.info(f"Success rate: {successful_benchmarks}/{total_benchmarks}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        summary = results.get('_summary', {})
        print(f"Total Time: {summary.get('total_time', 0):.2f}s")
        print(f"Success Rate: {summary.get('successful_benchmarks', 0)}/{summary.get('total_benchmarks', 0)}")
        print()
        
        for benchmark_name, benchmark_results in results.items():
            if benchmark_name.startswith('_'):
                continue
            
            print(f"{benchmark_name}:")
            if benchmark_results.get('success', False):
                print(f"  ✓ Completed in {benchmark_results.get('total_time', 0):.2f}s")
                
                # Print key metrics for each benchmark
                if benchmark_name == "Parallel Processing":
                    self._print_parallel_metrics(benchmark_results)
                elif benchmark_name == "Memory Management":
                    self._print_memory_metrics(benchmark_results)
                elif benchmark_name == "Caching":
                    self._print_cache_metrics(benchmark_results)
                elif benchmark_name == "Load Balancing":
                    self._print_load_balancing_metrics(benchmark_results)
                elif benchmark_name == "Streaming Processing":
                    self._print_streaming_metrics(benchmark_results)
            else:
                print(f"  ✗ Failed: {benchmark_results.get('error', 'Unknown error')}")
            print()
    
    def _print_parallel_metrics(self, results: Dict[str, Any]):
        """Print parallel processing metrics."""
        for key, value in results.items():
            if key.startswith('workers_') and isinstance(value, dict):
                throughput = value.get('throughput_tasks_per_second', 0)
                print(f"    {key}: {throughput:.1f} tasks/sec")
    
    def _print_memory_metrics(self, results: Dict[str, Any]):
        """Print memory management metrics."""
        if 'gc_performance' in results:
            gc_time = results['gc_performance'].get('gc_time', 0)
            memory_freed = results['gc_performance'].get('memory_freed_mb', 0)
            print(f"    GC Time: {gc_time:.3f}s, Memory Freed: {memory_freed:.1f}MB")
    
    def _print_cache_metrics(self, results: Dict[str, Any]):
        """Print caching metrics."""
        if 'hit_miss_performance' in results:
            hit_ops = results['hit_miss_performance'].get('hit_ops_per_second', 0)
            miss_ops = results['hit_miss_performance'].get('miss_ops_per_second', 0)
            print(f"    Cache Hits: {hit_ops:.0f} ops/sec, Misses: {miss_ops:.0f} ops/sec")
    
    def _print_load_balancing_metrics(self, results: Dict[str, Any]):
        """Print load balancing metrics."""
        for key, value in results.items():
            if key.startswith('strategy_') and isinstance(value, dict):
                throughput = value.get('throughput_requests_per_second', 0)
                print(f"    {key}: {throughput:.1f} req/sec")
    
    def _print_streaming_metrics(self, results: Dict[str, Any]):
        """Print streaming processing metrics."""
        for key, value in results.items():
            if key.startswith('mode_') and isinstance(value, dict):
                throughput = value.get('throughput_mbps', 0)
                print(f"    {key}: {throughput:.1f} MB/s")


def run_performance_benchmarks():
    """Main function to run performance benchmarks."""
    logging.basicConfig(level=logging.INFO)
    
    suite = ComprehensiveBenchmarkSuite()
    results = suite.run_all_benchmarks()
    
    # Print summary
    suite.print_summary(results)
    
    # Save results
    output_file = f"performance_benchmark_results_{int(time.time())}.json"
    suite.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    run_performance_benchmarks()