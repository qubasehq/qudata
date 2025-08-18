"""
Parallel processing implementation for multi-threaded execution.

Provides thread-safe parallel processing capabilities for document processing,
with configurable thread pools and task distribution.
"""

import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    THREAD = "thread"
    PROCESS = "process"
    HYBRID = "hybrid"


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None
    mode: ProcessingMode = ProcessingMode.THREAD
    chunk_size: int = 1
    timeout: Optional[float] = None
    enable_progress_tracking: bool = True
    queue_size: int = 1000


class ParallelProcessor:
    """
    Multi-threaded/multi-process processor for handling large workloads.
    
    Supports both thread-based and process-based parallelism with configurable
    worker pools, task queues, and progress tracking.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize the parallel processor."""
        self.config = config or ParallelConfig()
        self._executor = None
        self._task_queue = queue.Queue(maxsize=self.config.queue_size)
        self._results = {}
        self._lock = threading.Lock()
        self._active_tasks = set()
        self._completed_tasks = 0
        self._total_tasks = 0
        
        # Auto-detect optimal worker count if not specified
        if self.config.max_workers is None:
            if self.config.mode == ProcessingMode.THREAD:
                self.config.max_workers = min(32, (mp.cpu_count() or 1) + 4)
            else:
                self.config.max_workers = mp.cpu_count() or 1
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def start(self):
        """Start the executor."""
        if self._executor is not None:
            return
            
        if self.config.mode == ProcessingMode.THREAD:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        elif self.config.mode == ProcessingMode.PROCESS:
            self._executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            # Hybrid mode - use threads for I/O bound, processes for CPU bound
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Started parallel processor with {self.config.max_workers} workers in {self.config.mode.value} mode")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("Parallel processor shutdown complete")
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a single task for processing."""
        if self._executor is None:
            self.start()
        
        with self._lock:
            self._active_tasks.add(task.id)
            self._total_tasks += 1
        
        future = self._executor.submit(self._execute_task, task)
        future.add_done_callback(lambda f: self._task_completed(task.id, f))
        
        return task.id
    
    def submit_batch(self, tasks: List[ProcessingTask]) -> List[str]:
        """Submit a batch of tasks for processing."""
        task_ids = []
        for task in tasks:
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    def map_function(self, 
                    function: Callable, 
                    items: List[Any], 
                    *args, 
                    **kwargs) -> List[ProcessingResult]:
        """
        Map a function over a list of items in parallel.
        
        Args:
            function: Function to apply to each item
            items: List of items to process
            *args: Additional arguments to pass to function
            **kwargs: Additional keyword arguments to pass to function
            
        Returns:
            List of ProcessingResult objects
        """
        if self._executor is None:
            self.start()
        
        # Create tasks
        tasks = []
        for i, item in enumerate(items):
            task = ProcessingTask(
                id=f"map_task_{i}",
                function=function,
                args=(item,) + args,
                kwargs=kwargs,
                timeout=self.config.timeout
            )
            tasks.append(task)
        
        # Submit and collect results
        return self.process_batch(tasks)
    
    def process_batch(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """
        Process a batch of tasks and return results.
        
        Args:
            tasks: List of ProcessingTask objects
            
        Returns:
            List of ProcessingResult objects in the same order as input tasks
        """
        if self._executor is None:
            self.start()
        
        # Submit all tasks
        futures = {}
        for task in tasks:
            future = self._executor.submit(self._execute_task, task)
            futures[future] = task.id
        
        # Collect results
        results = {}
        for future in as_completed(futures, timeout=self.config.timeout):
            task_id = futures[future]
            try:
                result = future.result()
                results[task_id] = result
            except Exception as e:
                results[task_id] = ProcessingResult(
                    task_id=task_id,
                    success=False,
                    error=e
                )
        
        # Return results in original order
        ordered_results = []
        for task in tasks:
            ordered_results.append(results.get(task.id, ProcessingResult(
                task_id=task.id,
                success=False,
                error=Exception("Task not found in results")
            )))
        
        return ordered_results
    
    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute a single task."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            result = task.function(*task.args, **task.kwargs)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.id,
                success=True,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task {task.id} failed: {e}")
            
            return ProcessingResult(
                task_id=task.id,
                success=False,
                error=e,
                processing_time=processing_time,
                worker_id=worker_id
            )
    
    def _task_completed(self, task_id: str, future):
        """Handle task completion."""
        with self._lock:
            self._active_tasks.discard(task_id)
            self._completed_tasks += 1
            
            if self.config.enable_progress_tracking:
                progress = (self._completed_tasks / self._total_tasks) * 100
                logger.debug(f"Task {task_id} completed. Progress: {progress:.1f}%")
    
    def get_progress(self) -> Dict[str, Union[int, float]]:
        """Get current processing progress."""
        with self._lock:
            return {
                'total_tasks': self._total_tasks,
                'completed_tasks': self._completed_tasks,
                'active_tasks': len(self._active_tasks),
                'progress_percent': (self._completed_tasks / max(1, self._total_tasks)) * 100
            }
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all active tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if not self._active_tasks:
                    return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
    
    def cancel_all_tasks(self):
        """Cancel all pending tasks."""
        if self._executor:
            # Note: This doesn't cancel running tasks, only pending ones
            with self._lock:
                self._active_tasks.clear()
                logger.info("Cancelled all pending tasks")


class BatchProcessor:
    """
    Utility class for processing large datasets in batches with parallel execution.
    """
    
    def __init__(self, 
                 processor: ParallelProcessor,
                 batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            processor: ParallelProcessor instance
            batch_size: Number of items per batch
        """
        self.processor = processor
        self.batch_size = batch_size
    
    def process_items(self, 
                     items: List[Any], 
                     function: Callable,
                     *args,
                     **kwargs) -> List[ProcessingResult]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            function: Function to apply to each item
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            List of ProcessingResult objects
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self.processor.map_function(function, batch, *args, **kwargs)
            all_results.extend(batch_results)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(items) + self.batch_size - 1)//self.batch_size}")
        
        return all_results


def create_parallel_processor(mode: str = "thread", 
                            max_workers: Optional[int] = None) -> ParallelProcessor:
    """
    Factory function to create a parallel processor.
    
    Args:
        mode: Processing mode ("thread", "process", or "hybrid")
        max_workers: Maximum number of workers
        
    Returns:
        Configured ParallelProcessor instance
    """
    config = ParallelConfig(
        mode=ProcessingMode(mode),
        max_workers=max_workers
    )
    return ParallelProcessor(config)