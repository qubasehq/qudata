"""
Memory management for efficient resource usage.

Provides memory monitoring, optimization, and resource management
capabilities for processing large datasets.
"""

import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import sys
import os

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float
    pressure_level: MemoryPressure


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_percent: float = 80.0  # Maximum memory usage percentage
    warning_threshold: float = 70.0   # Warning threshold percentage
    critical_threshold: float = 90.0  # Critical threshold percentage
    gc_threshold: float = 75.0        # Garbage collection threshold
    monitoring_interval: float = 5.0  # Memory monitoring interval in seconds
    enable_auto_gc: bool = True       # Enable automatic garbage collection
    enable_monitoring: bool = True    # Enable memory monitoring
    chunk_size_factor: float = 0.1    # Factor for calculating optimal chunk sizes


class MemoryManager:
    """
    Memory manager for efficient resource usage and monitoring.
    
    Provides memory monitoring, automatic garbage collection, and
    resource optimization for large dataset processing.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory manager."""
        self.config = config or MemoryConfig()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._callbacks = []
        self._object_registry = weakref.WeakSet()
        self._lock = threading.Lock()
        self._stats_history = []
        self._max_history = 100
        
        # Get process handle
        self._process = psutil.Process()
        
        if self.config.enable_monitoring:
            self.start_monitoring()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_thread is not None:
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread."""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=1.0)
        self._monitoring_thread = None
        logger.info("Memory monitoring stopped")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory = self._process.memory_info()
        
        # Calculate pressure level
        pressure_level = self._calculate_pressure_level(system_memory.percent)
        
        return MemoryStats(
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            used_memory=system_memory.used,
            memory_percent=system_memory.percent,
            process_memory=process_memory.rss,
            process_memory_percent=(process_memory.rss / system_memory.total) * 100,
            pressure_level=pressure_level
        )
    
    def _calculate_pressure_level(self, memory_percent: float) -> MemoryPressure:
        """Calculate memory pressure level."""
        if memory_percent >= self.config.critical_threshold:
            return MemoryPressure.CRITICAL
        elif memory_percent >= self.config.max_memory_percent:
            return MemoryPressure.HIGH
        elif memory_percent >= self.config.warning_threshold:
            return MemoryPressure.MEDIUM
        else:
            return MemoryPressure.LOW
    
    def _monitor_memory(self):
        """Memory monitoring loop."""
        while not self._stop_monitoring.wait(self.config.monitoring_interval):
            try:
                stats = self.get_memory_stats()
                
                # Store stats history
                with self._lock:
                    self._stats_history.append(stats)
                    if len(self._stats_history) > self._max_history:
                        self._stats_history.pop(0)
                
                # Handle memory pressure
                self._handle_memory_pressure(stats)
                
                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Memory callback error: {e}")
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure situations."""
        if stats.pressure_level == MemoryPressure.CRITICAL:
            logger.warning(f"Critical memory pressure: {stats.memory_percent:.1f}%")
            if self.config.enable_auto_gc:
                self.force_garbage_collection()
        
        elif stats.pressure_level == MemoryPressure.HIGH:
            logger.warning(f"High memory pressure: {stats.memory_percent:.1f}%")
            if self.config.enable_auto_gc:
                self.force_garbage_collection()
        
        elif stats.memory_percent >= self.config.gc_threshold and self.config.enable_auto_gc:
            logger.debug(f"Memory threshold reached: {stats.memory_percent:.1f}%, triggering GC")
            self.force_garbage_collection()
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collection stats."""
        logger.debug("Forcing garbage collection")
        
        # Get initial memory
        initial_memory = self._process.memory_info().rss
        
        # Force garbage collection
        collected = {}
        for generation in range(3):
            collected[f"gen_{generation}"] = gc.collect(generation)
        
        # Get final memory
        final_memory = self._process.memory_info().rss
        memory_freed = initial_memory - final_memory
        
        logger.debug(f"GC completed: freed {memory_freed / 1024 / 1024:.1f} MB")
        
        return {
            **collected,
            'memory_freed_bytes': memory_freed,
            'memory_freed_mb': memory_freed / 1024 / 1024
        }
    
    def register_callback(self, callback: Callable[[MemoryStats], None]):
        """Register a memory monitoring callback."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[MemoryStats], None]):
        """Unregister a memory monitoring callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def register_object(self, obj: Any):
        """Register an object for memory tracking."""
        self._object_registry.add(obj)
    
    def get_optimal_chunk_size(self, 
                              item_size_bytes: int, 
                              total_items: int,
                              max_memory_mb: Optional[int] = None) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            item_size_bytes: Average size of each item in bytes
            total_items: Total number of items to process
            max_memory_mb: Maximum memory to use in MB (optional)
            
        Returns:
            Optimal chunk size
        """
        stats = self.get_memory_stats()
        
        if max_memory_mb is None:
            # Use a fraction of available memory
            available_bytes = stats.available_memory * self.config.chunk_size_factor
        else:
            available_bytes = max_memory_mb * 1024 * 1024
        
        # Calculate chunk size
        chunk_size = max(1, int(available_bytes / item_size_bytes))
        
        # Don't exceed total items
        chunk_size = min(chunk_size, total_items)
        
        logger.debug(f"Calculated optimal chunk size: {chunk_size} items")
        return chunk_size
    
    def get_optimal_batch_size(self, 
                              estimated_item_memory: int,
                              safety_factor: float = 0.8) -> int:
        """
        Get optimal batch size for processing based on available memory.
        
        Args:
            estimated_item_memory: Estimated memory per item in bytes
            safety_factor: Safety factor to apply (0.0-1.0)
            
        Returns:
            Optimal batch size
        """
        stats = self.get_memory_stats()
        available_memory = stats.available_memory * safety_factor
        
        batch_size = max(1, int(available_memory / estimated_item_memory))
        
        logger.debug(f"Calculated optimal batch size: {batch_size} items")
        return batch_size
    
    def check_memory_availability(self, required_memory_mb: float) -> bool:
        """
        Check if required memory is available.
        
        Args:
            required_memory_mb: Required memory in MB
            
        Returns:
            True if memory is available, False otherwise
        """
        stats = self.get_memory_stats()
        required_bytes = required_memory_mb * 1024 * 1024
        
        return stats.available_memory >= required_bytes
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations."""
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats.pressure_level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            recommendations.append("Consider reducing batch size or chunk size")
            recommendations.append("Enable streaming processing for large files")
            recommendations.append("Force garbage collection")
        
        if stats.memory_percent > 85:
            recommendations.append("Close unnecessary applications")
            recommendations.append("Consider using a machine with more RAM")
        
        return {
            'current_stats': stats,
            'recommendations': recommendations,
            'optimal_chunk_size_1mb': self.get_optimal_chunk_size(1024*1024, 1000),
            'optimal_batch_size_10kb': self.get_optimal_batch_size(10*1024)
        }
    
    def get_stats_history(self) -> List[MemoryStats]:
        """Get memory statistics history."""
        with self._lock:
            return self._stats_history.copy()
    
    def clear_stats_history(self):
        """Clear memory statistics history."""
        with self._lock:
            self._stats_history.clear()
    
    def optimize_for_large_dataset(self) -> Dict[str, Any]:
        """
        Optimize memory settings for large dataset processing.
        
        Returns:
            Dictionary with optimization settings
        """
        stats = self.get_memory_stats()
        
        # Force garbage collection
        gc_stats = self.force_garbage_collection()
        
        # Calculate optimal settings
        optimal_settings = {
            'chunk_size_1mb_items': self.get_optimal_chunk_size(1024*1024, 10000),
            'batch_size_100kb_items': self.get_optimal_batch_size(100*1024),
            'recommended_max_workers': max(1, min(8, int(stats.available_memory / (512 * 1024 * 1024)))),
            'gc_stats': gc_stats,
            'memory_stats': stats
        }
        
        logger.info(f"Memory optimization complete: {optimal_settings}")
        return optimal_settings


class MemoryPool:
    """
    Memory pool for reusing objects and reducing allocation overhead.
    """
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            factory: Function to create new objects
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self._pool = []
        self._lock = threading.Lock()
    
    def get(self) -> Any:
        """Get an object from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                return self.factory()
    
    def put(self, obj: Any):
        """Return an object to the pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()
    
    def size(self) -> int:
        """Get current pool size."""
        with self._lock:
            return len(self._pool)


def create_memory_manager(max_memory_percent: float = 80.0,
                         enable_monitoring: bool = True) -> MemoryManager:
    """
    Factory function to create a memory manager.
    
    Args:
        max_memory_percent: Maximum memory usage percentage
        enable_monitoring: Enable memory monitoring
        
    Returns:
        Configured MemoryManager instance
    """
    config = MemoryConfig(
        max_memory_percent=max_memory_percent,
        enable_monitoring=enable_monitoring
    )
    return MemoryManager(config)