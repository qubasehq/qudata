"""
Performance profiler for resource monitoring and optimization.

This module provides comprehensive performance profiling capabilities to monitor
resource usage, identify bottlenecks, and optimize processing pipeline performance.
"""

import gc
import os
import psutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class ProfilerMode(Enum):
    """Profiling modes."""
    CONTINUOUS = "continuous"
    SAMPLING = "sampling"
    EVENT_BASED = "event_based"


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None
    thread_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "process_cpu_percent": self.process_cpu_percent,
            "process_memory_mb": self.process_memory_mb,
            "thread_count": self.thread_count
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics over a time period."""
    duration_seconds: float
    cpu_avg: float
    cpu_max: float
    memory_avg_mb: float
    memory_max_mb: float
    memory_peak_percent: float
    disk_read_total_mb: float
    disk_write_total_mb: float
    network_sent_total_mb: float
    network_recv_total_mb: float
    gpu_avg: Optional[float] = None
    gpu_max: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    process_cpu_avg: Optional[float] = None
    process_memory_avg_mb: Optional[float] = None
    gc_collections: int = 0
    gc_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "cpu_avg": self.cpu_avg,
            "cpu_max": self.cpu_max,
            "memory_avg_mb": self.memory_avg_mb,
            "memory_max_mb": self.memory_max_mb,
            "memory_peak_percent": self.memory_peak_percent,
            "disk_read_total_mb": self.disk_read_total_mb,
            "disk_write_total_mb": self.disk_write_total_mb,
            "network_sent_total_mb": self.network_sent_total_mb,
            "network_recv_total_mb": self.network_recv_total_mb,
            "gpu_avg": self.gpu_avg,
            "gpu_max": self.gpu_max,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "process_cpu_avg": self.process_cpu_avg,
            "process_memory_avg_mb": self.process_memory_avg_mb,
            "gc_collections": self.gc_collections,
            "gc_time_seconds": self.gc_time_seconds
        }


@dataclass
class ProfilingResult:
    """Result of a profiling session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    function_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Add a resource snapshot."""
        self.snapshots.append(snapshot)
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate aggregated metrics from snapshots."""
        if not self.snapshots:
            return PerformanceMetrics(
                duration_seconds=0,
                cpu_avg=0, cpu_max=0,
                memory_avg_mb=0, memory_max_mb=0, memory_peak_percent=0,
                disk_read_total_mb=0, disk_write_total_mb=0,
                network_sent_total_mb=0, network_recv_total_mb=0
            )
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Extract values
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_used_mb for s in self.snapshots]
        memory_percent_values = [s.memory_percent for s in self.snapshots]
        
        # Calculate disk and network totals (approximate)
        disk_read_total = max(s.disk_read_mb for s in self.snapshots) if self.snapshots else 0
        disk_write_total = max(s.disk_write_mb for s in self.snapshots) if self.snapshots else 0
        network_sent_total = max(s.network_sent_mb for s in self.snapshots) if self.snapshots else 0
        network_recv_total = max(s.network_recv_mb for s in self.snapshots) if self.snapshots else 0
        
        # GPU metrics (if available)
        gpu_values = [s.gpu_percent for s in self.snapshots if s.gpu_percent is not None]
        gpu_memory_values = [s.gpu_memory_used_mb for s in self.snapshots if s.gpu_memory_used_mb is not None]
        
        # Process-specific metrics
        process_cpu_values = [s.process_cpu_percent for s in self.snapshots if s.process_cpu_percent is not None]
        process_memory_values = [s.process_memory_mb for s in self.snapshots if s.process_memory_mb is not None]
        
        self.metrics = PerformanceMetrics(
            duration_seconds=duration,
            cpu_avg=np.mean(cpu_values),
            cpu_max=max(cpu_values),
            memory_avg_mb=np.mean(memory_values),
            memory_max_mb=max(memory_values),
            memory_peak_percent=max(memory_percent_values),
            disk_read_total_mb=disk_read_total,
            disk_write_total_mb=disk_write_total,
            network_sent_total_mb=network_sent_total,
            network_recv_total_mb=network_recv_total,
            gpu_avg=np.mean(gpu_values) if gpu_values else None,
            gpu_max=max(gpu_values) if gpu_values else None,
            gpu_memory_peak_mb=max(gpu_memory_values) if gpu_memory_values else None,
            process_cpu_avg=np.mean(process_cpu_values) if process_cpu_values else None,
            process_memory_avg_mb=np.mean(process_memory_values) if process_memory_values else None
        )
        
        return self.metrics
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks from profiling data."""
        bottlenecks = []
        
        if not self.metrics:
            self.calculate_metrics()
        
        # CPU bottlenecks
        if self.metrics.cpu_max > 90:
            bottlenecks.append({
                "type": "cpu",
                "severity": "high",
                "description": f"High CPU usage detected (max: {self.metrics.cpu_max:.1f}%)",
                "recommendation": "Consider optimizing CPU-intensive operations or using parallel processing"
            })
        elif self.metrics.cpu_avg > 70:
            bottlenecks.append({
                "type": "cpu",
                "severity": "medium",
                "description": f"Sustained high CPU usage (avg: {self.metrics.cpu_avg:.1f}%)",
                "recommendation": "Monitor CPU usage and consider optimization"
            })
        
        # Memory bottlenecks
        if self.metrics.memory_peak_percent > 90:
            bottlenecks.append({
                "type": "memory",
                "severity": "high",
                "description": f"High memory usage detected (peak: {self.metrics.memory_peak_percent:.1f}%)",
                "recommendation": "Optimize memory usage or increase available memory"
            })
        elif self.metrics.memory_avg_mb > 1000:  # 1GB
            bottlenecks.append({
                "type": "memory",
                "severity": "medium",
                "description": f"High average memory usage ({self.metrics.memory_avg_mb:.0f} MB)",
                "recommendation": "Consider streaming processing for large datasets"
            })
        
        # Disk I/O bottlenecks
        total_disk_io = self.metrics.disk_read_total_mb + self.metrics.disk_write_total_mb
        if total_disk_io > 1000:  # 1GB
            bottlenecks.append({
                "type": "disk",
                "severity": "medium",
                "description": f"High disk I/O detected ({total_disk_io:.0f} MB)",
                "recommendation": "Consider using faster storage or optimizing I/O operations"
            })
        
        # GPU bottlenecks (if GPU monitoring is available)
        if self.metrics.gpu_max and self.metrics.gpu_max > 95:
            bottlenecks.append({
                "type": "gpu",
                "severity": "high",
                "description": f"High GPU usage detected (max: {self.metrics.gpu_max:.1f}%)",
                "recommendation": "GPU is fully utilized - consider batch size optimization"
            })
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        if not self.metrics:
            self.calculate_metrics()
        
        # Memory recommendations
        if self.metrics.memory_peak_percent > 80:
            recommendations.append("Consider implementing streaming processing to reduce memory usage")
            recommendations.append("Use memory-mapped files for large datasets")
            recommendations.append("Implement garbage collection optimization")
        
        # CPU recommendations
        if self.metrics.cpu_avg > 60:
            recommendations.append("Consider using multiprocessing for CPU-intensive tasks")
            recommendations.append("Profile individual functions to identify optimization opportunities")
            recommendations.append("Use vectorized operations where possible")
        
        # I/O recommendations
        total_io = self.metrics.disk_read_total_mb + self.metrics.disk_write_total_mb
        if total_io > 500:
            recommendations.append("Consider using SSD storage for better I/O performance")
            recommendations.append("Implement caching for frequently accessed data")
            recommendations.append("Use batch processing to reduce I/O overhead")
        
        # General recommendations
        if self.metrics.duration_seconds > 300:  # 5 minutes
            recommendations.append("Consider breaking down long-running operations into smaller chunks")
            recommendations.append("Implement progress tracking and checkpointing")
        
        self.recommendations = recommendations
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profiling result to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "function_profiles": self.function_profiles
        }


class ResourceMonitor:
    """Monitor system resources continuously."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Interval between resource snapshots in seconds
        """
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.initial_disk_io = psutil.disk_io_counters()
        self.initial_network_io = psutil.net_io_counters()
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False
    
    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        # System-wide metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.initial_disk_io.read_bytes) / 1024 / 1024
        disk_write_mb = (current_disk_io.write_bytes - self.initial_disk_io.write_bytes) / 1024 / 1024
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        network_sent_mb = (current_network_io.bytes_sent - self.initial_network_io.bytes_sent) / 1024 / 1024
        network_recv_mb = (current_network_io.bytes_recv - self.initial_network_io.bytes_recv) / 1024 / 1024
        
        # Process-specific metrics
        process_cpu = self.process.cpu_percent()
        process_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        thread_count = self.process.num_threads()
        
        # GPU metrics (if available)
        gpu_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        
        if self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_memory_total_mb = gpu.memoryTotal
            except Exception:
                pass  # GPU monitoring failed, continue without it
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_percent=gpu_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory,
            thread_count=thread_count
        )


class PerformanceProfiler:
    """Main performance profiler for monitoring and analyzing resource usage."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance profiler.
        
        Args:
            config: Configuration for profiling behavior
        """
        self.config = config or {}
        self.sampling_interval = self.config.get("sampling_interval", 1.0)
        self.mode = ProfilerMode(self.config.get("mode", "sampling"))
        self.monitor = ResourceMonitor(self.sampling_interval)
        self.active_sessions: Dict[str, ProfilingResult] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
    
    def start_profiling(self, session_id: str = None) -> str:
        """
        Start a profiling session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID for the started profiling session
        """
        if session_id is None:
            session_id = f"profile_{int(time.time())}"
        
        if session_id in self.active_sessions:
            raise ValueError(f"Profiling session '{session_id}' already active")
        
        # Create profiling result
        result = ProfilingResult(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=datetime.now()  # Will be updated when stopped
        )
        
        self.active_sessions[session_id] = result
        
        # Start monitoring thread if not already running
        if self.mode == ProfilerMode.CONTINUOUS and not self.monitoring_thread:
            self._start_monitoring_thread()
        
        # Take initial snapshot
        initial_snapshot = self.monitor.take_snapshot()
        result.add_snapshot(initial_snapshot)
        
        return session_id
    
    def stop_profiling(self, session_id: str) -> ProfilingResult:
        """
        Stop a profiling session and return results.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ProfilingResult with collected data and analysis
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"No active profiling session '{session_id}'")
        
        result = self.active_sessions[session_id]
        result.end_time = datetime.now()
        
        # Take final snapshot
        final_snapshot = self.monitor.take_snapshot()
        result.add_snapshot(final_snapshot)
        
        # Calculate metrics and detect bottlenecks
        result.calculate_metrics()
        result.detect_bottlenecks()
        result.generate_recommendations()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Stop monitoring thread if no active sessions
        if not self.active_sessions and self.monitoring_thread:
            self._stop_monitoring_thread()
        
        return result
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfilingResult]:
        """
        Profile a function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function_result, profiling_result)
        """
        session_id = f"func_{func.__name__}_{int(time.time())}"
        
        # Start profiling
        self.start_profiling(session_id)
        
        try:
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Add function-specific metrics
            profiling_result = self.active_sessions[session_id]
            profiling_result.function_profiles[func.__name__] = {
                "execution_time": execution_time,
                "success": True,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
            
            return result, self.stop_profiling(session_id)
            
        except Exception as e:
            # Record function failure
            profiling_result = self.active_sessions[session_id]
            profiling_result.function_profiles[func.__name__] = {
                "execution_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
            
            # Stop profiling and re-raise exception
            self.stop_profiling(session_id)
            raise
    
    def profile_context(self, session_id: str = None):
        """
        Context manager for profiling code blocks.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Context manager that profiles the enclosed code
        """
        return ProfilingContext(self, session_id)
    
    def take_snapshot(self, session_id: str) -> ResourceSnapshot:
        """
        Take a manual resource snapshot for an active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ResourceSnapshot of current resource usage
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"No active profiling session '{session_id}'")
        
        snapshot = self.monitor.take_snapshot()
        self.active_sessions[session_id].add_snapshot(snapshot)
        
        return snapshot
    
    def _start_monitoring_thread(self) -> None:
        """Start continuous monitoring thread."""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _stop_monitoring_thread(self) -> None:
        """Stop continuous monitoring thread."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Take snapshots for all active sessions
                for session_id, result in self.active_sessions.items():
                    snapshot = self.monitor.take_snapshot()
                    result.add_snapshot(snapshot)
                
                # Wait for next sampling interval
                self.stop_monitoring.wait(self.sampling_interval)
                
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for profiling context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_usage": {path: psutil.disk_usage(path)._asdict() 
                          for path in ["/", "C:\\"] if os.path.exists(path)},
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "platform": os.name,
            "gpu_available": self.monitor.gpu_available
        }
    
    def compare_profiles(self, result1: ProfilingResult, result2: ProfilingResult) -> Dict[str, Any]:
        """
        Compare two profiling results.
        
        Args:
            result1: First profiling result
            result2: Second profiling result
            
        Returns:
            Comparison analysis
        """
        if not result1.metrics or not result2.metrics:
            raise ValueError("Both profiling results must have calculated metrics")
        
        comparison = {
            "duration_change": result2.metrics.duration_seconds - result1.metrics.duration_seconds,
            "cpu_avg_change": result2.metrics.cpu_avg - result1.metrics.cpu_avg,
            "memory_avg_change": result2.metrics.memory_avg_mb - result1.metrics.memory_avg_mb,
            "memory_peak_change": result2.metrics.memory_max_mb - result1.metrics.memory_max_mb,
            "disk_io_change": (result2.metrics.disk_read_total_mb + result2.metrics.disk_write_total_mb) - 
                             (result1.metrics.disk_read_total_mb + result1.metrics.disk_write_total_mb),
            "performance_summary": "improved" if result2.metrics.duration_seconds < result1.metrics.duration_seconds else "degraded"
        }
        
        return comparison


class ProfilingContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, session_id: str = None):
        """
        Initialize profiling context.
        
        Args:
            profiler: PerformanceProfiler instance
            session_id: Optional session identifier
        """
        self.profiler = profiler
        self.session_id = session_id
        self.result: Optional[ProfilingResult] = None
    
    def __enter__(self) -> 'ProfilingContext':
        """Start profiling when entering context."""
        self.session_id = self.profiler.start_profiling(self.session_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop profiling when exiting context."""
        self.result = self.profiler.stop_profiling(self.session_id)
    
    def get_result(self) -> Optional[ProfilingResult]:
        """Get profiling result."""
        return self.result