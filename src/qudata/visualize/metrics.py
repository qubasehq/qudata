"""
Real-time metrics collection for pipeline monitoring.

Collects and stores performance metrics, quality scores, and processing statistics
for visualization and alerting purposes.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float, str]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics."""
    documents_processed: int = 0
    documents_failed: int = 0
    processing_time_total: float = 0.0
    processing_time_avg: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_size: int = 0
    throughput_docs_per_sec: float = 0.0


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    avg_quality_score: float = 0.0
    min_quality_score: float = 0.0
    max_quality_score: float = 0.0
    documents_below_threshold: int = 0
    duplicate_documents: int = 0
    language_distribution: Dict[str, int] = field(default_factory=dict)
    content_length_avg: float = 0.0


@dataclass
class ErrorMetrics:
    """Error and failure metrics."""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_stage: Dict[str, int] = field(default_factory=dict)
    error_rate_percent: float = 0.0
    last_error_time: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and manages real-time metrics for pipeline monitoring.
    
    Provides thread-safe metric collection, aggregation, and retrieval
    for dashboard visualization and alerting.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history_size: Maximum number of historical data points to keep
        """
        self.max_history_size = max_history_size
        self._metrics_history: deque = deque(maxlen=max_history_size)
        self._current_metrics: Dict[str, Any] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._start_time = datetime.now()
        
        # Initialize metric categories
        self.processing_metrics = ProcessingMetrics()
        self.quality_metrics = QualityMetrics()
        self.error_metrics = ErrorMetrics()
    
    def record_metric(self, name: str, value: Union[int, float, str], 
                     tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """
        Record a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            unit: Optional unit of measurement
        """
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit=unit
            )
            self._metrics_history.append(metric)
            self._current_metrics[name] = metric
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self.record_metric(name, self._counters[name], tags, "count")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            self._gauges[name] = value
            self.record_metric(name, value, tags, "gauge")
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement."""
        with self._lock:
            self._timers[name].append(duration)
            # Keep only recent timings
            if len(self._timers[name]) > 100:
                self._timers[name] = self._timers[name][-100:]
            
            avg_duration = sum(self._timers[name]) / len(self._timers[name])
            self.record_metric(f"{name}_avg", avg_duration, tags, "seconds")
            self.record_metric(f"{name}_latest", duration, tags, "seconds")
    
    def record_document_processed(self, processing_time: float, quality_score: float, 
                                language: str, content_length: int, success: bool = True):
        """
        Record metrics for a processed document.
        
        Args:
            processing_time: Time taken to process document
            quality_score: Quality score of processed document
            language: Detected language
            content_length: Length of content in characters
            success: Whether processing was successful
        """
        with self._lock:
            if success:
                self.processing_metrics.documents_processed += 1
                self.processing_metrics.processing_time_total += processing_time
                self.processing_metrics.processing_time_avg = (
                    self.processing_metrics.processing_time_total / 
                    self.processing_metrics.documents_processed
                )
                
                # Update quality metrics
                if self.processing_metrics.documents_processed == 1:
                    self.quality_metrics.min_quality_score = quality_score
                    self.quality_metrics.max_quality_score = quality_score
                    self.quality_metrics.avg_quality_score = quality_score
                else:
                    self.quality_metrics.min_quality_score = min(
                        self.quality_metrics.min_quality_score, quality_score
                    )
                    self.quality_metrics.max_quality_score = max(
                        self.quality_metrics.max_quality_score, quality_score
                    )
                    
                    # Update running average
                    total_docs = self.processing_metrics.documents_processed
                    current_avg = self.quality_metrics.avg_quality_score
                    self.quality_metrics.avg_quality_score = (
                        (current_avg * (total_docs - 1) + quality_score) / total_docs
                    )
                
                # Update language distribution
                self.quality_metrics.language_distribution[language] = (
                    self.quality_metrics.language_distribution.get(language, 0) + 1
                )
                
                # Update content length average
                if self.processing_metrics.documents_processed == 1:
                    self.quality_metrics.content_length_avg = content_length
                else:
                    total_docs = self.processing_metrics.documents_processed
                    current_avg = self.quality_metrics.content_length_avg
                    self.quality_metrics.content_length_avg = (
                        (current_avg * (total_docs - 1) + content_length) / total_docs
                    )
                
            else:
                self.processing_metrics.documents_failed += 1
            
            # Calculate throughput
            elapsed_time = (datetime.now() - self._start_time).total_seconds()
            if elapsed_time > 0:
                self.processing_metrics.throughput_docs_per_sec = (
                    self.processing_metrics.documents_processed / elapsed_time
                )
            
            # Record individual metrics
            self.record_timer("document_processing_time", processing_time)
            self.record_metric("quality_score", quality_score)
            self.increment_counter("documents_total")
            if not success:
                self.increment_counter("documents_failed")
    
    def record_error(self, error_type: str, stage: str, message: str):
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error (e.g., 'FileNotFound', 'ParseError')
            stage: Processing stage where error occurred
            message: Error message
        """
        with self._lock:
            self.error_metrics.total_errors += 1
            self.error_metrics.errors_by_type[error_type] += 1
            self.error_metrics.errors_by_stage[stage] += 1
            self.error_metrics.last_error_time = datetime.now()
            
            # Calculate error rate
            total_processed = (
                self.processing_metrics.documents_processed + 
                self.processing_metrics.documents_failed
            )
            if total_processed > 0:
                self.error_metrics.error_rate_percent = (
                    self.error_metrics.total_errors / total_processed * 100
                )
            
            self.increment_counter("errors_total")
            self.increment_counter(f"errors_{error_type}")
            self.increment_counter(f"errors_stage_{stage}")
    
    def update_system_metrics(self, memory_mb: float, cpu_percent: float, queue_size: int):
        """
        Update system resource metrics.
        
        Args:
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
            queue_size: Current processing queue size
        """
        with self._lock:
            self.processing_metrics.memory_usage_mb = memory_mb
            self.processing_metrics.cpu_usage_percent = cpu_percent
            self.processing_metrics.queue_size = queue_size
            
            self.set_gauge("memory_usage_mb", memory_mb)
            self.set_gauge("cpu_usage_percent", cpu_percent)
            self.set_gauge("queue_size", queue_size)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self._lock:
            return {
                'processing': self.processing_metrics,
                'quality': self.quality_metrics,
                'errors': self.error_metrics,
                'timestamp': datetime.now(),
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds()
            }
    
    def get_metrics_history(self, metric_name: Optional[str] = None, 
                          since: Optional[datetime] = None) -> List[Metric]:
        """
        Get historical metrics.
        
        Args:
            metric_name: Filter by specific metric name
            since: Only return metrics since this timestamp
            
        Returns:
            List of historical metrics
        """
        with self._lock:
            metrics = list(self._metrics_history)
            
            if metric_name:
                metrics = [m for m in metrics if m.name == metric_name]
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_metric_summary(self, metric_name: str, 
                          window_minutes: int = 60) -> Dict[str, float]:
        """
        Get statistical summary of a metric over a time window.
        
        Args:
            metric_name: Name of metric to summarize
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with min, max, avg, count statistics
        """
        since = datetime.now() - timedelta(minutes=window_minutes)
        metrics = self.get_metrics_history(metric_name, since)
        
        if not metrics:
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
        
        values = [float(m.value) for m in metrics if isinstance(m.value, (int, float))]
        
        if not values:
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Formatted metrics data
        """
        current_metrics = self.get_current_metrics()
        
        if format == 'json':
            return json.dumps(current_metrics, default=str, indent=2)
        elif format == 'csv':
            # Simple CSV export of current metrics
            lines = ['metric,value,timestamp']
            timestamp = current_metrics['timestamp']
            
            for category, data in current_metrics.items():
                if category == 'timestamp':
                    continue
                if hasattr(data, '__dict__'):
                    for key, value in data.__dict__.items():
                        lines.append(f"{category}_{key},{value},{timestamp}")
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self._metrics_history.clear()
            self._current_metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            
            self.processing_metrics = ProcessingMetrics()
            self.quality_metrics = QualityMetrics()
            self.error_metrics = ErrorMetrics()
            self._start_time = datetime.now()
            
        logger.info("Metrics reset to initial state")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector):
    """Set the global metrics collector instance."""
    global _global_collector
    _global_collector = collector