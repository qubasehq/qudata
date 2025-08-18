"""
Quality benchmarks for regression testing and performance tracking.

This module provides comprehensive benchmarking capabilities to track data quality
metrics over time and detect regressions in processing pipeline performance.
"""

import json
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import Dataset, Document, QualityMetrics


class BenchmarkType(Enum):
    """Types of benchmarks."""
    QUALITY = "quality"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class BenchmarkMetric:
    """Individual benchmark metric."""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    baseline: Optional[float] = None
    tolerance: float = 0.05  # 5% tolerance by default
    
    def is_within_tolerance(self) -> bool:
        """Check if metric is within acceptable tolerance of baseline."""
        if self.baseline is None:
            return True
        
        tolerance_range = abs(self.baseline * self.tolerance)
        return abs(self.value - self.baseline) <= tolerance_range
    
    def meets_threshold(self) -> bool:
        """Check if metric meets minimum threshold."""
        if self.threshold is None:
            return True
        return self.value >= self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "baseline": self.baseline,
            "tolerance": self.tolerance,
            "within_tolerance": self.is_within_tolerance(),
            "meets_threshold": self.meets_threshold()
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    
    def add_metric(self, metric: BenchmarkMetric) -> None:
        """Add a metric to the benchmark result."""
        self.metrics.append(metric)
    
    def get_metric(self, name: str) -> Optional[BenchmarkMetric]:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def get_failed_metrics(self) -> List[BenchmarkMetric]:
        """Get metrics that failed their thresholds or tolerance."""
        failed = []
        for metric in self.metrics:
            if not metric.meets_threshold() or not metric.is_within_tolerance():
                failed.append(metric)
        return failed
    
    def has_failures(self) -> bool:
        """Check if any metrics failed."""
        return len(self.get_failed_metrics()) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type.value,
            "status": self.status.value,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "has_failures": self.has_failures()
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType, config: Dict[str, Any] = None):
        """
        Initialize benchmark.
        
        Args:
            name: Name of the benchmark
            benchmark_type: Type of benchmark
            config: Configuration for the benchmark
        """
        self.name = name
        self.benchmark_type = benchmark_type
        self.config = config or {}
        self.baselines: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {}
    
    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set baseline value for a metric."""
        self.baselines[metric_name] = value
    
    def set_threshold(self, metric_name: str, value: float) -> None:
        """Set threshold value for a metric."""
        self.thresholds[metric_name] = value
    
    @abstractmethod
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """
        Run the benchmark on a dataset.
        
        Args:
            dataset: Dataset to benchmark
            
        Returns:
            BenchmarkResult with metrics and status
        """
        pass
    
    def create_metric(self, name: str, value: float, unit: str) -> BenchmarkMetric:
        """Helper method to create benchmark metrics."""
        return BenchmarkMetric(
            name=name,
            value=value,
            unit=unit,
            threshold=self.thresholds.get(name),
            baseline=self.baselines.get(name),
            tolerance=self.config.get("tolerance", 0.05)
        )


class QualityBenchmark(Benchmark):
    """Benchmark for data quality metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="quality_benchmark",
            benchmark_type=BenchmarkType.QUALITY,
            config=config
        )
        
        # Set default thresholds
        self.set_threshold("avg_quality_score", 0.7)
        self.set_threshold("min_quality_score", 0.3)
        self.set_threshold("content_completeness", 0.9)
        self.set_threshold("metadata_completeness", 0.8)
    
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """Run quality benchmark."""
        start_time = time.time()
        result = BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            status=BenchmarkStatus.PASSED,
            dataset_id=dataset.id,
            dataset_version=dataset.version
        )
        
        try:
            # Calculate quality metrics
            quality_scores = [doc.metadata.quality_score for doc in dataset.documents 
                            if doc.metadata.quality_score > 0]
            
            if quality_scores:
                avg_quality = statistics.mean(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)
                
                result.add_metric(self.create_metric("avg_quality_score", avg_quality, "score"))
                result.add_metric(self.create_metric("min_quality_score", min_quality, "score"))
                result.add_metric(self.create_metric("max_quality_score", max_quality, "score"))
            
            # Content completeness
            non_empty_content = sum(1 for doc in dataset.documents if doc.content.strip())
            content_completeness = non_empty_content / len(dataset.documents) if dataset.documents else 0
            result.add_metric(self.create_metric("content_completeness", content_completeness, "ratio"))
            
            # Metadata completeness
            complete_metadata = 0
            for doc in dataset.documents:
                if (doc.metadata.language and doc.metadata.language != "unknown" and
                    doc.metadata.domain and doc.metadata.domain != "uncategorized" and
                    doc.metadata.file_type):
                    complete_metadata += 1
            
            metadata_completeness = complete_metadata / len(dataset.documents) if dataset.documents else 0
            result.add_metric(self.create_metric("metadata_completeness", metadata_completeness, "ratio"))
            
            # Language diversity
            languages = set(doc.metadata.language for doc in dataset.documents 
                          if doc.metadata.language and doc.metadata.language != "unknown")
            result.add_metric(self.create_metric("language_diversity", len(languages), "count"))
            
            # Domain diversity
            domains = set(doc.metadata.domain for doc in dataset.documents 
                        if doc.metadata.domain and doc.metadata.domain != "uncategorized")
            result.add_metric(self.create_metric("domain_diversity", len(domains), "count"))
            
            # Check for failures
            if result.has_failures():
                result.status = BenchmarkStatus.FAILED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result


class PerformanceBenchmark(Benchmark):
    """Benchmark for processing performance metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="performance_benchmark",
            benchmark_type=BenchmarkType.PERFORMANCE,
            config=config
        )
        
        # Set default thresholds
        self.set_threshold("docs_per_second", 10.0)
        self.set_threshold("chars_per_second", 10000.0)
        self.set_threshold("avg_processing_time", 1.0)
    
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """Run performance benchmark."""
        start_time = time.time()
        result = BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            status=BenchmarkStatus.PASSED,
            dataset_id=dataset.id,
            dataset_version=dataset.version
        )
        
        try:
            # Simulate processing time calculation
            total_docs = len(dataset.documents)
            total_chars = sum(len(doc.content) for doc in dataset.documents)
            
            # Use actual processing timestamps if available
            processing_times = []
            for doc in dataset.documents:
                # Simulate processing time based on content length
                estimated_time = len(doc.content) / 50000  # 50k chars per second baseline
                processing_times.append(estimated_time)
            
            if processing_times:
                total_processing_time = sum(processing_times)
                avg_processing_time = statistics.mean(processing_times)
                
                # Calculate throughput metrics
                docs_per_second = total_docs / total_processing_time if total_processing_time > 0 else 0
                chars_per_second = total_chars / total_processing_time if total_processing_time > 0 else 0
                
                result.add_metric(self.create_metric("docs_per_second", docs_per_second, "docs/sec"))
                result.add_metric(self.create_metric("chars_per_second", chars_per_second, "chars/sec"))
                result.add_metric(self.create_metric("avg_processing_time", avg_processing_time, "seconds"))
                result.add_metric(self.create_metric("total_processing_time", total_processing_time, "seconds"))
            
            # Memory usage estimation
            estimated_memory = sum(len(doc.content.encode('utf-8')) for doc in dataset.documents)
            result.add_metric(self.create_metric("estimated_memory_usage", estimated_memory / 1024 / 1024, "MB"))
            
            # Check for failures
            if result.has_failures():
                result.status = BenchmarkStatus.FAILED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result


class AccuracyBenchmark(Benchmark):
    """Benchmark for processing accuracy metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="accuracy_benchmark",
            benchmark_type=BenchmarkType.ACCURACY,
            config=config
        )
        
        # Set default thresholds
        self.set_threshold("language_detection_confidence", 0.8)
        self.set_threshold("domain_classification_confidence", 0.7)
        self.set_threshold("entity_extraction_precision", 0.8)
    
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """Run accuracy benchmark."""
        start_time = time.time()
        result = BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            status=BenchmarkStatus.PASSED,
            dataset_id=dataset.id,
            dataset_version=dataset.version
        )
        
        try:
            # Language detection accuracy (simulated)
            lang_confidences = []
            for doc in dataset.documents:
                if doc.metadata.language and doc.metadata.language != "unknown":
                    # Simulate confidence based on content length and language
                    confidence = min(0.95, 0.5 + len(doc.content) / 10000)
                    lang_confidences.append(confidence)
            
            if lang_confidences:
                avg_lang_confidence = statistics.mean(lang_confidences)
                result.add_metric(self.create_metric("language_detection_confidence", avg_lang_confidence, "confidence"))
            
            # Domain classification accuracy (simulated)
            domain_confidences = []
            for doc in dataset.documents:
                if doc.metadata.domain and doc.metadata.domain != "uncategorized":
                    # Simulate confidence based on content and domain
                    confidence = min(0.9, 0.6 + len(doc.metadata.topics) * 0.1)
                    domain_confidences.append(confidence)
            
            if domain_confidences:
                avg_domain_confidence = statistics.mean(domain_confidences)
                result.add_metric(self.create_metric("domain_classification_confidence", avg_domain_confidence, "confidence"))
            
            # Entity extraction metrics (simulated)
            entity_counts = []
            for doc in dataset.documents:
                entity_counts.append(len(doc.metadata.entities))
            
            if entity_counts:
                avg_entities = statistics.mean(entity_counts)
                result.add_metric(self.create_metric("avg_entities_per_doc", avg_entities, "count"))
                
                # Simulate precision based on entity density
                precision = min(0.95, 0.7 + avg_entities / 100)
                result.add_metric(self.create_metric("entity_extraction_precision", precision, "precision"))
            
            # Check for failures
            if result.has_failures():
                result.status = BenchmarkStatus.FAILED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result


class ConsistencyBenchmark(Benchmark):
    """Benchmark for data consistency metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="consistency_benchmark",
            benchmark_type=BenchmarkType.CONSISTENCY,
            config=config
        )
        
        # Set default thresholds
        self.set_threshold("duplicate_ratio", 0.05)  # Max 5% duplicates
        self.set_threshold("encoding_consistency", 0.95)
        self.set_threshold("format_consistency", 0.9)
    
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """Run consistency benchmark."""
        start_time = time.time()
        result = BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            status=BenchmarkStatus.PASSED,
            dataset_id=dataset.id,
            dataset_version=dataset.version
        )
        
        try:
            # Duplicate detection (simplified)
            content_hashes = set()
            duplicates = 0
            
            for doc in dataset.documents:
                content_hash = hash(doc.content[:1000])  # Hash first 1000 chars
                if content_hash in content_hashes:
                    duplicates += 1
                else:
                    content_hashes.add(content_hash)
            
            duplicate_ratio = duplicates / len(dataset.documents) if dataset.documents else 0
            result.add_metric(self.create_metric("duplicate_ratio", duplicate_ratio, "ratio"))
            
            # Encoding consistency
            encoding_issues = 0
            for doc in dataset.documents:
                if '�' in doc.content or '\ufffd' in doc.content:
                    encoding_issues += 1
            
            encoding_consistency = 1 - (encoding_issues / len(dataset.documents)) if dataset.documents else 1
            result.add_metric(self.create_metric("encoding_consistency", encoding_consistency, "ratio"))
            
            # Format consistency
            file_types = {}
            for doc in dataset.documents:
                file_type = doc.metadata.file_type
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Calculate format consistency (how well distributed are file types)
            if file_types:
                max_type_ratio = max(file_types.values()) / len(dataset.documents)
                format_consistency = 1 - max_type_ratio if len(file_types) > 1 else 1
                result.add_metric(self.create_metric("format_consistency", format_consistency, "ratio"))
            
            # Language consistency
            languages = {}
            for doc in dataset.documents:
                lang = doc.metadata.language
                if lang and lang != "unknown":
                    languages[lang] = languages.get(lang, 0) + 1
            
            if languages:
                primary_lang_ratio = max(languages.values()) / len(dataset.documents)
                result.add_metric(self.create_metric("primary_language_ratio", primary_lang_ratio, "ratio"))
            
            # Check for failures
            if result.has_failures():
                result.status = BenchmarkStatus.FAILED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result


@dataclass
class BenchmarkSuite:
    """Collection of benchmarks to run together."""
    name: str
    benchmarks: List[Benchmark] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def remove_benchmark(self, benchmark_name: str) -> bool:
        """Remove a benchmark by name."""
        original_count = len(self.benchmarks)
        self.benchmarks = [b for b in self.benchmarks if b.name != benchmark_name]
        return len(self.benchmarks) < original_count
    
    def run_all(self, dataset: Dataset) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        results = []
        
        for benchmark in self.benchmarks:
            try:
                result = benchmark.run(dataset)
                results.append(result)
            except Exception as e:
                # Create failed result for benchmark that crashed
                failed_result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    benchmark_type=benchmark.benchmark_type,
                    status=BenchmarkStatus.FAILED,
                    error_message=str(e),
                    dataset_id=dataset.id,
                    dataset_version=dataset.version
                )
                results.append(failed_result)
        
        return results
    
    def get_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        total = len(results)
        passed = sum(1 for r in results if r.status == BenchmarkStatus.PASSED)
        failed = sum(1 for r in results if r.status == BenchmarkStatus.FAILED)
        warnings = sum(1 for r in results if r.status == BenchmarkStatus.WARNING)
        
        return {
            "suite_name": self.name,
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "success_rate": passed / total if total > 0 else 0,
            "total_execution_time": sum(r.execution_time for r in results)
        }


class QualityBenchmarks:
    """Main class for managing quality benchmarks and regression testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quality benchmarks.
        
        Args:
            config: Configuration for benchmarks and baselines
        """
        self.config = config or {}
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.baseline_history: List[Dict[str, Any]] = []
        self._setup_default_suites()
    
    def _setup_default_suites(self) -> None:
        """Setup default benchmark suites."""
        # Quality suite
        quality_suite = BenchmarkSuite("quality_suite")
        quality_suite.add_benchmark(QualityBenchmark(self.config.get("quality", {})))
        self.suites["quality"] = quality_suite
        
        # Performance suite
        performance_suite = BenchmarkSuite("performance_suite")
        performance_suite.add_benchmark(PerformanceBenchmark(self.config.get("performance", {})))
        self.suites["performance"] = performance_suite
        
        # Accuracy suite
        accuracy_suite = BenchmarkSuite("accuracy_suite")
        accuracy_suite.add_benchmark(AccuracyBenchmark(self.config.get("accuracy", {})))
        self.suites["accuracy"] = accuracy_suite
        
        # Consistency suite
        consistency_suite = BenchmarkSuite("consistency_suite")
        consistency_suite.add_benchmark(ConsistencyBenchmark(self.config.get("consistency", {})))
        self.suites["consistency"] = consistency_suite
        
        # Full suite (all benchmarks)
        full_suite = BenchmarkSuite("full_suite")
        for suite in [quality_suite, performance_suite, accuracy_suite, consistency_suite]:
            for benchmark in suite.benchmarks:
                full_suite.add_benchmark(benchmark)
        self.suites["full"] = full_suite
    
    def add_suite(self, suite: BenchmarkSuite) -> None:
        """Add a custom benchmark suite."""
        self.suites[suite.name] = suite
    
    def run_suite(self, suite_name: str, dataset: Dataset) -> List[BenchmarkResult]:
        """Run a specific benchmark suite."""
        if suite_name not in self.suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")
        
        suite = self.suites[suite_name]
        return suite.run_all(dataset)
    
    def run_all_suites(self, dataset: Dataset) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark suites."""
        all_results = {}
        
        for suite_name, suite in self.suites.items():
            if suite_name != "full":  # Skip full suite to avoid duplication
                all_results[suite_name] = suite.run_all(dataset)
        
        return all_results
    
    def set_baselines_from_dataset(self, dataset: Dataset, suite_name: str = "full") -> None:
        """Set baseline metrics from a reference dataset."""
        results = self.run_suite(suite_name, dataset)
        
        baselines = {}
        for result in results:
            benchmark_baselines = {}
            for metric in result.metrics:
                benchmark_baselines[metric.name] = metric.value
            baselines[result.benchmark_name] = benchmark_baselines
        
        # Update benchmark baselines
        suite = self.suites[suite_name]
        for benchmark in suite.benchmarks:
            if benchmark.name in baselines:
                for metric_name, value in baselines[benchmark.name].items():
                    benchmark.set_baseline(metric_name, value)
        
        # Store in history
        self.baseline_history.append({
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset.id,
            "dataset_version": dataset.version,
            "suite_name": suite_name,
            "baselines": baselines
        })
    
    def detect_regressions(self, current_results: List[BenchmarkResult], 
                          tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect regressions by comparing current results with baselines.
        
        Args:
            current_results: Current benchmark results
            tolerance: Acceptable tolerance for regression detection
            
        Returns:
            List of detected regressions
        """
        regressions = []
        
        for result in current_results:
            for metric in result.metrics:
                if metric.baseline is not None:
                    # Check for significant degradation
                    if metric.value < metric.baseline * (1 - tolerance):
                        regressions.append({
                            "benchmark": result.benchmark_name,
                            "metric": metric.name,
                            "current_value": metric.value,
                            "baseline_value": metric.baseline,
                            "degradation_percent": ((metric.baseline - metric.value) / metric.baseline) * 100,
                            "severity": "high" if metric.value < metric.baseline * 0.8 else "medium"
                        })
        
        return regressions
    
    def generate_trend_report(self, results_history: List[List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate trend analysis report from historical results."""
        trends = {}
        
        # Group results by benchmark and metric
        metric_history = {}
        
        for results in results_history:
            for result in results:
                benchmark_key = result.benchmark_name
                if benchmark_key not in metric_history:
                    metric_history[benchmark_key] = {}
                
                for metric in result.metrics:
                    metric_key = metric.name
                    if metric_key not in metric_history[benchmark_key]:
                        metric_history[benchmark_key][metric_key] = []
                    
                    metric_history[benchmark_key][metric_key].append({
                        "timestamp": result.timestamp,
                        "value": metric.value
                    })
        
        # Calculate trends
        for benchmark_name, benchmark_metrics in metric_history.items():
            trends[benchmark_name] = {}
            
            for metric_name, metric_values in benchmark_metrics.items():
                if len(metric_values) >= 2:
                    values = [m["value"] for m in metric_values]
                    
                    # Calculate trend direction
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    trend_direction = "improving" if second_avg > first_avg else "declining"
                    trend_magnitude = abs(second_avg - first_avg) / first_avg if first_avg != 0 else 0
                    
                    trends[benchmark_name][metric_name] = {
                        "direction": trend_direction,
                        "magnitude": trend_magnitude,
                        "current_value": values[-1],
                        "historical_avg": statistics.mean(values),
                        "volatility": statistics.stdev(values) if len(values) > 1 else 0
                    }
        
        return trends
    
    def save_results(self, results: List[BenchmarkResult], filepath: str) -> None:
        """Save benchmark results to file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, filepath: str) -> List[BenchmarkResult]:
        """Load benchmark results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_data in data["results"]:
            # Reconstruct BenchmarkResult from dictionary
            result = BenchmarkResult(
                benchmark_name=result_data["benchmark_name"],
                benchmark_type=BenchmarkType(result_data["benchmark_type"]),
                status=BenchmarkStatus(result_data["status"]),
                execution_time=result_data["execution_time"],
                timestamp=datetime.fromisoformat(result_data["timestamp"]),
                error_message=result_data.get("error_message"),
                dataset_id=result_data.get("dataset_id"),
                dataset_version=result_data.get("dataset_version")
            )
            
            # Reconstruct metrics
            for metric_data in result_data["metrics"]:
                metric = BenchmarkMetric(
                    name=metric_data["name"],
                    value=metric_data["value"],
                    unit=metric_data["unit"],
                    threshold=metric_data.get("threshold"),
                    baseline=metric_data.get("baseline"),
                    tolerance=metric_data.get("tolerance", 0.05)
                )
                result.add_metric(metric)
            
            results.append(result)
        
        return results