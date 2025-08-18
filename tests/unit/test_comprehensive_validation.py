"""
Comprehensive unit tests for the validation and testing suite.

This module tests all components of the validation system including dataset validation,
quality benchmarks, performance profiling, integration testing, and sample data generation.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
from src.qudata.validation import (
    # Dataset Validator
    DatasetValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
    ValidationRule,
    SchemaValidationRule,
    ContentValidationRule,
    QualityValidationRule,
    ConsistencyValidationRule,
    CompletenessValidationRule,
    
    # Quality Benchmarks
    QualityBenchmarks,
    BenchmarkResult,
    BenchmarkMetric,
    BenchmarkType,
    BenchmarkStatus,
    Benchmark,
    QualityBenchmark,
    PerformanceBenchmark,
    AccuracyBenchmark,
    ConsistencyBenchmark,
    BenchmarkSuite,
    
    # Performance Profiler
    PerformanceProfiler,
    ProfilingResult,
    ResourceSnapshot,
    PerformanceMetrics,
    ResourceMonitor,
    ProfilingContext,
    ResourceType,
    ProfilerMode,
    
    # Integration Tester
    IntegrationTester,
    TestCase,
    TestResult,
    TestSuite,
    TestExecution,
    TestType,
    TestStatus,
    PipelineTestRunner,
    TestDataProvider,
    FileTestDataProvider,
    
    # Sample Data Generator
    SampleDataGenerator,
    SampleDocument,
    GenerationConfig,
    ContentGenerator,
    TextContentGenerator,
    JSONContentGenerator,
    CSVContentGenerator,
    HTMLContentGenerator,
    DataType,
    ContentCategory,
    QualityLevel
)


class TestDatasetValidator(unittest.TestCase):
    """Test dataset validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DatasetValidator()
        self.sample_dataset = self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en",
            author="Test Author",
            creation_date=datetime.now(),
            domain="test",
            topics=["testing"],
            entities=[],
            quality_score=0.8
        )
        
        document = Document(
            id="test_doc_1",
            source_path="/test/path.txt",
            content="This is test content for validation.",
            metadata=metadata,
            processing_timestamp=datetime.now(),
            version="1.0"
        )
        
        quality_metrics = QualityMetrics(
            overall_score=0.8,
            length_score=0.8,
            language_score=0.95,
            coherence_score=0.7,
            uniqueness_score=0.8,
            completeness_score=0.9
        )
        
        return Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0",
            documents=[document],
            metadata={},
            quality_metrics=quality_metrics
        )
    
    def test_schema_validation_rule(self):
        """Test schema validation rule."""
        rule = SchemaValidationRule()
        issues = rule.validate(self.sample_dataset)
        
        # Should pass for valid dataset
        self.assertEqual(len(issues), 0)
        
        # Test with invalid dataset
        invalid_dataset = Dataset(
            id="",  # Invalid empty ID
            name="",  # Invalid empty name
            version="",  # Invalid empty version
            documents=[],
            metadata={},
            quality_metrics=None
        )
        
        issues = rule.validate(invalid_dataset)
        self.assertGreater(len(issues), 0)
        
        # Check for critical issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        self.assertGreater(len(critical_issues), 0)
    
    def test_content_validation_rule(self):
        """Test content validation rule."""
        rule = ContentValidationRule(min_content_length=5, max_content_length=1000)
        issues = rule.validate(self.sample_dataset)
        
        # Should pass for valid content
        self.assertEqual(len(issues), 0)
        
        # Test with short content
        short_metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=2,
            language="en",
            author="Test Author",
            creation_date=datetime.now(),
            domain="test",
            topics=["testing"],
            entities=[],
            quality_score=0.5
        )
        
        short_content_doc = Document(
            id="short_doc",
            source_path="/test/short.txt",
            content="Hi",  # Too short
            metadata=short_metadata,
            processing_timestamp=datetime.now(),
            version="1.0"
        )
        
        short_dataset = Dataset(
            id="short_dataset",
            name="Short Dataset",
            version="1.0",
            documents=[short_content_doc],
            metadata={},
            quality_metrics=self.sample_dataset.quality_metrics
        )
        
        issues = rule.validate(short_dataset)
        self.assertGreater(len(issues), 0)
    
    def test_quality_validation_rule(self):
        """Test quality validation rule."""
        rule = QualityValidationRule(min_quality_score=0.5)
        issues = rule.validate(self.sample_dataset)
        
        # Should pass for high-quality dataset
        self.assertEqual(len(issues), 0)
        
        # Test with low-quality dataset
        low_quality_metrics = QualityMetrics(
            overall_score=0.3,  # Below threshold
            length_score=0.3,
            language_score=0.5,
            coherence_score=0.2,
            uniqueness_score=0.3,
            completeness_score=0.4
        )
        
        low_quality_dataset = Dataset(
            id="low_quality_dataset",
            name="Low Quality Dataset",
            version="1.0",
            documents=self.sample_dataset.documents,
            metadata={},
            quality_metrics=low_quality_metrics
        )
        
        issues = rule.validate(low_quality_dataset)
        self.assertGreater(len(issues), 0)
    
    def test_consistency_validation_rule(self):
        """Test consistency validation rule."""
        rule = ConsistencyValidationRule()
        issues = rule.validate(self.sample_dataset)
        
        # Should pass for consistent dataset
        self.assertEqual(len(issues), 0)
        
        # Test with duplicate IDs
        duplicate_metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=100,
            language="en",
            author="Test Author",
            creation_date=datetime.now(),
            domain="test",
            topics=["testing"],
            entities=[],
            quality_score=0.7
        )
        
        duplicate_doc = Document(
            id="test_doc_1",  # Same ID as existing document
            source_path="/test/duplicate.txt",
            content="Duplicate content",
            metadata=duplicate_metadata,
            processing_timestamp=datetime.now(),
            version="1.0"
        )
        
        duplicate_dataset = Dataset(
            id="duplicate_dataset",
            name="Duplicate Dataset",
            version="1.0",
            documents=[self.sample_dataset.documents[0], duplicate_doc],
            metadata={},
            quality_metrics=self.sample_dataset.quality_metrics
        )
        
        issues = rule.validate(duplicate_dataset)
        self.assertGreater(len(issues), 0)
        
        # Check for critical duplicate ID issue
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        self.assertGreater(len(critical_issues), 0)
    
    def test_completeness_validation_rule(self):
        """Test completeness validation rule."""
        rule = CompletenessValidationRule()
        issues = rule.validate(self.sample_dataset)
        
        # Should pass for complete dataset
        self.assertEqual(len(issues), 0)
        
        # Test with empty dataset
        empty_dataset = Dataset(
            id="empty_dataset",
            name="Empty Dataset",
            version="1.0",
            documents=[],  # No documents
            metadata={},
            quality_metrics=None
        )
        
        issues = rule.validate(empty_dataset)
        self.assertGreater(len(issues), 0)
        
        # Check for critical empty dataset issue
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        self.assertGreater(len(critical_issues), 0)
    
    def test_dataset_validator_integration(self):
        """Test complete dataset validator."""
        result = self.validator.validate(self.sample_dataset)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.documents_validated, 1)
        self.assertGreater(result.validation_duration, 0)
        self.assertGreater(len(result.rules_applied), 0)
    
    def test_validation_result_methods(self):
        """Test ValidationResult helper methods."""
        result = ValidationResult(is_valid=True)
        
        # Add test issues
        result.add_issue(ValidationIssue(
            category=ValidationCategory.SCHEMA,
            severity=ValidationSeverity.ERROR,
            message="Test error"
        ))
        
        result.add_issue(ValidationIssue(
            category=ValidationCategory.QUALITY,
            severity=ValidationSeverity.WARNING,
            message="Test warning"
        ))
        
        # Test methods
        self.assertFalse(result.is_valid)  # Should be false after adding error
        self.assertEqual(len(result.get_error_issues()), 1)
        self.assertEqual(len(result.get_issues_by_severity(ValidationSeverity.WARNING)), 1)
        self.assertEqual(len(result.get_issues_by_category(ValidationCategory.SCHEMA)), 1)
        
        summary = result.get_summary()
        self.assertEqual(summary["error"], 1)
        self.assertEqual(summary["warning"], 1)


class TestQualityBenchmarks(unittest.TestCase):
    """Test quality benchmarking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmarks = QualityBenchmarks()
        self.sample_dataset = self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for benchmarking."""
        documents = []
        for i in range(5):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=100 + i * 10,
                language="en",
                author=f"Author {i}",
                creation_date=datetime.now(),
                domain="test",
                topics=[f"topic_{i}"],
                entities=[],
                quality_score=0.7 + i * 0.05
            )
            
            document = Document(
                id=f"doc_{i}",
                source_path=f"/test/doc_{i}.txt",
                content=f"Test content for document {i}. " * (10 + i),
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        quality_metrics = QualityMetrics(
            overall_score=0.8,
            length_score=0.8,
            language_score=0.95,
            coherence_score=0.7,
            uniqueness_score=0.8,
            completeness_score=0.9
        )
        
        return Dataset(
            id="benchmark_dataset",
            name="Benchmark Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=quality_metrics
        )
    
    def test_quality_benchmark(self):
        """Test quality benchmark execution."""
        benchmark = QualityBenchmark()
        result = benchmark.run(self.sample_dataset)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_type, BenchmarkType.QUALITY)
        self.assertGreater(len(result.metrics), 0)
        self.assertGreater(result.execution_time, 0)
        
        # Check for expected metrics
        metric_names = [m.name for m in result.metrics]
        self.assertIn("avg_quality_score", metric_names)
        self.assertIn("content_completeness", metric_names)
        self.assertIn("metadata_completeness", metric_names)
    
    def test_performance_benchmark(self):
        """Test performance benchmark execution."""
        benchmark = PerformanceBenchmark()
        result = benchmark.run(self.sample_dataset)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_type, BenchmarkType.PERFORMANCE)
        self.assertGreater(len(result.metrics), 0)
        
        # Check for expected metrics
        metric_names = [m.name for m in result.metrics]
        self.assertIn("docs_per_second", metric_names)
        self.assertIn("chars_per_second", metric_names)
    
    def test_accuracy_benchmark(self):
        """Test accuracy benchmark execution."""
        benchmark = AccuracyBenchmark()
        result = benchmark.run(self.sample_dataset)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_type, BenchmarkType.ACCURACY)
        self.assertGreater(len(result.metrics), 0)
    
    def test_consistency_benchmark(self):
        """Test consistency benchmark execution."""
        benchmark = ConsistencyBenchmark()
        result = benchmark.run(self.sample_dataset)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_type, BenchmarkType.CONSISTENCY)
        self.assertGreater(len(result.metrics), 0)
        
        # Check for expected metrics
        metric_names = [m.name for m in result.metrics]
        self.assertIn("duplicate_ratio", metric_names)
        self.assertIn("encoding_consistency", metric_names)
    
    def test_benchmark_suite(self):
        """Test benchmark suite execution."""
        suite = BenchmarkSuite("test_suite")
        suite.add_benchmark(QualityBenchmark())
        suite.add_benchmark(PerformanceBenchmark())
        
        results = suite.run_all(self.sample_dataset)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], BenchmarkResult)
        self.assertIsInstance(results[1], BenchmarkResult)
        
        summary = suite.get_summary(results)
        self.assertEqual(summary["total_benchmarks"], 2)
        self.assertGreaterEqual(summary["success_rate"], 0)
    
    def test_benchmark_metric(self):
        """Test benchmark metric functionality."""
        metric = BenchmarkMetric(
            name="test_metric",
            value=0.8,
            unit="score",
            threshold=0.7,
            baseline=0.75,
            tolerance=0.1
        )
        
        self.assertTrue(metric.meets_threshold())
        self.assertTrue(metric.is_within_tolerance())
        
        # Test metric that fails threshold
        failing_metric = BenchmarkMetric(
            name="failing_metric",
            value=0.5,
            unit="score",
            threshold=0.7
        )
        
        self.assertFalse(failing_metric.meets_threshold())
    
    def test_quality_benchmarks_integration(self):
        """Test complete quality benchmarks system."""
        results = self.benchmarks.run_suite("quality", self.sample_dataset)
        
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], BenchmarkResult)
        
        # Test baseline setting
        self.benchmarks.set_baselines_from_dataset(self.sample_dataset, "quality")
        
        # Test regression detection
        regressions = self.benchmarks.detect_regressions(results, tolerance=0.1)
        self.assertIsInstance(regressions, list)


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_resource_monitor(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test resource monitoring."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=1024*1024*1024, available=512*1024*1024)
        mock_disk.return_value = Mock(read_bytes=1024*1024, write_bytes=512*1024)
        mock_net.return_value = Mock(bytes_sent=1024*100, bytes_recv=1024*200)
        
        monitor = ResourceMonitor()
        snapshot = monitor.take_snapshot()
        
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertEqual(snapshot.cpu_percent, 50.0)
        self.assertEqual(snapshot.memory_percent, 60.0)
        self.assertGreater(snapshot.memory_used_mb, 0)
    
    def test_profiling_session(self):
        """Test profiling session management."""
        session_id = self.profiler.start_profiling("test_session")
        
        self.assertIn(session_id, self.profiler.active_sessions)
        
        # Simulate some work
        time.sleep(0.1)
        
        result = self.profiler.stop_profiling(session_id)
        
        self.assertIsInstance(result, ProfilingResult)
        self.assertEqual(result.session_id, session_id)
        self.assertGreater(len(result.snapshots), 0)
        self.assertIsNotNone(result.metrics)
        self.assertNotIn(session_id, self.profiler.active_sessions)
    
    def test_function_profiling(self):
        """Test function profiling."""
        def test_function(x, y):
            time.sleep(0.05)  # Simulate work
            return x + y
        
        result, profiling_result = self.profiler.profile_function(test_function, 1, 2)
        
        self.assertEqual(result, 3)
        self.assertIsInstance(profiling_result, ProfilingResult)
        self.assertIn("test_function", profiling_result.function_profiles)
        self.assertTrue(profiling_result.function_profiles["test_function"]["success"])
    
    def test_profiling_context(self):
        """Test profiling context manager."""
        with self.profiler.profile_context("context_test") as ctx:
            time.sleep(0.05)  # Simulate work
        
        result = ctx.get_result()
        self.assertIsInstance(result, ProfilingResult)
        self.assertGreater(len(result.snapshots), 0)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        result = ProfilingResult(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add test snapshots
        for i in range(3):
            snapshot = ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=50.0 + i * 10,
                memory_percent=60.0 + i * 5,
                memory_used_mb=1000.0 + i * 100,
                memory_available_mb=500.0,
                disk_read_mb=10.0 + i,
                disk_write_mb=5.0 + i,
                network_sent_mb=1.0 + i,
                network_recv_mb=2.0 + i
            )
            result.add_snapshot(snapshot)
        
        metrics = result.calculate_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.cpu_avg, 0)
        self.assertGreater(metrics.memory_avg_mb, 0)
        self.assertGreater(metrics.cpu_max, metrics.cpu_avg)
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        result = ProfilingResult(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add high CPU usage snapshot
        high_cpu_snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=95.0,  # High CPU usage
            memory_percent=50.0,
            memory_used_mb=1000.0,
            memory_available_mb=500.0,
            disk_read_mb=10.0,
            disk_write_mb=5.0,
            network_sent_mb=1.0,
            network_recv_mb=2.0
        )
        result.add_snapshot(high_cpu_snapshot)
        
        result.calculate_metrics()
        bottlenecks = result.detect_bottlenecks()
        
        self.assertGreater(len(bottlenecks), 0)
        cpu_bottlenecks = [b for b in bottlenecks if b["type"] == "cpu"]
        self.assertGreater(len(cpu_bottlenecks), 0)


class TestIntegrationTester(unittest.TestCase):
    """Test integration testing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_pipeline = Mock()
        self.tester = IntegrationTester(self.mock_pipeline)
    
    def test_test_case_creation(self):
        """Test test case creation."""
        test_case = TestCase(
            name="test_basic_processing",
            description="Test basic file processing",
            test_type=TestType.SMOKE,
            input_data={"files": [{"type": "txt", "content": "test"}]},
            expected_output={"documents_processed": {"min": 1}}
        )
        
        self.assertEqual(test_case.name, "test_basic_processing")
        self.assertEqual(test_case.test_type, TestType.SMOKE)
        self.assertIn("files", test_case.input_data)
    
    def test_test_suite_management(self):
        """Test test suite management."""
        suite = TestSuite("test_suite", "Test suite description")
        
        test_case = TestCase(
            name="test_case_1",
            description="First test case",
            test_type=TestType.FUNCTIONAL,
            input_data={},
            expected_output={}
        )
        
        suite.add_test_case(test_case)
        
        self.assertEqual(len(suite.test_cases), 1)
        self.assertEqual(suite.test_cases[0].name, "test_case_1")
        
        functional_tests = suite.get_test_cases_by_type(TestType.FUNCTIONAL)
        self.assertEqual(len(functional_tests), 1)
    
    def test_file_test_data_provider(self):
        """Test file test data provider."""
        provider = FileTestDataProvider()
        
        test_case = TestCase(
            name="test_file_creation",
            description="Test file creation",
            test_type=TestType.SMOKE,
            input_data={
                "files": [
                    {"type": "txt", "content": "Test content", "size": 100}
                ]
            },
            expected_output={}
        )
        
        test_data = provider.generate_test_data(test_case)
        
        self.assertIn("files", test_data)
        self.assertEqual(len(test_data["files"]), 1)
        
        # Check file was created
        file_path = test_data["files"][0]
        self.assertTrue(Path(file_path).exists())
        
        # Cleanup
        provider.cleanup_test_data(test_data)
        self.assertFalse(Path(file_path).exists())
    
    def test_test_execution(self):
        """Test test execution tracking."""
        execution = TestExecution(
            suite_name="test_suite",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Add test results
        result1 = TestResult(
            test_case=TestCase("test1", "desc", TestType.SMOKE, {}, {}),
            status=TestStatus.PASSED,
            execution_time=1.0
        )
        
        result2 = TestResult(
            test_case=TestCase("test2", "desc", TestType.SMOKE, {}, {}),
            status=TestStatus.FAILED,
            execution_time=2.0,
            error_message="Test failed"
        )
        
        execution.add_result(result1)
        execution.add_result(result2)
        
        self.assertEqual(execution.summary["total"], 2)
        self.assertEqual(execution.summary["passed"], 1)
        self.assertEqual(execution.summary["failed"], 1)
        self.assertEqual(execution.get_success_rate(), 0.5)
        
        failed_tests = execution.get_failed_tests()
        self.assertEqual(len(failed_tests), 1)
        self.assertEqual(failed_tests[0].test_case.name, "test2")
    
    def test_integration_tester_default_suites(self):
        """Test default test suites creation."""
        self.assertIn("smoke", self.tester.test_suites)
        self.assertIn("functional", self.tester.test_suites)
        self.assertIn("performance", self.tester.test_suites)
        self.assertIn("e2e", self.tester.test_suites)
        
        smoke_suite = self.tester.test_suites["smoke"]
        self.assertGreater(len(smoke_suite.test_cases), 0)


class TestSampleDataGenerator(unittest.TestCase):
    """Test sample data generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SampleDataGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.generator.cleanup_generated_files()
    
    def test_text_content_generator(self):
        """Test text content generation."""
        generator = TextContentGenerator()
        
        config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.TECHNICAL,
            quality_level=QualityLevel.HIGH,
            size_range=(100, 200)
        )
        
        content = generator.generate_content(config)
        
        self.assertIsInstance(content, str)
        self.assertGreaterEqual(len(content), 100)
        self.assertLessEqual(len(content), 200)
        
        metadata = generator.get_expected_metadata(config)
        self.assertIn("expected_quality_score", metadata)
        self.assertIn("expected_language", metadata)
        self.assertIn("expected_domain", metadata)
    
    def test_json_content_generator(self):
        """Test JSON content generation."""
        generator = JSONContentGenerator()
        
        config = GenerationConfig(
            data_type=DataType.JSON,
            content_category=ContentCategory.TECHNICAL,
            quality_level=QualityLevel.HIGH
        )
        
        content = generator.generate_content(config)
        
        self.assertIsInstance(content, str)
        
        # Should be valid JSON
        try:
            parsed = json.loads(content)
            self.assertIsInstance(parsed, dict)
        except json.JSONDecodeError:
            self.fail("Generated content is not valid JSON")
    
    def test_csv_content_generator(self):
        """Test CSV content generation."""
        generator = CSVContentGenerator()
        
        config = GenerationConfig(
            data_type=DataType.CSV,
            content_category=ContentCategory.FINANCIAL,
            quality_level=QualityLevel.HIGH
        )
        
        content = generator.generate_content(config)
        
        self.assertIsInstance(content, str)
        self.assertIn(",", content)  # Should contain CSV separators
        self.assertIn("\n", content)  # Should contain line breaks
        
        lines = content.split("\n")
        self.assertGreater(len(lines), 1)  # Should have header + data
    
    def test_html_content_generator(self):
        """Test HTML content generation."""
        generator = HTMLContentGenerator()
        
        config = GenerationConfig(
            data_type=DataType.HTML,
            content_category=ContentCategory.NEWS,
            quality_level=QualityLevel.HIGH
        )
        
        content = generator.generate_content(config)
        
        self.assertIsInstance(content, str)
        self.assertIn("<html>", content)
        self.assertIn("</html>", content)
        self.assertIn("<body>", content)
    
    def test_corrupted_content_generation(self):
        """Test corrupted content generation."""
        generator = TextContentGenerator()
        
        config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.CORRUPTED,
            quality_level=QualityLevel.CORRUPTED
        )
        
        content = generator.generate_content(config)
        
        self.assertIsInstance(content, str)
        # Should contain corruption artifacts
        self.assertTrue(any(char in content for char in ["�", "\ufffd", "\x00", "\x01", "\x02"]))
    
    def test_sample_document_generation(self):
        """Test sample document generation."""
        config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.GENERAL,
            quality_level=QualityLevel.HIGH,
            size_range=(50, 100)
        )
        
        document = self.generator.generate_sample_document(config)
        
        self.assertIsInstance(document, SampleDocument)
        self.assertIsInstance(document.content, str)
        self.assertIsInstance(document.metadata, dict)
        self.assertIsNotNone(document.expected_quality_score)
        self.assertIsNotNone(document.expected_language)
        self.assertIsNotNone(document.expected_domain)
    
    def test_sample_dataset_generation(self):
        """Test sample dataset generation."""
        configs = [
            GenerationConfig(
                data_type=DataType.TEXT,
                content_category=ContentCategory.TECHNICAL,
                quality_level=QualityLevel.HIGH
            ),
            GenerationConfig(
                data_type=DataType.JSON,
                content_category=ContentCategory.GENERAL,
                quality_level=QualityLevel.MEDIUM
            )
        ]
        
        documents = self.generator.generate_sample_dataset(configs, save_files=True)
        
        self.assertEqual(len(documents), 2)
        self.assertIsInstance(documents[0], SampleDocument)
        self.assertIsInstance(documents[1], SampleDocument)
        
        # Check files were created
        for doc in documents:
            self.assertIsNotNone(doc.file_path)
            self.assertTrue(Path(doc.file_path).exists())
    
    def test_test_suite_data_generation(self):
        """Test comprehensive test suite data generation."""
        test_suite = self.generator.generate_test_suite_data("comprehensive_test")
        
        self.assertIn("high_quality", test_suite)
        self.assertIn("medium_quality", test_suite)
        self.assertIn("low_quality", test_suite)
        self.assertIn("corrupted", test_suite)
        self.assertIn("large_files", test_suite)
        
        # Check each category has documents
        for category, documents in test_suite.items():
            self.assertGreater(len(documents), 0)
            for doc in documents:
                self.assertIsInstance(doc, SampleDocument)
    
    def test_generation_config_serialization(self):
        """Test generation config serialization."""
        config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.TECHNICAL,
            quality_level=QualityLevel.HIGH,
            size_range=(100, 500),
            language="en",
            corruption_rate=0.1
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["data_type"], "text")
        self.assertEqual(config_dict["content_category"], "technical")
        self.assertEqual(config_dict["quality_level"], "high")
        self.assertEqual(config_dict["size_range"], (100, 500))
        self.assertEqual(config_dict["corruption_rate"], 0.1)


class TestValidationIntegration(unittest.TestCase):
    """Test integration between all validation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SampleDataGenerator(self.temp_dir)
        self.validator = DatasetValidator()
        self.benchmarks = QualityBenchmarks()
        self.profiler = PerformanceProfiler()
        
        # Create mock pipeline for integration tester
        self.mock_pipeline = Mock()
        self.integration_tester = IntegrationTester(self.mock_pipeline)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.generator.cleanup_generated_files()
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow from data generation to reporting."""
        # 1. Generate sample dataset
        configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.LOW)
        ]
        
        sample_documents = self.generator.generate_sample_dataset(configs, save_files=True)
        
        # 2. Convert to Dataset format for validation
        documents = []
        for i, sample_doc in enumerate(sample_documents):
            metadata = DocumentMetadata(
                file_type=sample_doc.generation_config.data_type.value,
                size_bytes=len(sample_doc.content.encode('utf-8')),
                language=sample_doc.expected_language or "en",
                domain=sample_doc.expected_domain or "general",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.5
            )
            
            document = Document(
                id=f"test_doc_{i}",
                source_path=sample_doc.file_path or f"test_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        quality_metrics = QualityMetrics(
            overall_score=0.7,
            length_score=0.7,
            language_score=0.9,
            coherence_score=0.6,
            uniqueness_score=0.7,
            completeness_score=0.8
        )
        
        dataset = Dataset(
            id="integration_test_dataset",
            name="Integration Test Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=quality_metrics
        )
        
        # 3. Validate dataset
        validation_result = self.validator.validate(dataset)
        self.assertIsInstance(validation_result, ValidationResult)
        
        # 4. Run benchmarks
        benchmark_results = self.benchmarks.run_suite("quality", dataset)
        self.assertGreater(len(benchmark_results), 0)
        
        # 5. Profile the validation process
        with self.profiler.profile_context("validation_profiling") as ctx:
            # Simulate validation work
            time.sleep(0.1)
            validation_result2 = self.validator.validate(dataset)
        
        profiling_result = ctx.get_result()
        self.assertIsInstance(profiling_result, ProfilingResult)
        self.assertGreater(len(profiling_result.snapshots), 0)
        
        # 6. Generate comprehensive report
        report = self._generate_comprehensive_report(
            dataset, validation_result, benchmark_results, profiling_result
        )
        
        self.assertIn("dataset_summary", report)
        self.assertIn("validation_summary", report)
        self.assertIn("benchmark_summary", report)
        self.assertIn("performance_summary", report)
    
    def _generate_comprehensive_report(self, dataset: Dataset, validation_result: ValidationResult,
                                     benchmark_results: List[BenchmarkResult],
                                     profiling_result: ProfilingResult) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "dataset_summary": {
                "id": dataset.id,
                "document_count": len(dataset.documents),
                "total_content_size": sum(len(doc.content) for doc in dataset.documents),
                "avg_quality_score": sum(doc.metadata.quality_score for doc in dataset.documents) / len(dataset.documents)
            },
            "validation_summary": {
                "is_valid": validation_result.is_valid,
                "total_issues": len(validation_result.issues),
                "critical_issues": len(validation_result.get_critical_issues()),
                "error_issues": len(validation_result.get_error_issues()),
                "validation_duration": validation_result.validation_duration
            },
            "benchmark_summary": {
                "total_benchmarks": len(benchmark_results),
                "passed_benchmarks": sum(1 for r in benchmark_results if r.status == BenchmarkStatus.PASSED),
                "failed_benchmarks": sum(1 for r in benchmark_results if r.status == BenchmarkStatus.FAILED),
                "total_metrics": sum(len(r.metrics) for r in benchmark_results)
            },
            "performance_summary": {
                "session_id": profiling_result.session_id,
                "duration": profiling_result.metrics.duration_seconds if profiling_result.metrics else 0,
                "peak_memory_mb": profiling_result.metrics.memory_max_mb if profiling_result.metrics else 0,
                "avg_cpu_percent": profiling_result.metrics.cpu_avg if profiling_result.metrics else 0,
                "bottleneck_count": len(profiling_result.bottlenecks)
            }
        }
    
    def test_validation_with_edge_cases(self):
        """Test validation with edge case datasets."""
        # Test with empty dataset
        empty_dataset = Dataset(
            id="empty_test",
            name="Empty Test",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=None
        )
        
        result = self.validator.validate(empty_dataset)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.get_critical_issues()), 0)
        
        # Test with corrupted content dataset
        corrupted_config = GenerationConfig(
            DataType.TEXT,
            ContentCategory.CORRUPTED,
            QualityLevel.CORRUPTED
        )
        
        corrupted_doc = self.generator.generate_sample_document(corrupted_config)
        
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(corrupted_doc.content.encode('utf-8')),
            language="unknown",
            domain="unknown",
            topics=[],
            entities=[],
            quality_score=0.1
        )
        
        document = Document(
            id="corrupted_doc",
            source_path="corrupted.txt",
            content=corrupted_doc.content,
            metadata=metadata,
            processing_timestamp=datetime.now(),
            version="1.0"
        )
        
        corrupted_dataset = Dataset(
            id="corrupted_test",
            name="Corrupted Test",
            version="1.0",
            documents=[document],
            metadata={},
            quality_metrics=QualityMetrics(0.1, 0.1, 0.2, 0.1, 0.1)
        )
        
        result = self.validator.validate(corrupted_dataset)
        # Should have validation issues but not necessarily be invalid
        self.assertGreater(len(result.issues), 0)
    
    def test_benchmark_regression_detection(self):
        """Test benchmark regression detection."""
        # Create baseline dataset
        baseline_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH)
            for _ in range(5)
        ]
        
        baseline_docs = self.generator.generate_sample_dataset(baseline_configs)
        baseline_dataset = self._create_dataset_from_samples(baseline_docs, "baseline")
        
        # Set baselines
        self.benchmarks.set_baselines_from_dataset(baseline_dataset, "quality")
        
        # Create regression dataset (lower quality)
        regression_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.LOW)
            for _ in range(5)
        ]
        
        regression_docs = self.generator.generate_sample_dataset(regression_configs)
        regression_dataset = self._create_dataset_from_samples(regression_docs, "regression")
        
        # Run benchmarks on regression dataset
        regression_results = self.benchmarks.run_suite("quality", regression_dataset)
        
        # Detect regressions
        regressions = self.benchmarks.detect_regressions(regression_results, tolerance=0.1)
        
        # Should detect some regressions due to lower quality
        self.assertGreater(len(regressions), 0)
        
        for regression in regressions:
            self.assertIn("benchmark", regression)
            self.assertIn("metric", regression)
            self.assertIn("current_value", regression)
            self.assertIn("baseline_value", regression)
            self.assertIn("degradation_percent", regression)
    
    def _create_dataset_from_samples(self, sample_docs: List[SampleDocument], dataset_id: str) -> Dataset:
        """Helper to create Dataset from sample documents."""
        documents = []
        total_quality = 0
        
        for i, sample_doc in enumerate(sample_docs):
            metadata = DocumentMetadata(
                file_type=sample_doc.generation_config.data_type.value,
                size_bytes=len(sample_doc.content.encode('utf-8')),
                language=sample_doc.expected_language or "en",
                domain=sample_doc.expected_domain or "general",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.5
            )
            
            document = Document(
                id=f"{dataset_id}_doc_{i}",
                source_path=sample_doc.file_path or f"{dataset_id}_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
            total_quality += document.metadata.quality_score
        
        avg_quality = total_quality / len(documents) if documents else 0
        
        quality_metrics = QualityMetrics(
            overall_score=avg_quality,
            length_score=avg_quality,
            language_score=0.9,
            coherence_score=0.7,
            uniqueness_score=avg_quality,
            completeness_score=0.8
        )
        
        return Dataset(
            id=dataset_id,
            name=f"{dataset_id.title()} Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=quality_metrics
        )
    
    def test_performance_profiling_accuracy(self):
        """Test performance profiling accuracy and bottleneck detection."""
        # Test CPU-intensive operation
        def cpu_intensive_task():
            # Simulate CPU work
            total = 0
            for i in range(100000):
                total += i * i
            return total
        
        result, profiling_result = self.profiler.profile_function(cpu_intensive_task)
        
        self.assertIsInstance(result, int)
        self.assertIsInstance(profiling_result, ProfilingResult)
        self.assertGreater(profiling_result.metrics.cpu_avg, 0)
        
        # Test memory-intensive operation
        def memory_intensive_task():
            # Simulate memory allocation
            data = []
            for i in range(10000):
                data.append("x" * 100)
            return len(data)
        
        result, profiling_result = self.profiler.profile_function(memory_intensive_task)
        
        self.assertIsInstance(result, int)
        self.assertGreater(profiling_result.metrics.memory_avg_mb, 0)
        
        # Check bottleneck detection
        bottlenecks = profiling_result.detect_bottlenecks()
        recommendations = profiling_result.generate_recommendations()
        
        self.assertIsInstance(bottlenecks, list)
        self.assertIsInstance(recommendations, list)
    
    def test_integration_test_execution(self):
        """Test integration test execution with mock pipeline."""
        # Configure mock pipeline to return test dataset
        test_dataset = self._create_dataset_from_samples([
            self.generator.generate_sample_document(
                GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH)
            )
        ], "mock_test")
        
        self.mock_pipeline.process_files.return_value = test_dataset
        self.mock_pipeline.export_dataset.return_value = "/tmp/test_export.jsonl"
        
        # Run smoke test suite
        execution = self.integration_tester.run_test_suite("smoke")
        
        self.assertIsInstance(execution, TestExecution)
        self.assertEqual(execution.suite_name, "smoke")
        self.assertGreater(len(execution.results), 0)
        
        # Check test results
        for result in execution.results:
            self.assertIsInstance(result, TestResult)
            self.assertIn(result.status, [TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR])
    
    def test_comprehensive_error_handling(self):
        """Test error handling across all validation components."""
        # Test validator with invalid input
        with self.assertRaises(AttributeError):
            self.validator.validate(None)
        
        # Test benchmarks with invalid dataset
        invalid_dataset = Dataset(
            id="invalid",
            name="Invalid",
            version="1.0",
            documents=[],  # Empty documents should cause issues
            metadata={},
            quality_metrics=None
        )
        
        # Should handle gracefully and return results with errors
        results = self.benchmarks.run_suite("quality", invalid_dataset)
        self.assertIsInstance(results, list)
        
        # Test profiler error handling
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            self.profiler.profile_function(failing_function)
        
        # Test sample generator with invalid config
        invalid_config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.TECHNICAL,
            quality_level=QualityLevel.HIGH,
            size_range=(-100, -50)  # Invalid size range
        )
        
        # Should handle gracefully
        try:
            doc = self.generator.generate_sample_document(invalid_config)
            self.assertIsInstance(doc, SampleDocument)
        except Exception as e:
            # Should not crash, but may produce unexpected results
            self.assertIsInstance(e, Exception)


    def test_sample_dataset_generation(self):
        """Test sample dataset generation."""
        configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.CSV, ContentCategory.FINANCIAL, QualityLevel.LOW)
        ]
        
        documents = self.generator.generate_sample_dataset(configs, save_files=True)
        
        self.assertEqual(len(documents), 3)
        for document in documents:
            self.assertIsInstance(document, SampleDocument)
            self.assertIsNotNone(document.file_path)
            self.assertTrue(Path(document.file_path).exists())
    
    def test_test_suite_data_generation(self):
        """Test comprehensive test suite data generation."""
        test_suite = self.generator.generate_test_suite_data("comprehensive_test")
        
        self.assertIn("high_quality", test_suite)
        self.assertIn("medium_quality", test_suite)
        self.assertIn("low_quality", test_suite)
        self.assertIn("corrupted", test_suite)
        self.assertIn("large_files", test_suite)
        
        # Verify each category has documents
        for category, documents in test_suite.items():
            self.assertGreater(len(documents), 0)
            for document in documents:
                self.assertIsInstance(document, SampleDocument)
    
    def test_regression_test_data_generation(self):
        """Test regression test data generation."""
        documents = self.generator.generate_regression_test_data()
        
        self.assertGreater(len(documents), 0)
        
        # Check variety of quality levels
        quality_levels = set(doc.generation_config.quality_level for doc in documents)
        self.assertGreater(len(quality_levels), 1)
    
    def test_metadata_saving(self):
        """Test test suite metadata saving."""
        test_suite = {
            "high_quality": [
                SampleDocument(
                    content="Test content",
                    metadata={"test": True},
                    expected_quality_score=0.9
                )
            ]
        }
        
        metadata_path = Path(self.temp_dir) / "test_metadata.json"
        self.generator.save_test_suite_metadata(test_suite, str(metadata_path))
        
        self.assertTrue(metadata_path.exists())
        
        # Load and verify metadata
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertIn("high_quality", loaded_metadata)
        self.assertEqual(len(loaded_metadata["high_quality"]), 1)
    
    def test_cleanup_generated_files(self):
        """Test cleanup of generated files."""
        # Generate some files
        configs = [GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH)]
        documents = self.generator.generate_sample_dataset(configs, save_files=True)
        
        # Verify files exist
        for document in documents:
            self.assertTrue(Path(document.file_path).exists())
        
        # Cleanup
        self.generator.cleanup_generated_files()
        
        # Verify files are removed
        for document in documents:
            self.assertFalse(Path(document.file_path).exists())


class TestComprehensiveValidationWorkflow(unittest.TestCase):
    """Test complete validation workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_pipeline = Mock()
        
        # Setup components
        self.validator = DatasetValidator()
        self.benchmarks = QualityBenchmarks()
        self.profiler = PerformanceProfiler()
        self.tester = IntegrationTester(self.mock_pipeline)
        self.generator = SampleDataGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.generator.cleanup_generated_files()
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow."""
        # 1. Generate sample data
        test_suite = self.generator.generate_test_suite_data("workflow_test")
        
        # 2. Create mock dataset from sample data
        sample_docs = test_suite["high_quality"][:3]  # Use first 3 high-quality docs
        
        documents = []
        for i, sample_doc in enumerate(sample_docs):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=len(sample_doc.content),
                language=sample_doc.expected_language or "en",
                domain=sample_doc.expected_domain or "general",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.8
            )
            
            document = Document(
                id=f"workflow_doc_{i}",
                source_path=sample_doc.file_path or f"/test/doc_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        quality_metrics = QualityMetrics(
            overall_score=0.8,
            length_score=0.8,
            language_score=0.95,
            coherence_score=0.7,
            uniqueness_score=0.8,
            completeness_score=0.9
        )
        
        dataset = Dataset(
            id="workflow_dataset",
            name="Workflow Test Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=quality_metrics
        )
        
        # 3. Validate dataset
        validation_result = self.validator.validate(dataset)
        self.assertIsInstance(validation_result, ValidationResult)
        
        # 4. Run benchmarks
        benchmark_results = self.benchmarks.run_suite("quality", dataset)
        self.assertGreater(len(benchmark_results), 0)
        
        # 5. Profile validation process
        with self.profiler.profile_context("validation_workflow") as ctx:
            # Simulate some processing
            time.sleep(0.1)
        
        profiling_result = ctx.get_result()
        self.assertIsInstance(profiling_result, ProfilingResult)
        
        # 6. Verify all components worked together
        self.assertTrue(validation_result.is_valid or len(validation_result.issues) > 0)
        self.assertGreater(len(benchmark_results[0].metrics), 0)
        self.assertGreater(len(profiling_result.snapshots), 0)
    
    def test_regression_testing_workflow(self):
        """Test regression testing workflow."""
        # Generate baseline data
        baseline_docs = self.generator.generate_regression_test_data()
        
        # Create baseline dataset
        documents = []
        for i, sample_doc in enumerate(baseline_docs[:5]):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=len(sample_doc.content),
                language="en",
                domain="test",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.7
            )
            
            document = Document(
                id=f"regression_doc_{i}",
                source_path=f"/test/regression_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        baseline_dataset = Dataset(
            id="baseline_dataset",
            name="Baseline Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(
            overall_score=0.7,
            length_score=0.7,
            language_score=0.9,
            coherence_score=0.6,
            uniqueness_score=0.7,
            completeness_score=0.8
        )
        )
        
        # Set baselines
        self.benchmarks.set_baselines_from_dataset(baseline_dataset, "quality")
        
        # Run current benchmarks
        current_results = self.benchmarks.run_suite("quality", baseline_dataset)
        
        # Detect regressions (should be none for same dataset)
        regressions = self.benchmarks.detect_regressions(current_results, tolerance=0.1)
        
        # Should have no regressions when comparing to itself
        self.assertEqual(len(regressions), 0)
    
    def test_performance_monitoring_workflow(self):
        """Test performance monitoring workflow."""
        # Generate large dataset for performance testing
        large_configs = [
            GenerationConfig(
                DataType.TEXT,
                ContentCategory.TECHNICAL,
                QualityLevel.HIGH,
                size_range=(5000, 10000)  # Larger content
            ) for _ in range(10)
        ]
        
        large_documents = self.generator.generate_sample_dataset(large_configs, save_files=False)
        
        # Create dataset
        documents = []
        for i, sample_doc in enumerate(large_documents):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=len(sample_doc.content),
                language="en",
                domain="technical",
                topics=["performance"],
                entities=[],
                quality_score=0.8
            )
            
            document = Document(
                id=f"perf_doc_{i}",
                source_path=f"/test/perf_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        large_dataset = Dataset(
            id="performance_dataset",
            name="Performance Test Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(
            overall_score=0.8,
            length_score=0.8,
            language_score=0.95,
            coherence_score=0.8,
            uniqueness_score=0.8,
            completeness_score=0.9
        )
        )
        
        # Profile validation of large dataset
        def validate_large_dataset():
            return self.validator.validate(large_dataset)
        
        validation_result, profiling_result = self.profiler.profile_function(validate_large_dataset)
        
        # Verify performance metrics
        self.assertIsInstance(validation_result, ValidationResult)
        self.assertIsInstance(profiling_result, ProfilingResult)
        self.assertGreater(profiling_result.metrics.duration_seconds, 0)
        
        # Check for performance bottlenecks
        bottlenecks = profiling_result.detect_bottlenecks()
        recommendations = profiling_result.generate_recommendations()
        
        self.assertIsInstance(bottlenecks, list)
        self.assertIsInstance(recommendations, list)
    
    def test_quality_assurance_workflow(self):
        """Test quality assurance workflow."""
        # Generate mixed quality data
        mixed_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.LOW),
            GenerationConfig(DataType.TEXT, ContentCategory.CORRUPTED, QualityLevel.CORRUPTED)
        ]
        
        mixed_documents = self.generator.generate_sample_dataset(mixed_configs, save_files=False)
        
        # Create dataset with mixed quality
        documents = []
        for i, sample_doc in enumerate(mixed_documents):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=len(sample_doc.content),
                language=sample_doc.expected_language or "en",
                domain=sample_doc.expected_domain or "general",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.5
            )
            
            document = Document(
                id=f"qa_doc_{i}",
                source_path=f"/test/qa_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        # Calculate overall quality metrics
        quality_scores = [doc.metadata.quality_score for doc in documents]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        qa_dataset = Dataset(
            id="qa_dataset",
            name="Quality Assurance Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(
            overall_score=avg_quality,
            length_score=avg_quality,
            language_score=0.7,
            coherence_score=0.6,
            uniqueness_score=avg_quality,
            completeness_score=0.8
        )
        )
        
        # Run comprehensive validation
        validation_result = self.validator.validate(qa_dataset)
        
        # Run quality benchmarks
        quality_results = self.benchmarks.run_suite("quality", qa_dataset)
        
        # Verify quality issues are detected
        self.assertGreater(len(validation_result.issues), 0)
        
        # Check for quality-related validation issues
        quality_issues = validation_result.get_issues_by_category(ValidationCategory.QUALITY)
        content_issues = validation_result.get_issues_by_category(ValidationCategory.CONTENT)
        
        # Should have some quality or content issues due to mixed quality data
        self.assertGreater(len(quality_issues) + len(content_issues), 0)
        
        # Verify benchmark results
        self.assertGreater(len(quality_results), 0)
        quality_benchmark_result = quality_results[0]
        
        # Should have quality metrics
        quality_metrics = [m for m in quality_benchmark_result.metrics if "quality" in m.name]
        self.assertGreater(len(quality_metrics), 0)
    
    def test_integration_testing_workflow(self):
        """Test integration testing workflow."""
        # Mock pipeline behavior
        def mock_process_files(files):
            # Create mock dataset based on input files
            documents = []
            for i, file_path in enumerate(files):
                # Read file content if it exists
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                except:
                    content = f"Mock content for file {i}"
                
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=len(content),
                    language="en",
                    domain="test",
                    topics=[],
                    entities=[],
                    quality_score=0.8
                )
                
                document = Document(
                    id=f"integration_doc_{i}",
                    source_path=file_path,
                    content=content,
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                documents.append(document)
            
            return Dataset(
                id="integration_dataset",
                name="Integration Test Dataset",
                version="1.0",
                documents=documents,
                metadata={},
                quality_metrics=QualityMetrics(
            overall_score=0.8,
            length_score=0.8,
            language_score=0.95,
            coherence_score=0.7,
            uniqueness_score=0.8,
            completeness_score=0.9
        )
            )
        
        self.mock_pipeline.process_files = mock_process_files
        
        # Run smoke tests
        smoke_execution = self.tester.run_test_suite("smoke")
        
        self.assertIsInstance(smoke_execution, TestExecution)
        self.assertGreater(len(smoke_execution.results), 0)
        
        # Check success rate
        success_rate = smoke_execution.get_success_rate()
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
        
        # Generate test report
        report = self.tester.generate_test_report(smoke_execution, format="text")
        self.assertIsInstance(report, str)
        self.assertIn("Integration Test Report", report)


if __name__ == '__main__':
    unittest.main()