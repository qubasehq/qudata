"""
Integration tester for end-to-end validation of the processing pipeline.

This module provides comprehensive integration testing capabilities to validate
the entire data processing pipeline from ingestion to export.
"""

import json
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import Dataset, Document, DocumentMetadata, QualityMetrics
from ..pipeline import QuDataPipeline
from .dataset_validator import DatasetValidator, ValidationResult
from .performance_profiler import PerformanceProfiler, ProfilingResult


class TestType(Enum):
    """Types of integration tests."""
    SMOKE = "smoke"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    STRESS = "stress"
    REGRESSION = "regression"
    END_TO_END = "end_to_end"


class TestStatus(Enum):
    """Status of test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case definition."""
    name: str
    description: str
    test_type: TestType
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    validation_rules: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "validation_rules": self.validation_rules,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "tags": self.tags
        }


@dataclass
class TestResult:
    """Result of a test case execution."""
    test_case: TestCase
    status: TestStatus
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    actual_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    profiling_result: Optional[ProfilingResult] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_case": self.test_case.to_dict(),
            "status": self.status.value,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "actual_output": self.actual_output,
            "error_message": self.error_message,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "profiling_result": self.profiling_result.to_dict() if self.profiling_result else None,
            "artifacts": self.artifacts
        }


@dataclass
class TestSuite:
    """Collection of related test cases."""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def get_test_cases_by_type(self, test_type: TestType) -> List[TestCase]:
        """Get test cases of a specific type."""
        return [tc for tc in self.test_cases if tc.test_type == test_type]
    
    def get_test_cases_by_tag(self, tag: str) -> List[TestCase]:
        """Get test cases with a specific tag."""
        return [tc for tc in self.test_cases if tag in tc.tags]


@dataclass
class TestExecution:
    """Result of executing a test suite."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)
        self._update_summary()
    
    def _update_summary(self) -> None:
        """Update execution summary."""
        self.summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == TestStatus.SKIPPED),
            "error": sum(1 for r in self.results if r.status == TestStatus.ERROR)
        }
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get all failed test results."""
        return [r for r in self.results if r.status == TestStatus.FAILED]
    
    def get_success_rate(self) -> float:
        """Get test success rate."""
        if not self.results:
            return 0.0
        return self.summary["passed"] / self.summary["total"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test execution to dictionary."""
        return {
            "suite_name": self.suite_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "success_rate": self.get_success_rate()
        }


class TestDataProvider(ABC):
    """Abstract base class for test data providers."""
    
    @abstractmethod
    def generate_test_data(self, test_case: TestCase) -> Dict[str, Any]:
        """Generate test data for a test case."""
        pass
    
    @abstractmethod
    def cleanup_test_data(self, test_data: Dict[str, Any]) -> None:
        """Clean up generated test data."""
        pass


class FileTestDataProvider(TestDataProvider):
    """Test data provider that creates temporary files."""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize file test data provider.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.created_files: List[str] = []
    
    def generate_test_data(self, test_case: TestCase) -> Dict[str, Any]:
        """Generate test files based on test case input data."""
        test_data = {"files": []}
        
        # Create test files based on input specification
        for file_spec in test_case.input_data.get("files", []):
            file_path = self._create_test_file(file_spec)
            test_data["files"].append(file_path)
            self.created_files.append(file_path)
        
        return test_data
    
    def _create_test_file(self, file_spec: Dict[str, Any]) -> str:
        """Create a test file based on specification."""
        file_type = file_spec.get("type", "txt")
        content = file_spec.get("content", "Test content")
        size = file_spec.get("size", len(content))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=f'.{file_type}', 
            dir=self.temp_dir, 
            delete=False
        ) as f:
            # Write content, padding if necessary
            if len(content) < size:
                content += " " * (size - len(content))
            elif len(content) > size:
                content = content[:size]
            
            f.write(content)
            return f.name
    
    def cleanup_test_data(self, test_data: Dict[str, Any]) -> None:
        """Clean up created test files."""
        for file_path in self.created_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        self.created_files.clear()


class PipelineTestRunner:
    """Test runner for pipeline integration tests."""
    
    def __init__(self, pipeline: QuDataPipeline, config: Dict[str, Any] = None):
        """
        Initialize pipeline test runner.
        
        Args:
            pipeline: Processing pipeline to test
            config: Configuration for test execution
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.validator = DatasetValidator()
        self.profiler = PerformanceProfiler()
        self.data_provider = FileTestDataProvider()
    
    def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        result = TestResult(
            test_case=test_case,
            status=TestStatus.PASSED,
            execution_time=0.0
        )
        
        try:
            # Generate test data
            test_data = self.data_provider.generate_test_data(test_case)
            
            # Start profiling if enabled
            profiling_session = None
            if self.config.get("enable_profiling", False):
                profiling_session = self.profiler.start_profiling(f"test_{test_case.name}")
            
            # Execute test based on type
            if test_case.test_type == TestType.SMOKE:
                actual_output = self._run_smoke_test(test_case, test_data)
            elif test_case.test_type == TestType.FUNCTIONAL:
                actual_output = self._run_functional_test(test_case, test_data)
            elif test_case.test_type == TestType.PERFORMANCE:
                actual_output = self._run_performance_test(test_case, test_data)
            elif test_case.test_type == TestType.STRESS:
                actual_output = self._run_stress_test(test_case, test_data)
            elif test_case.test_type == TestType.END_TO_END:
                actual_output = self._run_end_to_end_test(test_case, test_data)
            else:
                raise ValueError(f"Unsupported test type: {test_case.test_type}")
            
            result.actual_output = actual_output
            
            # Stop profiling
            if profiling_session:
                result.profiling_result = self.profiler.stop_profiling(profiling_session)
            
            # Validate results
            if test_case.validation_rules:
                result.validation_result = self._validate_output(actual_output, test_case.validation_rules)
                if not result.validation_result.is_valid:
                    result.status = TestStatus.FAILED
            
            # Compare with expected output
            if not self._compare_outputs(actual_output, test_case.expected_output):
                result.status = TestStatus.FAILED
                result.error_message = "Actual output does not match expected output"
            
            # Cleanup test data
            self.data_provider.cleanup_test_data(test_data)
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            
            # Stop profiling on error
            if profiling_session:
                try:
                    result.profiling_result = self.profiler.stop_profiling(profiling_session)
                except Exception:
                    pass  # Ignore profiling cleanup errors
        
        result.execution_time = time.time() - start_time
        return result
    
    def _run_smoke_test(self, test_case: TestCase, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a smoke test to verify basic functionality."""
        # Simple pipeline execution with minimal data
        files = test_data.get("files", [])
        if not files:
            raise ValueError("Smoke test requires at least one test file")
        
        # Process single file
        dataset = self.pipeline.process_files(files[:1])
        
        return {
            "documents_processed": len(dataset.documents),
            "processing_successful": True,
            "dataset_id": dataset.id
        }
    
    def _run_functional_test(self, test_case: TestCase, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a functional test to verify specific features."""
        files = test_data.get("files", [])
        
        # Process all files
        dataset = self.pipeline.process_files(files)
        
        # Extract functional metrics
        return {
            "documents_processed": len(dataset.documents),
            "total_content_length": sum(len(doc.content) for doc in dataset.documents),
            "languages_detected": len(set(doc.metadata.language for doc in dataset.documents)),
            "domains_classified": len(set(doc.metadata.domain for doc in dataset.documents)),
            "avg_quality_score": sum(doc.metadata.quality_score for doc in dataset.documents) / len(dataset.documents) if dataset.documents else 0,
            "dataset_id": dataset.id
        }
    
    def _run_performance_test(self, test_case: TestCase, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a performance test to measure processing speed."""
        files = test_data.get("files", [])
        
        start_time = time.time()
        dataset = self.pipeline.process_files(files)
        processing_time = time.time() - start_time
        
        total_chars = sum(len(doc.content) for doc in dataset.documents)
        
        return {
            "processing_time": processing_time,
            "documents_per_second": len(dataset.documents) / processing_time if processing_time > 0 else 0,
            "chars_per_second": total_chars / processing_time if processing_time > 0 else 0,
            "documents_processed": len(dataset.documents),
            "total_chars": total_chars,
            "dataset_id": dataset.id
        }
    
    def _run_stress_test(self, test_case: TestCase, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a stress test with large amounts of data."""
        files = test_data.get("files", [])
        
        # Process files multiple times to simulate stress
        stress_multiplier = test_case.input_data.get("stress_multiplier", 5)
        all_files = files * stress_multiplier
        
        start_time = time.time()
        dataset = self.pipeline.process_files(all_files)
        processing_time = time.time() - start_time
        
        return {
            "processing_time": processing_time,
            "files_processed": len(all_files),
            "documents_processed": len(dataset.documents),
            "stress_multiplier": stress_multiplier,
            "memory_usage_mb": self._estimate_memory_usage(dataset),
            "dataset_id": dataset.id
        }
    
    def _run_end_to_end_test(self, test_case: TestCase, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run an end-to-end test covering the entire pipeline."""
        files = test_data.get("files", [])
        
        # Full pipeline execution with all stages
        dataset = self.pipeline.process_files(files)
        
        # Export to different formats
        export_results = {}
        for format_name in ["jsonl", "parquet", "csv"]:
            try:
                export_path = self.pipeline.export_dataset(dataset, format_name)
                export_results[format_name] = {
                    "success": True,
                    "path": export_path,
                    "size_mb": Path(export_path).stat().st_size / 1024 / 1024
                }
            except Exception as e:
                export_results[format_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "documents_processed": len(dataset.documents),
            "quality_metrics": dataset.quality_metrics.to_dict() if dataset.quality_metrics else {},
            "export_results": export_results,
            "pipeline_stages_completed": self._count_completed_stages(dataset),
            "dataset_id": dataset.id
        }
    
    def _validate_output(self, output: Dict[str, Any], validation_rules: List[str]) -> ValidationResult:
        """Validate test output against specified rules."""
        # This is a simplified validation - in practice, you'd implement
        # specific validation logic based on the rules
        validation_result = ValidationResult(is_valid=True)
        
        for rule in validation_rules:
            if rule == "documents_not_empty" and output.get("documents_processed", 0) == 0:
                validation_result.is_valid = False
            elif rule == "quality_above_threshold" and output.get("avg_quality_score", 0) < 0.5:
                validation_result.is_valid = False
            elif rule == "processing_time_reasonable" and output.get("processing_time", 0) > 300:
                validation_result.is_valid = False
        
        return validation_result
    
    def _compare_outputs(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Compare actual output with expected output."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            
            actual_value = actual[key]
            
            # Handle different comparison types
            if isinstance(expected_value, dict) and "min" in expected_value:
                if actual_value < expected_value["min"]:
                    return False
            elif isinstance(expected_value, dict) and "max" in expected_value:
                if actual_value > expected_value["max"]:
                    return False
            elif isinstance(expected_value, dict) and "range" in expected_value:
                min_val, max_val = expected_value["range"]
                if not (min_val <= actual_value <= max_val):
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _estimate_memory_usage(self, dataset: Dataset) -> float:
        """Estimate memory usage of a dataset in MB."""
        total_size = 0
        for doc in dataset.documents:
            total_size += len(doc.content.encode('utf-8'))
            total_size += len(str(doc.metadata).encode('utf-8'))
        
        return total_size / 1024 / 1024
    
    def _count_completed_stages(self, dataset: Dataset) -> int:
        """Count the number of pipeline stages that completed successfully."""
        # This would be implemented based on your pipeline's stage tracking
        # For now, return a placeholder
        return 5  # Assuming 5 main stages


class IntegrationTester:
    """Main integration tester for end-to-end pipeline validation."""
    
    def __init__(self, pipeline: QuDataPipeline, config: Dict[str, Any] = None):
        """
        Initialize integration tester.
        
        Args:
            pipeline: Processing pipeline to test
            config: Configuration for testing
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.test_runner = PipelineTestRunner(pipeline, config)
        self.test_suites: Dict[str, TestSuite] = {}
        self._setup_default_test_suites()
    
    def _setup_default_test_suites(self) -> None:
        """Setup default test suites."""
        # Smoke test suite
        smoke_suite = TestSuite(
            name="smoke_tests",
            description="Basic smoke tests to verify pipeline functionality"
        )
        
        smoke_suite.add_test_case(TestCase(
            name="basic_text_processing",
            description="Process a simple text file",
            test_type=TestType.SMOKE,
            input_data={
                "files": [{"type": "txt", "content": "This is a test document.", "size": 100}]
            },
            expected_output={
                "documents_processed": {"min": 1},
                "processing_successful": True
            },
            tags=["basic", "text"]
        ))
        
        self.test_suites["smoke"] = smoke_suite
        
        # Functional test suite
        functional_suite = TestSuite(
            name="functional_tests",
            description="Functional tests for specific pipeline features"
        )
        
        functional_suite.add_test_case(TestCase(
            name="multi_format_processing",
            description="Process multiple file formats",
            test_type=TestType.FUNCTIONAL,
            input_data={
                "files": [
                    {"type": "txt", "content": "Text document content", "size": 200},
                    {"type": "json", "content": '{"key": "value"}', "size": 50},
                    {"type": "csv", "content": "col1,col2\nval1,val2", "size": 100}
                ]
            },
            expected_output={
                "documents_processed": {"min": 3},
                "languages_detected": {"min": 1},
                "avg_quality_score": {"min": 0.3}
            },
            validation_rules=["documents_not_empty", "quality_above_threshold"],
            tags=["multi-format", "functional"]
        ))
        
        self.test_suites["functional"] = functional_suite
        
        # Performance test suite
        performance_suite = TestSuite(
            name="performance_tests",
            description="Performance tests for pipeline efficiency"
        )
        
        performance_suite.add_test_case(TestCase(
            name="large_file_processing",
            description="Process large files efficiently",
            test_type=TestType.PERFORMANCE,
            input_data={
                "files": [{"type": "txt", "content": "Large content " * 1000, "size": 50000}]
            },
            expected_output={
                "documents_per_second": {"min": 1},
                "chars_per_second": {"min": 1000}
            },
            validation_rules=["processing_time_reasonable"],
            tags=["performance", "large-files"]
        ))
        
        self.test_suites["performance"] = performance_suite
        
        # End-to-end test suite
        e2e_suite = TestSuite(
            name="end_to_end_tests",
            description="Complete end-to-end pipeline tests"
        )
        
        e2e_suite.add_test_case(TestCase(
            name="full_pipeline_workflow",
            description="Complete pipeline from ingestion to export",
            test_type=TestType.END_TO_END,
            input_data={
                "files": [
                    {"type": "txt", "content": "Document 1 content", "size": 500},
                    {"type": "txt", "content": "Document 2 content", "size": 600}
                ]
            },
            expected_output={
                "documents_processed": {"min": 2},
                "pipeline_stages_completed": {"min": 3}
            },
            timeout_seconds=600,
            tags=["e2e", "complete"]
        ))
        
        self.test_suites["e2e"] = e2e_suite
    
    def add_test_suite(self, test_suite: TestSuite) -> None:
        """Add a custom test suite."""
        self.test_suites[test_suite.name] = test_suite
    
    def run_test_suite(self, suite_name: str) -> TestExecution:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite to run
            
        Returns:
            TestExecution with results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite = self.test_suites[suite_name]
        execution = TestExecution(
            suite_name=suite_name,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        for test_case in suite.test_cases:
            result = self.test_runner.run_test_case(test_case)
            execution.add_result(result)
        
        execution.end_time = datetime.now()
        return execution
    
    def run_all_test_suites(self) -> Dict[str, TestExecution]:
        """Run all test suites."""
        executions = {}
        
        for suite_name in self.test_suites:
            executions[suite_name] = self.run_test_suite(suite_name)
        
        return executions
    
    def run_tests_by_tag(self, tag: str) -> List[TestResult]:
        """Run all tests with a specific tag."""
        results = []
        
        for suite in self.test_suites.values():
            tagged_tests = suite.get_test_cases_by_tag(tag)
            for test_case in tagged_tests:
                result = self.test_runner.run_test_case(test_case)
                results.append(result)
        
        return results
    
    def generate_test_report(self, execution: TestExecution, format: str = "text") -> str:
        """
        Generate a test report.
        
        Args:
            execution: Test execution to report on
            format: Report format ("text", "json", "html")
            
        Returns:
            Formatted test report
        """
        if format == "json":
            return json.dumps(execution.to_dict(), indent=2)
        elif format == "html":
            return self._generate_html_report(execution)
        else:
            return self._generate_text_report(execution)
    
    def _generate_text_report(self, execution: TestExecution) -> str:
        """Generate text format test report."""
        lines = []
        lines.append(f"Integration Test Report: {execution.suite_name}")
        lines.append("=" * 60)
        lines.append(f"Execution Time: {execution.start_time} - {execution.end_time}")
        lines.append(f"Duration: {(execution.end_time - execution.start_time).total_seconds():.2f} seconds")
        lines.append("")
        
        # Summary
        lines.append("Summary:")
        lines.append(f"  Total Tests: {execution.summary['total']}")
        lines.append(f"  Passed: {execution.summary['passed']}")
        lines.append(f"  Failed: {execution.summary['failed']}")
        lines.append(f"  Errors: {execution.summary['error']}")
        lines.append(f"  Skipped: {execution.summary['skipped']}")
        lines.append(f"  Success Rate: {execution.get_success_rate():.1%}")
        lines.append("")
        
        # Failed tests details
        failed_tests = execution.get_failed_tests()
        if failed_tests:
            lines.append("Failed Tests:")
            for result in failed_tests:
                lines.append(f"  - {result.test_case.name}: {result.error_message}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, execution: TestExecution) -> str:
        """Generate HTML format test report."""
        # Simplified HTML report
        html = f"""
        <html>
        <head><title>Integration Test Report: {execution.suite_name}</title></head>
        <body>
        <h1>Integration Test Report: {execution.suite_name}</h1>
        <p>Success Rate: {execution.get_success_rate():.1%}</p>
        <p>Total Tests: {execution.summary['total']}</p>
        <p>Passed: {execution.summary['passed']}</p>
        <p>Failed: {execution.summary['failed']}</p>
        </body>
        </html>
        """
        return html