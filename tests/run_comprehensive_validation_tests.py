#!/usr/bin/env python3
"""
Comprehensive validation test runner.

This script runs all validation tests and generates comprehensive reports
on test coverage, performance, and validation effectiveness.
"""

import json
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.qudata.validation import (
    DatasetValidator, QualityBenchmarks, PerformanceProfiler,
    IntegrationTester, SampleDataGenerator
)


class ComprehensiveTestRunner:
    """Comprehensive test runner for validation components."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize test runner.
        
        Args:
            output_dir: Directory for test outputs and reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive validation tests."""
        print("Starting comprehensive validation test suite...")
        self.start_time = datetime.now()
        
        # Run unit tests
        print("\n1. Running unit tests...")
        unit_test_results = self._run_unit_tests()
        self.test_results["unit_tests"] = unit_test_results
        
        # Run component tests
        print("\n2. Running component validation tests...")
        component_results = self._run_component_tests()
        self.test_results["component_tests"] = component_results
        
        # Run integration tests
        print("\n3. Running integration tests...")
        integration_results = self._run_integration_tests()
        self.test_results["integration_tests"] = integration_results
        
        # Run performance tests
        print("\n4. Running performance tests...")
        performance_results = self._run_performance_tests()
        self.test_results["performance_tests"] = performance_results
        
        # Run regression tests
        print("\n5. Running regression tests...")
        regression_results = self._run_regression_tests()
        self.test_results["regression_tests"] = regression_results
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        print("\n6. Generating comprehensive report...")
        self._generate_comprehensive_report()
        
        print(f"\nTest suite completed in {(self.end_time - self.start_time).total_seconds():.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return self.test_results
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests using unittest."""
        test_modules = [
            "tests.unit.test_comprehensive_validation",
            "tests.unit.test_validation_comprehensive_coverage"
        ]
        
        results = {}
        
        for module in test_modules:
            print(f"  Running {module}...")
            
            # Load and run tests
            loader = unittest.TestLoader()
            try:
                suite = loader.loadTestsFromName(module)
                runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
                result = runner.run(suite)
                
                results[module] = {
                    "tests_run": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
                    "failure_details": [str(f[1]) for f in result.failures],
                    "error_details": [str(e[1]) for e in result.errors]
                }
                
                print(f"    {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
                
            except Exception as e:
                results[module] = {
                    "error": str(e),
                    "tests_run": 0,
                    "success_rate": 0
                }
                print(f"    Error loading module: {e}")
        
        return results
    
    def _run_component_tests(self) -> Dict[str, Any]:
        """Run individual component tests."""
        results = {}
        
        # Test DatasetValidator
        print("  Testing DatasetValidator...")
        validator_results = self._test_dataset_validator()
        results["dataset_validator"] = validator_results
        
        # Test QualityBenchmarks
        print("  Testing QualityBenchmarks...")
        benchmark_results = self._test_quality_benchmarks()
        results["quality_benchmarks"] = benchmark_results
        
        # Test PerformanceProfiler
        print("  Testing PerformanceProfiler...")
        profiler_results = self._test_performance_profiler()
        results["performance_profiler"] = profiler_results
        
        # Test SampleDataGenerator
        print("  Testing SampleDataGenerator...")
        generator_results = self._test_sample_data_generator()
        results["sample_data_generator"] = generator_results
        
        return results
    
    def _test_dataset_validator(self) -> Dict[str, Any]:
        """Test DatasetValidator component."""
        try:
            from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
            
            validator = DatasetValidator()
            
            # Create test dataset
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=100,
                language="en",
                author="Test",
                creation_date=datetime.now(),
                domain="test",
                topics=["testing"],
                entities=[],
                quality_score=0.8
            )
            
            document = Document(
                id="test_doc",
                source_path="/test/test.txt",
                content="This is test content for validation testing.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            
            dataset = Dataset(
                id="test_dataset",
                name="Test Dataset",
                version="1.0",
                documents=[document],
                metadata={},
                quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            )
            
            # Run validation
            start_time = time.time()
            result = validator.validate(dataset)
            validation_time = time.time() - start_time
            
            return {
                "success": True,
                "validation_time": validation_time,
                "is_valid": result.is_valid,
                "documents_validated": result.documents_validated,
                "issues_found": len(result.issues),
                "rules_applied": len(result.rules_applied)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_quality_benchmarks(self) -> Dict[str, Any]:
        """Test QualityBenchmarks component."""
        try:
            from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
            
            benchmarks = QualityBenchmarks()
            
            # Create test dataset
            documents = []
            for i in range(5):
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=100,
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
                    content=f"Test content for document {i}. " * 10,
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                documents.append(document)
            
            dataset = Dataset(
                id="benchmark_dataset",
                name="Benchmark Dataset",
                version="1.0",
                documents=documents,
                metadata={},
                quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            )
            
            # Run benchmarks
            start_time = time.time()
            results = benchmarks.run_suite("quality", dataset)
            benchmark_time = time.time() - start_time
            
            return {
                "success": True,
                "benchmark_time": benchmark_time,
                "benchmarks_run": len(results),
                "all_passed": all(r.status.value in ["passed", "warning"] for r in results),
                "metrics_collected": sum(len(r.metrics) for r in results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_performance_profiler(self) -> Dict[str, Any]:
        """Test PerformanceProfiler component."""
        try:
            profiler = PerformanceProfiler()
            
            # Test function profiling
            def test_function():
                time.sleep(0.1)
                return sum(range(1000))
            
            start_time = time.time()
            result, profiling_result = profiler.profile_function(test_function)
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "profiling_time": total_time,
                "function_result": result,
                "snapshots_collected": len(profiling_result.snapshots),
                "metrics_calculated": profiling_result.metrics is not None,
                "bottlenecks_detected": len(profiling_result.bottlenecks),
                "recommendations_generated": len(profiling_result.recommendations)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_sample_data_generator(self) -> Dict[str, Any]:
        """Test SampleDataGenerator component."""
        try:
            from src.qudata.validation import GenerationConfig, DataType, ContentCategory, QualityLevel
            
            generator = SampleDataGenerator()
            
            # Test different data types
            configs = [
                GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
                GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.MEDIUM),
                GenerationConfig(DataType.CSV, ContentCategory.FINANCIAL, QualityLevel.HIGH),
                GenerationConfig(DataType.HTML, ContentCategory.NEWS, QualityLevel.MEDIUM)
            ]
            
            start_time = time.time()
            documents = generator.generate_sample_dataset(configs, save_files=False)
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "generation_time": generation_time,
                "documents_generated": len(documents),
                "data_types_tested": len(set(config.data_type for config in configs)),
                "quality_levels_tested": len(set(config.quality_level for config in configs)),
                "all_have_content": all(len(doc.content) > 0 for doc in documents),
                "all_have_metadata": all(doc.metadata for doc in documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            from unittest.mock import Mock
            
            # Create mock pipeline
            mock_pipeline = Mock()
            mock_dataset = Mock()
            mock_dataset.id = "test_dataset"
            mock_dataset.documents = [Mock() for _ in range(3)]
            mock_pipeline.process_files.return_value = mock_dataset
            
            tester = IntegrationTester(mock_pipeline)
            
            # Run smoke tests
            start_time = time.time()
            execution = tester.run_test_suite("smoke")
            integration_time = time.time() - start_time
            
            return {
                "success": True,
                "integration_time": integration_time,
                "tests_run": execution.summary["total"],
                "tests_passed": execution.summary["passed"],
                "tests_failed": execution.summary["failed"],
                "success_rate": execution.get_success_rate(),
                "suite_name": execution.suite_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
            
            # Create large dataset for performance testing
            documents = []
            for i in range(100):
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=1000,
                    language="en",
                    author=f"Author {i}",
                    creation_date=datetime.now(),
                    domain="test",
                    topics=[f"topic_{i}"],
                    entities=[],
                    quality_score=0.7
                )
                
                document = Document(
                    id=f"perf_doc_{i}",
                    source_path=f"/test/perf_doc_{i}.txt",
                    content=f"Performance test content for document {i}. " * 50,
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                documents.append(document)
            
            large_dataset = Dataset(
                id="performance_dataset",
                name="Performance Dataset",
                version="1.0",
                documents=documents,
                metadata={},
                quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            )
            
            # Test validation performance
            validator = DatasetValidator()
            start_time = time.time()
            validation_result = validator.validate(large_dataset)
            validation_time = time.time() - start_time
            
            # Test benchmark performance
            benchmarks = QualityBenchmarks()
            start_time = time.time()
            benchmark_results = benchmarks.run_suite("quality", large_dataset)
            benchmark_time = time.time() - start_time
            
            return {
                "success": True,
                "dataset_size": len(documents),
                "validation_time": validation_time,
                "validation_throughput": len(documents) / validation_time,
                "benchmark_time": benchmark_time,
                "benchmark_throughput": len(documents) / benchmark_time,
                "validation_passed": validation_result.is_valid,
                "benchmarks_completed": len(benchmark_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests."""
        try:
            from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
            
            benchmarks = QualityBenchmarks()
            
            # Create baseline dataset
            baseline_documents = []
            for i in range(10):
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=100,
                    language="en",
                    author=f"Author {i}",
                    creation_date=datetime.now(),
                    domain="test",
                    topics=[f"topic_{i}"],
                    entities=[],
                    quality_score=0.8
                )
                
                document = Document(
                    id=f"baseline_doc_{i}",
                    source_path=f"/test/baseline_doc_{i}.txt",
                    content=f"Baseline content for document {i}. " * 10,
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                baseline_documents.append(document)
            
            baseline_dataset = Dataset(
                id="baseline_dataset",
                name="Baseline Dataset",
                version="1.0",
                documents=baseline_documents,
                metadata={},
                quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            )
            
            # Set baselines
            benchmarks.set_baselines_from_dataset(baseline_dataset, "quality")
            
            # Create current dataset (slightly different)
            current_documents = []
            for i in range(10):
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=100,
                    language="en",
                    author=f"Author {i}",
                    creation_date=datetime.now(),
                    domain="test",
                    topics=[f"topic_{i}"],
                    entities=[],
                    quality_score=0.75  # Slightly lower quality
                )
                
                document = Document(
                    id=f"current_doc_{i}",
                    source_path=f"/test/current_doc_{i}.txt",
                    content=f"Current content for document {i}. " * 10,
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                current_documents.append(document)
            
            current_dataset = Dataset(
                id="current_dataset",
                name="Current Dataset",
                version="1.0",
                documents=current_documents,
                metadata={},
                quality_metrics=QualityMetrics(0.75, 0.75, 0.75, 0.75, 0.75, 0.75)
            )
            
            # Run regression detection
            current_results = benchmarks.run_suite("quality", current_dataset)
            regressions = benchmarks.detect_regressions(current_results, tolerance=0.1)
            
            return {
                "success": True,
                "baseline_set": True,
                "current_results": len(current_results),
                "regressions_detected": len(regressions),
                "regression_details": regressions
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_comprehensive_report(self) -> None:
        """Generate comprehensive test report."""
        report = {
            "test_execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "results": self.test_results,
            "summary": self._generate_summary()
        }
        
        # Save JSON report
        json_report_path = self.output_dir / "comprehensive_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text report
        text_report_path = self.output_dir / "comprehensive_test_report.txt"
        with open(text_report_path, 'w') as f:
            f.write(self._generate_text_report(report))
        
        print(f"  JSON report: {json_report_path}")
        print(f"  Text report: {text_report_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        summary = {
            "overall_success": True,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "components_tested": 0,
            "performance_metrics": {}
        }
        
        # Unit test summary
        if "unit_tests" in self.test_results:
            for module, results in self.test_results["unit_tests"].items():
                if "tests_run" in results:
                    summary["total_tests"] += results["tests_run"]
                    passed = results["tests_run"] - results.get("failures", 0) - results.get("errors", 0)
                    summary["passed_tests"] += passed
                    summary["failed_tests"] += results.get("failures", 0) + results.get("errors", 0)
        
        # Component test summary
        if "component_tests" in self.test_results:
            summary["components_tested"] = len(self.test_results["component_tests"])
            for component, results in self.test_results["component_tests"].items():
                if not results.get("success", False):
                    summary["overall_success"] = False
        
        # Performance metrics
        if "performance_tests" in self.test_results:
            perf_results = self.test_results["performance_tests"]
            if perf_results.get("success", False):
                summary["performance_metrics"] = {
                    "validation_throughput": perf_results.get("validation_throughput", 0),
                    "benchmark_throughput": perf_results.get("benchmark_throughput", 0),
                    "dataset_size_tested": perf_results.get("dataset_size", 0)
                }
        
        # Calculate success rate
        if summary["total_tests"] > 0:
            summary["success_rate"] = summary["passed_tests"] / summary["total_tests"]
        else:
            summary["success_rate"] = 1.0 if summary["overall_success"] else 0.0
        
        return summary
    
    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate text format report."""
        lines = []
        lines.append("COMPREHENSIVE VALIDATION TEST REPORT")
        lines.append("=" * 50)
        lines.append(f"Execution Time: {report['test_execution']['start_time']} - {report['test_execution']['end_time']}")
        lines.append(f"Duration: {report['test_execution']['duration_seconds']:.2f} seconds")
        lines.append("")
        
        # Summary
        summary = report["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Overall Success: {'PASS' if summary['overall_success'] else 'FAIL'}")
        lines.append(f"Total Tests: {summary['total_tests']}")
        lines.append(f"Passed: {summary['passed_tests']}")
        lines.append(f"Failed: {summary['failed_tests']}")
        lines.append(f"Success Rate: {summary['success_rate']:.1%}")
        lines.append(f"Components Tested: {summary['components_tested']}")
        lines.append("")
        
        # Performance metrics
        if summary["performance_metrics"]:
            lines.append("PERFORMANCE METRICS")
            lines.append("-" * 20)
            perf = summary["performance_metrics"]
            lines.append(f"Validation Throughput: {perf['validation_throughput']:.2f} docs/sec")
            lines.append(f"Benchmark Throughput: {perf['benchmark_throughput']:.2f} docs/sec")
            lines.append(f"Dataset Size Tested: {perf['dataset_size_tested']} documents")
            lines.append("")
        
        # Component results
        if "component_tests" in report["results"]:
            lines.append("COMPONENT TEST RESULTS")
            lines.append("-" * 25)
            for component, results in report["results"]["component_tests"].items():
                status = "PASS" if results.get("success", False) else "FAIL"
                lines.append(f"{component}: {status}")
                if not results.get("success", False) and "error" in results:
                    lines.append(f"  Error: {results['error']}")
            lines.append("")
        
        # Integration results
        if "integration_tests" in report["results"]:
            lines.append("INTEGRATION TEST RESULTS")
            lines.append("-" * 25)
            int_results = report["results"]["integration_tests"]
            if int_results.get("success", False):
                lines.append(f"Tests Run: {int_results['tests_run']}")
                lines.append(f"Success Rate: {int_results['success_rate']:.1%}")
            else:
                lines.append(f"FAILED: {int_results.get('error', 'Unknown error')}")
            lines.append("")
        
        # Regression results
        if "regression_tests" in report["results"]:
            lines.append("REGRESSION TEST RESULTS")
            lines.append("-" * 25)
            reg_results = report["results"]["regression_tests"]
            if reg_results.get("success", False):
                lines.append(f"Regressions Detected: {reg_results['regressions_detected']}")
                if reg_results["regressions_detected"] > 0:
                    lines.append("Regression Details:")
                    for regression in reg_results["regression_details"]:
                        lines.append(f"  - {regression['metric']}: {regression['degradation_percent']:.1f}% degradation")
            else:
                lines.append(f"FAILED: {reg_results.get('error', 'Unknown error')}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive validation tests")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(args.output_dir)
    results = runner.run_all_tests()
    
    # Print summary
    summary = results.get("summary", {})
    if summary:
        print(f"\nOVERALL RESULT: {'PASS' if summary['overall_success'] else 'FAIL'}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"Components Tested: {summary.get('components_tested', 0)}")
    
    # Exit with appropriate code
    sys.exit(0 if summary.get("overall_success", False) else 1)


if __name__ == "__main__":
    main()