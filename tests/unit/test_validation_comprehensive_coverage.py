"""
Additional comprehensive test coverage for validation components.

This module provides additional test coverage for edge cases, error conditions,
and integration scenarios not covered in the main test file.
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.qudata.models import Dataset, Document, DocumentMetadata, QualityMetrics
from src.qudata.validation import (
    DatasetValidator, ValidationResult, ValidationIssue, ValidationSeverity,
    QualityBenchmarks, PerformanceProfiler, IntegrationTester,
    SampleDataGenerator, GenerationConfig, DataType, ContentCategory, QualityLevel
)


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in validation components."""
    
    def test_empty_dataset_validation(self):
        """Test validation of completely empty dataset."""
        validator = DatasetValidator()
        
        empty_dataset = Dataset(
            id="",
            name="",
            version="",
            documents=[],
            metadata={},
            quality_metrics=None
        )
        
        result = validator.validate(empty_dataset)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.get_critical_issues()), 0)
        self.assertGreater(len(result.get_error_issues()), 0)
    
    def test_malformed_document_validation(self):
        """Test validation of malformed documents."""
        validator = DatasetValidator()
        
        # Document with None values
        malformed_metadata = DocumentMetadata(
            file_type=None,
            size_bytes=-1,
            language=None,
            author=None,
            creation_date=None,
            domain=None,
            topics=[],
            entities=[],
            quality_score=-0.5  # Invalid negative score
        )
        
        malformed_doc = Document(
            id=None,
            source_path=None,
            content=None,
            metadata=malformed_metadata,
            processing_timestamp=None,
            version=None
        )
        
        malformed_dataset = Dataset(
            id="malformed_test",
            name="Malformed Test",
            version="1.0",
            documents=[malformed_doc],
            metadata={},
            quality_metrics=QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        
        result = validator.validate(malformed_dataset)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)
    
    def test_validation_rule_exception_handling(self):
        """Test handling of exceptions in validation rules."""
        validator = DatasetValidator()
        
        # Create a mock rule that raises an exception
        mock_rule = Mock()
        mock_rule.name = "failing_rule"
        mock_rule.validate.side_effect = Exception("Rule failed")
        
        validator.add_rule(mock_rule)
        
        sample_dataset = Dataset(
            id="test",
            name="Test",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        
        result = validator.validate(sample_dataset)
        
        # Should handle the exception gracefully
        self.assertFalse(result.is_valid)
        critical_issues = result.get_critical_issues()
        self.assertGreater(len(critical_issues), 0)
        
        # Check that the exception was recorded
        exception_issues = [i for i in critical_issues if "failed" in i.message.lower()]
        self.assertGreater(len(exception_issues), 0)
    
    def test_benchmark_with_empty_dataset(self):
        """Test benchmarks with empty dataset."""
        benchmarks = QualityBenchmarks()
        
        empty_dataset = Dataset(
            id="empty",
            name="Empty",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        
        results = benchmarks.run_suite("quality", empty_dataset)
        
        self.assertGreater(len(results), 0)
        # Should handle empty dataset gracefully
        for result in results:
            self.assertIsNotNone(result.status)
    
    def test_profiler_with_very_short_operations(self):
        """Test profiler with operations that complete very quickly."""
        profiler = PerformanceProfiler()
        
        def quick_operation():
            return "done"
        
        result, profiling_result = profiler.profile_function(quick_operation)
        
        self.assertEqual(result, "done")
        self.assertIsNotNone(profiling_result)
        self.assertGreaterEqual(profiling_result.metrics.duration_seconds, 0)
    
    def test_profiler_with_failing_function(self):
        """Test profiler with function that raises exception."""
        profiler = PerformanceProfiler()
        
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            profiler.profile_function(failing_function)
    
    def test_sample_generator_with_extreme_sizes(self):
        """Test sample data generator with extreme size requirements."""
        generator = SampleDataGenerator()
        
        # Very small size
        small_config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.GENERAL,
            quality_level=QualityLevel.HIGH,
            size_range=(1, 5)
        )
        
        small_doc = generator.generate_sample_document(small_config)
        self.assertLessEqual(len(small_doc.content), 5)
        
        # Very large size
        large_config = GenerationConfig(
            data_type=DataType.TEXT,
            content_category=ContentCategory.GENERAL,
            quality_level=QualityLevel.HIGH,
            size_range=(10000, 15000)
        )
        
        large_doc = generator.generate_sample_document(large_config)
        self.assertGreaterEqual(len(large_doc.content), 10000)
        self.assertLessEqual(len(large_doc.content), 15000)
    
    def test_integration_tester_with_mock_pipeline_failure(self):
        """Test integration tester when pipeline fails."""
        mock_pipeline = Mock()
        mock_pipeline.process_files.side_effect = Exception("Pipeline failed")
        
        tester = IntegrationTester(mock_pipeline)
        
        # Run a simple test that should fail
        smoke_suite = tester.test_suites["smoke"]
        if smoke_suite.test_cases:
            test_case = smoke_suite.test_cases[0]
            result = tester.test_runner.run_test_case(test_case)
            
            self.assertEqual(result.status.value, "error")
            self.assertIsNotNone(result.error_message)


class TestValidationPerformance(unittest.TestCase):
    """Test performance characteristics of validation components."""
    
    def test_validator_performance_with_large_dataset(self):
        """Test validator performance with large dataset."""
        validator = DatasetValidator()
        
        # Create large dataset
        documents = []
        for i in range(1000):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=100,
                language="en",
                author=f"Author {i}",
                creation_date=datetime.now(),
                domain="test",
                topics=[f"topic_{i}"],
                entities=[],
                quality_score=0.7
            )
            
            document = Document(
                id=f"doc_{i}",
                source_path=f"/test/doc_{i}.txt",
                content=f"Test content for document {i}",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        large_dataset = Dataset(
            id="large_dataset",
            name="Large Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
        )
        
        start_time = time.time()
        result = validator.validate(large_dataset)
        validation_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 10 seconds)
        self.assertLess(validation_time, 10.0)
        self.assertEqual(result.documents_validated, 1000)
        self.assertGreater(result.validation_duration, 0)
    
    def test_benchmark_performance_scaling(self):
        """Test benchmark performance with datasets of different sizes."""
        benchmarks = QualityBenchmarks()
        
        # Test with different dataset sizes
        sizes = [10, 100, 500]
        execution_times = []
        
        for size in sizes:
            documents = []
            for i in range(size):
                metadata = DocumentMetadata(
                    file_type="txt",
                    size_bytes=100,
                    language="en",
                    author=f"Author {i}",
                    creation_date=datetime.now(),
                    domain="test",
                    topics=[f"topic_{i}"],
                    entities=[],
                    quality_score=0.7
                )
                
                document = Document(
                    id=f"doc_{i}",
                    source_path=f"/test/doc_{i}.txt",
                    content=f"Test content for document {i}",
                    metadata=metadata,
                    processing_timestamp=datetime.now(),
                    version="1.0"
                )
                documents.append(document)
            
            dataset = Dataset(
                id=f"dataset_{size}",
                name=f"Dataset {size}",
                version="1.0",
                documents=documents,
                metadata={},
                quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            )
            
            start_time = time.time()
            results = benchmarks.run_suite("quality", dataset)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            self.assertGreater(len(results), 0)
        
        # Execution time should scale reasonably (not exponentially)
        # Allow for some variance in timing
        self.assertLess(execution_times[-1] / execution_times[0], 100)  # Should not be 100x slower


class TestValidationReporting(unittest.TestCase):
    """Test validation reporting and output generation."""
    
    def test_validation_report_formats(self):
        """Test different validation report formats."""
        validator = DatasetValidator()
        
        sample_dataset = Dataset(
            id="test",
            name="Test",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        
        result = validator.validate(sample_dataset)
        
        # Test text report
        text_report = validator.generate_report(result, "text")
        self.assertIsInstance(text_report, str)
        self.assertIn("Dataset Validation Report", text_report)
        
        # Test JSON report
        json_report = validator.generate_report(result, "json")
        self.assertIsInstance(json_report, str)
        
        # Should be valid JSON
        try:
            parsed_json = json.loads(json_report)
            self.assertIn("is_valid", parsed_json)
        except json.JSONDecodeError:
            self.fail("JSON report is not valid JSON")
        
        # Test HTML report
        html_report = validator.generate_report(result, "html")
        self.assertIsInstance(html_report, str)
        self.assertIn("<html>", html_report)
    
    def test_benchmark_result_serialization(self):
        """Test benchmark result serialization."""
        benchmarks = QualityBenchmarks()
        
        sample_dataset = Dataset(
            id="test",
            name="Test",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        
        results = benchmarks.run_suite("quality", sample_dataset)
        
        # Test serialization to dictionary
        for result in results:
            result_dict = result.to_dict()
            self.assertIsInstance(result_dict, dict)
            self.assertIn("benchmark_name", result_dict)
            self.assertIn("status", result_dict)
            self.assertIn("metrics", result_dict)
        
        # Test saving results to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            benchmarks.save_results(results, temp_path)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(Path(temp_path).exists())
            
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("results", saved_data)
            self.assertIn("timestamp", saved_data)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_profiling_result_export(self):
        """Test profiling result export and analysis."""
        profiler = PerformanceProfiler()
        
        def test_function():
            time.sleep(0.1)
            return "done"
        
        result, profiling_result = profiler.profile_function(test_function)
        
        # Test dictionary conversion
        result_dict = profiling_result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn("session_id", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertIn("bottlenecks", result_dict)
        self.assertIn("recommendations", result_dict)
        
        # Test JSON serialization
        json_str = json.dumps(result_dict, default=str)
        self.assertIsInstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["session_id"], profiling_result.session_id)


class TestValidationConfiguration(unittest.TestCase):
    """Test validation configuration and customization."""
    
    def test_custom_validation_rules(self):
        """Test adding custom validation rules."""
        validator = DatasetValidator()
        
        # Create custom rule
        class CustomRule:
            def __init__(self):
                self.name = "custom_rule"
                self.description = "Custom validation rule"
                self.category = "custom"
            
            def validate(self, dataset):
                issues = []
                if len(dataset.documents) > 100:
                    from src.qudata.validation.dataset_validator import ValidationIssue, ValidationSeverity, ValidationCategory
                    issues.append(ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=ValidationSeverity.WARNING,
                        message="Dataset has more than 100 documents"
                    ))
                return issues
        
        custom_rule = CustomRule()
        validator.add_rule(custom_rule)
        
        # Test with large dataset
        documents = []
        for i in range(150):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=100,
                language="en",
                author=f"Author {i}",
                creation_date=datetime.now(),
                domain="test",
                topics=[],
                entities=[],
                quality_score=0.7
            )
            
            document = Document(
                id=f"doc_{i}",
                source_path=f"/test/doc_{i}.txt",
                content=f"Content {i}",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        large_dataset = Dataset(
            id="large",
            name="Large",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
        )
        
        result = validator.validate(large_dataset)
        
        # Should include custom rule in applied rules
        self.assertIn("custom_rule", result.rules_applied)
        
        # Should have warning from custom rule
        warnings = result.get_issues_by_severity(ValidationSeverity.WARNING)
        custom_warnings = [w for w in warnings if "100 documents" in w.message]
        self.assertGreater(len(custom_warnings), 0)
    
    def test_validator_configuration(self):
        """Test validator configuration options."""
        config = {
            "min_content_length": 50,
            "max_content_length": 5000,
            "min_quality_score": 0.6
        }
        
        validator = DatasetValidator(config)
        
        # Create dataset that violates configured thresholds
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=10,
            language="en",
            author="Test",
            creation_date=datetime.now(),
            domain="test",
            topics=[],
            entities=[],
            quality_score=0.4  # Below configured threshold
        )
        
        document = Document(
            id="test_doc",
            source_path="/test/test.txt",
            content="Short",  # Below configured minimum length
            metadata=metadata,
            processing_timestamp=datetime.now(),
            version="1.0"
        )
        
        test_dataset = Dataset(
            id="test",
            name="Test",
            version="1.0",
            documents=[document],
            metadata={},
            quality_metrics=QualityMetrics(0.4, 0.4, 0.4, 0.4, 0.4, 0.4)
        )
        
        result = validator.validate(test_dataset)
        
        # Should have issues based on configured thresholds
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)
    
    def test_benchmark_configuration(self):
        """Test benchmark configuration options."""
        config = {
            "quality": {
                "tolerance": 0.15
            },
            "performance": {
                "timeout": 60
            }
        }
        
        benchmarks = QualityBenchmarks(config)
        
        sample_dataset = Dataset(
            id="test",
            name="Test",
            version="1.0",
            documents=[],
            metadata={},
            quality_metrics=QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        
        results = benchmarks.run_suite("quality", sample_dataset)
        
        # Should use configured settings
        self.assertGreater(len(results), 0)
        for result in results:
            for metric in result.metrics:
                if metric.baseline is not None:
                    self.assertEqual(metric.tolerance, 0.15)


class TestValidationIntegrationScenarios(unittest.TestCase):
    """Test complex integration scenarios."""
    
    def test_multi_stage_validation_pipeline(self):
        """Test multi-stage validation pipeline."""
        # Stage 1: Dataset validation
        validator = DatasetValidator()
        
        # Stage 2: Quality benchmarking
        benchmarks = QualityBenchmarks()
        
        # Stage 3: Performance profiling
        profiler = PerformanceProfiler()
        
        # Create test dataset
        documents = []
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
                quality_score=0.7 + i * 0.02
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
        
        test_dataset = Dataset(
            id="multi_stage_test",
            name="Multi-stage Test",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
        )
        
        # Execute multi-stage pipeline
        session_id = profiler.start_profiling("multi_stage_validation")
        
        # Stage 1: Validation
        validation_result = validator.validate(test_dataset)
        
        # Stage 2: Benchmarking
        benchmark_results = benchmarks.run_suite("quality", test_dataset)
        
        # Stage 3: Complete profiling
        profiling_result = profiler.stop_profiling(session_id)
        
        # Verify all stages completed successfully
        self.assertTrue(validation_result.is_valid)
        self.assertGreater(len(benchmark_results), 0)
        self.assertIsNotNone(profiling_result.metrics)
        
        # Verify integration between stages
        self.assertEqual(validation_result.documents_validated, len(test_dataset.documents))
        
        for benchmark_result in benchmark_results:
            self.assertEqual(benchmark_result.dataset_id, test_dataset.id)
        
        self.assertGreater(profiling_result.metrics.duration_seconds, 0)
    
    def test_validation_with_sample_data(self):
        """Test validation using generated sample data."""
        # Generate sample data
        generator = SampleDataGenerator()
        
        configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.TEXT, ContentCategory.CORRUPTED, QualityLevel.CORRUPTED)
        ]
        
        sample_documents = generator.generate_sample_dataset(configs, save_files=False)
        
        # Convert to dataset format
        documents = []
        for i, sample_doc in enumerate(sample_documents):
            metadata = DocumentMetadata(
                file_type="txt",
                size_bytes=len(sample_doc.content.encode('utf-8')),
                language=sample_doc.expected_language or "en",
                author="Generated",
                creation_date=datetime.now(),
                domain=sample_doc.expected_domain or "test",
                topics=[],
                entities=[],
                quality_score=sample_doc.expected_quality_score or 0.5
            )
            
            document = Document(
                id=f"generated_doc_{i}",
                source_path=f"/generated/doc_{i}.txt",
                content=sample_doc.content,
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
            documents.append(document)
        
        generated_dataset = Dataset(
            id="generated_dataset",
            name="Generated Dataset",
            version="1.0",
            documents=documents,
            metadata={},
            quality_metrics=QualityMetrics(0.6, 0.6, 0.6, 0.6, 0.6, 0.6)
        )
        
        # Validate generated dataset
        validator = DatasetValidator()
        result = validator.validate(generated_dataset)
        
        # Should handle various quality levels appropriately
        self.assertIsNotNone(result)
        self.assertEqual(result.documents_validated, len(sample_documents))
        
        # Corrupted content should generate validation issues
        if any(config.quality_level == QualityLevel.CORRUPTED for config in configs):
            self.assertGreater(len(result.issues), 0)


if __name__ == '__main__':
    unittest.main()