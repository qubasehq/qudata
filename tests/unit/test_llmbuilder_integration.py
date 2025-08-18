"""
Unit tests for LLMBuilder integration layer.

Tests all components of the LLMBuilder integration including:
- DatasetExporter for automatic data export
- TrainingTrigger for automated model training
- ModelTracker for performance correlation
- VersionManager for dataset-model versioning
- LLMBuilderConnector for API integration
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.qudata.export.llmbuilder import (
    DatasetExporter, TrainingTrigger, ModelTracker, VersionManager,
    LLMBuilderConnector, ModelConfig, TrainingJob, ModelVersion,
    ExportFormat, JobStatus, ExportResult, CorrelationReport
)
from src.qudata.models import Dataset, Document, DocumentMetadata


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating a model configuration."""
        config = ModelConfig(
            name="test_model",
            architecture="gpt",
            vocab_size=16000,
            embedding_dim=256,
            num_layers=4,
            num_heads=8
        )
        
        self.assertEqual(config.name, "test_model")
        self.assertEqual(config.architecture, "gpt")
        self.assertEqual(config.vocab_size, 16000)
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.num_heads, 8)
    
    def test_model_config_to_dict(self):
        """Test converting model config to dictionary."""
        config = ModelConfig(name="test_model")
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_model")
        self.assertEqual(config_dict["architecture"], "gpt")
        self.assertEqual(config_dict["vocab_size"], 16000)


class TestTrainingJob(unittest.TestCase):
    """Test TrainingJob class."""
    
    def test_training_job_creation(self):
        """Test creating a training job."""
        model_config = ModelConfig(name="test_model")
        job = TrainingJob(
            job_id="test_job_123",
            dataset_path="/path/to/dataset.txt",
            model_config=model_config
        )
        
        self.assertEqual(job.job_id, "test_job_123")
        self.assertEqual(job.dataset_path, "/path/to/dataset.txt")
        self.assertEqual(job.model_config, model_config)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.progress, 0.0)
    
    def test_training_job_to_dict(self):
        """Test converting training job to dictionary."""
        model_config = ModelConfig(name="test_model")
        job = TrainingJob(
            job_id="test_job_123",
            dataset_path="/path/to/dataset.txt",
            model_config=model_config,
            status=JobStatus.RUNNING,
            progress=0.5
        )
        
        job_dict = job.to_dict()
        self.assertIsInstance(job_dict, dict)
        self.assertEqual(job_dict["job_id"], "test_job_123")
        self.assertEqual(job_dict["status"], "running")
        self.assertEqual(job_dict["progress"], 0.5)


class TestDatasetExporter(unittest.TestCase):
    """Test DatasetExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DatasetExporter()
        
        # Create test dataset
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=1000,
            language="en",
            domain="technology",
            topics=["AI", "ML"]
        )
        
        documents = [
            Document(
                id="doc1",
                source_path="test1.txt",
                content="This is test content for document 1.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            ),
            Document(
                id="doc2",
                source_path="test2.txt",
                content="This is test content for document 2.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
        ]
        
        self.test_dataset = Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0",
            documents=documents
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_text_format(self):
        """Test exporting dataset in text format."""
        result = self.exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.TEXT
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.format, ExportFormat.TEXT)
        self.assertEqual(result.file_count, 1)
        self.assertGreater(result.total_size_bytes, 0)
        
        # Check file exists and has content
        output_file = Path(self.temp_dir) / "dataset.txt"
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("This is test content for document 1.", content)
            self.assertIn("This is test content for document 2.", content)
    
    def test_export_jsonl_format(self):
        """Test exporting dataset in JSONL format."""
        result = self.exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.JSONL
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.format, ExportFormat.JSONL)
        
        # Check file exists and has valid JSON lines
        output_file = Path(self.temp_dir) / "dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Two documents
            
            # Validate JSON structure
            for line in lines:
                data = json.loads(line.strip())
                self.assertIn("id", data)
                self.assertIn("text", data)
                self.assertIn("metadata", data)
    
    def test_export_chatml_format(self):
        """Test exporting dataset in ChatML format."""
        result = self.exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.CHATML
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.format, ExportFormat.CHATML)
        
        # Check file exists and has valid ChatML structure
        output_file = Path(self.temp_dir) / "dataset_chatml.jsonl"
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            
            # Validate ChatML structure
            for line in lines:
                data = json.loads(line.strip())
                self.assertIn("messages", data)
                messages = data["messages"]
                self.assertEqual(len(messages), 3)  # system, user, assistant
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[1]["role"], "user")
                self.assertEqual(messages[2]["role"], "assistant")
    
    def test_export_alpaca_format(self):
        """Test exporting dataset in Alpaca format."""
        result = self.exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.ALPACA
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.format, ExportFormat.ALPACA)
        
        # Check file exists and has valid Alpaca structure
        output_file = Path(self.temp_dir) / "dataset_alpaca.jsonl"
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            
            # Validate Alpaca structure
            for line in lines:
                data = json.loads(line.strip())
                self.assertIn("instruction", data)
                self.assertIn("input", data)
                self.assertIn("output", data)
    
    def test_manifest_creation(self):
        """Test that manifest file is created."""
        self.exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.TEXT
        )
        
        manifest_file = Path(self.temp_dir) / "manifest.json"
        self.assertTrue(manifest_file.exists())
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            self.assertEqual(manifest["dataset_id"], "test_dataset")
            self.assertEqual(manifest["dataset_name"], "Test Dataset")
            self.assertEqual(manifest["dataset_version"], "1.0")
            self.assertEqual(manifest["document_count"], 2)
    
    def test_export_validation(self):
        """Test export validation."""
        # Test with validation enabled
        exporter = DatasetExporter({"validate_exports": True})
        result = exporter.export_dataset(
            self.test_dataset,
            self.temp_dir,
            ExportFormat.JSONL
        )
        
        self.assertTrue(result.success)
    
    def test_export_error_handling(self):
        """Test export error handling."""
        # Test with invalid directory that cannot be created (use a path with invalid characters on Windows)
        invalid_path = "/invalid/path/with/\x00/null/character" if os.name != 'nt' else "C:\\invalid\\path\\with\\<>|\\characters"
        result = self.exporter.export_dataset(
            self.test_dataset,
            invalid_path,
            ExportFormat.TEXT
        )
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


class TestTrainingTrigger(unittest.TestCase):
    """Test TrainingTrigger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trigger = TrainingTrigger({
            "default_output_dir": self.temp_dir,
            "auto_start": False  # Don't actually start training in tests
        })
        self.model_config = ModelConfig(name="test_model")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trigger_training(self):
        """Test triggering model training."""
        dataset_path = "/path/to/dataset.txt"
        
        job = self.trigger.trigger_training(dataset_path, self.model_config)
        
        self.assertIsInstance(job, TrainingJob)
        self.assertEqual(job.dataset_path, dataset_path)
        self.assertEqual(job.model_config, self.model_config)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertIsNotNone(job.job_id)
        self.assertIsNotNone(job.output_dir)
    
    def test_training_config_creation(self):
        """Test creation of training configuration file."""
        dataset_path = "/path/to/dataset.txt"
        output_dir = os.path.join(self.temp_dir, "test_output")
        
        config_path = self.trigger._create_training_config(
            self.model_config, dataset_path, output_dir
        )
        
        self.assertTrue(os.path.exists(config_path))
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.assertIn("model", config)
            self.assertIn("training", config)
            self.assertIn("data", config)
            self.assertIn("output", config)
            self.assertEqual(config["data"]["dataset_path"], dataset_path)
    
    def test_job_status_methods(self):
        """Test job status query methods."""
        # These methods return None/False in the current implementation
        # as they would typically integrate with external job management systems
        status = self.trigger.get_job_status("test_job_id")
        self.assertIsNone(status)
        
        cancelled = self.trigger.cancel_job("test_job_id")
        self.assertFalse(cancelled)


class TestModelTracker(unittest.TestCase):
    """Test ModelTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.tracker = ModelTracker({
            "tracking_db_path": self.temp_file.name
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_track_performance(self):
        """Test tracking performance metrics."""
        dataset_id = "test_dataset"
        model_id = "test_model"
        metrics = {"loss": 0.5, "accuracy": 0.85, "perplexity": 2.1}
        
        self.tracker.track_performance(dataset_id, model_id, metrics)
        
        # Verify data was stored
        history = self.tracker.get_performance_history(dataset_id)
        self.assertIn(model_id, history)
        self.assertEqual(history[model_id]["metrics"], metrics)
        self.assertIn("timestamp", history[model_id])
    
    def test_get_performance_history(self):
        """Test getting performance history."""
        dataset_id = "test_dataset"
        
        # Initially empty
        history = self.tracker.get_performance_history(dataset_id)
        self.assertEqual(history, {})
        
        # Add some data
        self.tracker.track_performance(dataset_id, "model1", {"loss": 0.5})
        self.tracker.track_performance(dataset_id, "model2", {"loss": 0.3})
        
        history = self.tracker.get_performance_history(dataset_id)
        self.assertEqual(len(history), 2)
        self.assertIn("model1", history)
        self.assertIn("model2", history)
    
    def test_correlation_report(self):
        """Test generating correlation report."""
        dataset_id = "test_dataset"
        
        # Add some performance data
        self.tracker.track_performance(dataset_id, "model1", {"loss": 0.5})
        self.tracker.track_performance(dataset_id, "model2", {"loss": 0.3})
        self.tracker.track_performance(dataset_id, "model3", {"loss": 0.7})
        
        report = self.tracker.get_correlation_report(dataset_id)
        
        self.assertIsInstance(report, CorrelationReport)
        self.assertEqual(report.dataset_id, dataset_id)
        self.assertIn("average_loss", report.correlations)
        self.assertGreater(len(report.insights), 0)
    
    def test_data_persistence(self):
        """Test that tracking data persists across instances."""
        dataset_id = "test_dataset"
        model_id = "test_model"
        metrics = {"loss": 0.5}
        
        # Track with first instance
        self.tracker.track_performance(dataset_id, model_id, metrics)
        
        # Create new instance with same file
        new_tracker = ModelTracker({
            "tracking_db_path": self.temp_file.name
        })
        
        # Verify data is loaded
        history = new_tracker.get_performance_history(dataset_id)
        self.assertIn(model_id, history)
        self.assertEqual(history[model_id]["metrics"], metrics)


class TestVersionManager(unittest.TestCase):
    """Test VersionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.version_manager = VersionManager({
            "versions_db_path": self.temp_file.name
        })
        self.model_config = ModelConfig(name="test_model")
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_create_model_version(self):
        """Test creating a new model version."""
        dataset_version = "dataset_v1.0"
        
        model_version = self.version_manager.create_model_version(
            dataset_version, self.model_config
        )
        
        self.assertIsInstance(model_version, ModelVersion)
        self.assertEqual(model_version.dataset_version, dataset_version)
        self.assertEqual(model_version.model_config, self.model_config)
        self.assertIsNotNone(model_version.model_id)
        self.assertIsNotNone(model_version.version)
    
    def test_get_model_version(self):
        """Test getting model version by ID."""
        dataset_version = "dataset_v1.0"
        
        # Create a version
        created_version = self.version_manager.create_model_version(
            dataset_version, self.model_config
        )
        
        # Retrieve it
        retrieved_version = self.version_manager.get_model_version(
            created_version.model_id
        )
        
        self.assertIsNotNone(retrieved_version)
        self.assertEqual(retrieved_version.model_id, created_version.model_id)
        self.assertEqual(retrieved_version.dataset_version, dataset_version)
    
    def test_list_versions_for_dataset(self):
        """Test listing versions for a dataset."""
        dataset_version = "dataset_v1.0"
        
        # Create multiple versions for the same dataset
        version1 = self.version_manager.create_model_version(
            dataset_version, self.model_config
        )
        version2 = self.version_manager.create_model_version(
            dataset_version, self.model_config
        )
        
        # Create version for different dataset
        other_version = self.version_manager.create_model_version(
            "dataset_v2.0", self.model_config
        )
        
        # List versions for first dataset
        versions = self.version_manager.list_versions_for_dataset(dataset_version)
        
        self.assertEqual(len(versions), 2)
        model_ids = [v.model_id for v in versions]
        self.assertIn(version1.model_id, model_ids)
        self.assertIn(version2.model_id, model_ids)
        self.assertNotIn(other_version.model_id, model_ids)
    
    def test_version_persistence(self):
        """Test that version data persists across instances."""
        dataset_version = "dataset_v1.0"
        
        # Create version with first instance
        created_version = self.version_manager.create_model_version(
            dataset_version, self.model_config
        )
        
        # Create new instance with same file
        new_manager = VersionManager({
            "versions_db_path": self.temp_file.name
        })
        
        # Verify version is loaded
        retrieved_version = new_manager.get_model_version(created_version.model_id)
        self.assertIsNotNone(retrieved_version)
        self.assertEqual(retrieved_version.model_id, created_version.model_id)


class TestLLMBuilderConnector(unittest.TestCase):
    """Test LLMBuilderConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.connector = LLMBuilderConnector({
            "llmbuilder_root": self.temp_dir,
            "export": {"validate_exports": False},
            "training": {"auto_start": False}
        })
        
        # Create test dataset
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=1000,
            language="en",
            domain="technology",
            topics=["AI"]
        )
        
        documents = [
            Document(
                id="doc1",
                source_path="test1.txt",
                content="Test content for integration.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
        ]
        
        self.test_dataset = Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0",
            documents=documents
        )
        
        self.model_config = ModelConfig(name="test_model")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_to_llmbuilder(self):
        """Test exporting dataset to LLMBuilder."""
        result = self.connector.export_to_llmbuilder(self.test_dataset)
        
        self.assertTrue(result.success)
        self.assertIn("data", result.export_path)
        self.assertIn("clean", result.export_path)
    
    def test_trigger_training(self):
        """Test triggering training through connector."""
        dataset_path = "/path/to/dataset.txt"
        
        job = self.connector.trigger_training(dataset_path, self.model_config)
        
        self.assertIsInstance(job, TrainingJob)
        self.assertEqual(job.dataset_path, dataset_path)
        self.assertEqual(job.model_config, self.model_config)
    
    def test_track_performance(self):
        """Test tracking performance through connector."""
        dataset_id = "test_dataset"
        model_id = "test_model"
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        # Should not raise any exceptions
        self.connector.track_performance(dataset_id, model_id, metrics)
        
        # Verify data was tracked
        history = self.connector.model_tracker.get_performance_history(dataset_id)
        self.assertIn(model_id, history)
    
    def test_create_model_version(self):
        """Test creating model version through connector."""
        dataset_version = "dataset_v1.0"
        
        model_version = self.connector.create_model_version(
            dataset_version, self.model_config
        )
        
        self.assertIsInstance(model_version, ModelVersion)
        self.assertEqual(model_version.dataset_version, dataset_version)
    
    def test_get_correlation_report(self):
        """Test getting correlation report through connector."""
        dataset_id = "test_dataset"
        
        # Add some performance data first
        self.connector.track_performance(dataset_id, "model1", {"loss": 0.5})
        self.connector.track_performance(dataset_id, "model2", {"loss": 0.3})
        
        report = self.connector.get_correlation_report(dataset_id)
        
        self.assertIsInstance(report, CorrelationReport)
        self.assertEqual(report.dataset_id, dataset_id)
    
    def test_setup_llmbuilder_environment(self):
        """Test setting up LLMBuilder environment."""
        success = self.connector.setup_llmbuilder_environment()
        
        self.assertTrue(success)
        
        # Check that directories were created
        expected_dirs = [
            "data",
            "data/raw",
            "data/clean",
            "models",
            "checkpoints",
            "logs"
        ]
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(self.temp_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} not created")
    
    def test_validate_llmbuilder_installation(self):
        """Test validating LLMBuilder installation."""
        # Test when llmbuilder is not available (which is expected in our test environment)
        result = self.connector.validate_llmbuilder_installation()
        self.assertFalse(result)  # We expect this to be False since llmbuilder is not installed
    
    def test_get_integration_status(self):
        """Test getting integration status."""
        # Setup environment first
        self.connector.setup_llmbuilder_environment()
        
        status = self.connector.get_integration_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("llmbuilder_installed", status)
        self.assertIn("environment_setup", status)
        self.assertIn("data_directory", status)
        self.assertIn("config", status)
        
        self.assertTrue(status["environment_setup"])
        self.assertTrue(status["data_directory"])


class TestIntegrationWorkflows(unittest.TestCase):
    """Test complete integration workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.connector = LLMBuilderConnector({
            "llmbuilder_root": self.temp_dir,
            "export": {"validate_exports": False},
            "training": {"auto_start": False}
        })
        
        # Setup environment
        self.connector.setup_llmbuilder_environment()
        
        # Create test dataset
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=1000,
            language="en",
            domain="technology",
            topics=["AI", "ML"]
        )
        
        documents = [
            Document(
                id="doc1",
                source_path="test1.txt",
                content="This is comprehensive test content for machine learning.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            ),
            Document(
                id="doc2",
                source_path="test2.txt",
                content="Advanced artificial intelligence concepts and applications.",
                metadata=metadata,
                processing_timestamp=datetime.now(),
                version="1.0"
            )
        ]
        
        self.test_dataset = Dataset(
            id="test_dataset_integration",
            name="Integration Test Dataset",
            version="1.0",
            documents=documents
        )
        
        self.model_config = ModelConfig(
            name="integration_test_model",
            vocab_size=8000,
            embedding_dim=128,
            num_layers=2,
            num_heads=4
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_integration_workflow(self):
        """Test complete integration workflow from export to tracking."""
        # Step 1: Export dataset
        export_result = self.connector.export_to_llmbuilder(
            self.test_dataset, format=ExportFormat.JSONL
        )
        self.assertTrue(export_result.success)
        
        # Step 2: Create model version
        model_version = self.connector.create_model_version(
            self.test_dataset.version, self.model_config
        )
        self.assertIsNotNone(model_version)
        
        # Step 3: Trigger training (simulated)
        training_job = self.connector.trigger_training(
            export_result.export_path, self.model_config
        )
        self.assertEqual(training_job.status, JobStatus.PENDING)
        
        # Step 4: Track performance metrics
        performance_metrics = {
            "loss": 0.45,
            "accuracy": 0.87,
            "perplexity": 2.3,
            "bleu_score": 0.72
        }
        
        self.connector.track_performance(
            self.test_dataset.id,
            model_version.model_id,
            performance_metrics
        )
        
        # Step 5: Generate correlation report
        correlation_report = self.connector.get_correlation_report(
            self.test_dataset.id
        )
        
        self.assertEqual(correlation_report.dataset_id, self.test_dataset.id)
        self.assertIn("average_loss", correlation_report.correlations)
        
        # Verify all components worked together
        self.assertTrue(os.path.exists(export_result.export_path))
        self.assertIsNotNone(training_job.job_id)
        self.assertGreater(len(correlation_report.insights), 0)
    
    def test_multi_format_export_workflow(self):
        """Test workflow with multiple export formats."""
        formats_to_test = [
            ExportFormat.TEXT,
            ExportFormat.JSONL,
            ExportFormat.CHATML,
            ExportFormat.ALPACA
        ]
        
        export_results = []
        
        for format_type in formats_to_test:
            result = self.connector.export_to_llmbuilder(
                self.test_dataset, format=format_type
            )
            self.assertTrue(result.success, f"Export failed for format {format_type}")
            export_results.append(result)
        
        # Verify all exports succeeded
        self.assertEqual(len(export_results), len(formats_to_test))
        
        # Verify different file sizes (formats should produce different outputs)
        sizes = [result.total_size_bytes for result in export_results]
        self.assertGreater(len(set(sizes)), 1, "All formats produced same size output")
    
    def test_version_tracking_workflow(self):
        """Test version tracking across multiple model iterations."""
        import uuid
        dataset_version = f"test_dataset_v{uuid.uuid4().hex[:8]}"
        
        # Create multiple model versions
        model_configs = [
            ModelConfig(name="small_model", embedding_dim=64, num_layers=2),
            ModelConfig(name="medium_model", embedding_dim=128, num_layers=4),
            ModelConfig(name="large_model", embedding_dim=256, num_layers=6)
        ]
        
        model_versions = []
        for config in model_configs:
            version = self.connector.create_model_version(dataset_version, config)
            model_versions.append(version)
            
            # Simulate different performance metrics
            metrics = {
                "loss": 0.5 - (config.num_layers * 0.05),  # Better with more layers
                "accuracy": 0.7 + (config.embedding_dim / 1000),  # Better with larger embeddings
            }
            
            self.connector.track_performance(
                self.test_dataset.id, version.model_id, metrics
            )
        
        # Verify all versions were created
        self.assertEqual(len(model_versions), 3)
        
        # Verify performance tracking
        correlation_report = self.connector.get_correlation_report(self.test_dataset.id)
        self.assertEqual(len(correlation_report.correlations), 1)  # average_loss
        
        # Verify version listing
        listed_versions = self.connector.version_manager.list_versions_for_dataset(
            dataset_version
        )
        self.assertEqual(len(listed_versions), 3)


if __name__ == '__main__':
    unittest.main()