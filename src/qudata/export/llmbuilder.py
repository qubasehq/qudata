"""
LLMBuilder Integration Layer

This module provides seamless integration with LLMBuilder for automated model training.
It handles dataset export, training triggers, model tracking, and version management.
"""

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..models import Dataset, Document, ProcessingError, ErrorSeverity


class JobStatus(Enum):
    """Status of training jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(Enum):
    """Supported export formats for LLMBuilder."""
    JSONL = "jsonl"
    TEXT = "text"
    CHATML = "chatml"
    ALPACA = "alpaca"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    name: str
    architecture: str = "gpt"
    vocab_size: int = 16000
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    max_seq_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 6e-4
    batch_size: int = 32
    num_epochs: int = 5
    save_every: int = 1000
    eval_every: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "save_every": self.save_every,
            "eval_every": self.eval_every
        }


@dataclass
class TrainingJob:
    """Represents a training job in LLMBuilder."""
    job_id: str
    dataset_path: str
    model_config: ModelConfig
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    progress: float = 0.0
    current_epoch: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "dataset_path": self.dataset_path,
            "model_config": self.model_config.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "output_dir": self.output_dir,
            "error_message": self.error_message
        }


@dataclass
class ModelVersion:
    """Represents a version of a trained model."""
    model_id: str
    version: str
    dataset_version: str
    model_config: ModelConfig
    training_job_id: str
    creation_time: datetime = field(default_factory=datetime.now)
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "dataset_version": self.dataset_version,
            "model_config": self.model_config.to_dict(),
            "training_job_id": self.training_job_id,
            "creation_time": self.creation_time.isoformat(),
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class ExportResult:
    """Result of dataset export operation."""
    success: bool
    export_path: str
    format: ExportFormat
    file_count: int = 0
    total_size_bytes: int = 0
    export_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "export_path": self.export_path,
            "format": self.format.value,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "export_time": self.export_time,
            "error_message": self.error_message
        }


@dataclass
class CorrelationReport:
    """Report showing dataset-model performance correlations."""
    dataset_id: str
    model_versions: List[ModelVersion]
    correlations: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "model_versions": [mv.to_dict() for mv in self.model_versions],
            "correlations": self.correlations,
            "insights": self.insights
        }


class DatasetExporter:
    """Handles automatic export of datasets to LLMBuilder format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize dataset exporter.
        
        Args:
            config: Configuration for export operations
        """
        self.config = config or {}
        self.default_format = ExportFormat(self.config.get("default_format", "text"))
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.validate_exports = self.config.get("validate_exports", True)
    
    def export_dataset(self, dataset: Dataset, target_dir: str, 
                      format: ExportFormat = None) -> ExportResult:
        """
        Export dataset to LLMBuilder format.
        
        Args:
            dataset: Dataset to export
            target_dir: Target directory for export
            format: Export format (defaults to configured format)
            
        Returns:
            ExportResult with export details
        """
        start_time = time.time()
        format = format or self.default_format
        
        try:
            # Create target directory
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == ExportFormat.TEXT:
                result = self._export_text_format(dataset, target_path)
            elif format == ExportFormat.JSONL:
                result = self._export_jsonl_format(dataset, target_path)
            elif format == ExportFormat.CHATML:
                result = self._export_chatml_format(dataset, target_path)
            elif format == ExportFormat.ALPACA:
                result = self._export_alpaca_format(dataset, target_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Create manifest file
            self._create_manifest(dataset, target_path, format)
            
            # Validate export if enabled
            if self.validate_exports:
                self._validate_export(target_path, format)
            
            result.export_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ExportResult(
                success=False,
                export_path=str(target_path),
                format=format,
                export_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _export_text_format(self, dataset: Dataset, target_path: Path) -> ExportResult:
        """Export dataset in plain text format."""
        output_file = target_path / "dataset.txt"
        total_size = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in dataset.documents:
                f.write(doc.content)
                f.write('\n\n')
                total_size += len(doc.content.encode('utf-8'))
        
        return ExportResult(
            success=True,
            export_path=str(output_file),
            format=ExportFormat.TEXT,
            file_count=1,
            total_size_bytes=total_size
        )
    
    def _export_jsonl_format(self, dataset: Dataset, target_path: Path) -> ExportResult:
        """Export dataset in JSONL format."""
        output_file = target_path / "dataset.jsonl"
        total_size = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in dataset.documents:
                record = {
                    "id": doc.id,
                    "text": doc.content,
                    "metadata": doc.metadata.to_dict()
                }
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + '\n')
                total_size += len(line.encode('utf-8'))
        
        return ExportResult(
            success=True,
            export_path=str(output_file),
            format=ExportFormat.JSONL,
            file_count=1,
            total_size_bytes=total_size
        )
    
    def _export_chatml_format(self, dataset: Dataset, target_path: Path) -> ExportResult:
        """Export dataset in ChatML format."""
        output_file = target_path / "dataset_chatml.jsonl"
        total_size = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in dataset.documents:
                # Convert document to ChatML format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Please process this content: {doc.content[:500]}..."},
                    {"role": "assistant", "content": doc.content}
                ]
                record = {"messages": messages}
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + '\n')
                total_size += len(line.encode('utf-8'))
        
        return ExportResult(
            success=True,
            export_path=str(output_file),
            format=ExportFormat.CHATML,
            file_count=1,
            total_size_bytes=total_size
        )
    
    def _export_alpaca_format(self, dataset: Dataset, target_path: Path) -> ExportResult:
        """Export dataset in Alpaca format."""
        output_file = target_path / "dataset_alpaca.jsonl"
        total_size = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in dataset.documents:
                # Convert document to Alpaca format
                record = {
                    "instruction": f"Process and understand the following {doc.metadata.domain} content:",
                    "input": doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content,
                    "output": "Content processed and understood."
                }
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + '\n')
                total_size += len(line.encode('utf-8'))
        
        return ExportResult(
            success=True,
            export_path=str(output_file),
            format=ExportFormat.ALPACA,
            file_count=1,
            total_size_bytes=total_size
        )
    
    def _create_manifest(self, dataset: Dataset, target_path: Path, format: ExportFormat):
        """Create manifest file with dataset information."""
        manifest = {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "dataset_version": dataset.version,
            "export_format": format.value,
            "export_timestamp": datetime.now().isoformat(),
            "document_count": len(dataset.documents),
            "total_words": sum(doc.get_word_count() for doc in dataset.documents),
            "languages": list(set(doc.metadata.language for doc in dataset.documents)),
            "domains": list(set(doc.metadata.domain for doc in dataset.documents)),
            "quality_metrics": dataset.quality_metrics.to_dict() if hasattr(dataset, 'quality_metrics') else {}
        }
        
        manifest_file = target_path / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    def _validate_export(self, target_path: Path, format: ExportFormat):
        """Validate exported files."""
        if format == ExportFormat.TEXT:
            file_path = target_path / "dataset.txt"
        elif format == ExportFormat.JSONL:
            file_path = target_path / "dataset.jsonl"
        elif format == ExportFormat.CHATML:
            file_path = target_path / "dataset_chatml.jsonl"
        elif format == ExportFormat.ALPACA:
            file_path = target_path / "dataset_alpaca.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Export file not found: {file_path}")
        
        # Validate file is not empty
        if file_path.stat().st_size == 0:
            raise ValueError(f"Export file is empty: {file_path}")
        
        # Validate JSON formats
        if format in [ExportFormat.JSONL, ExportFormat.CHATML, ExportFormat.ALPACA]:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")


class TrainingTrigger:
    """Handles automated training pipeline initiation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize training trigger.
        
        Args:
            config: Configuration for training operations
        """
        self.config = config or {}
        self.llmbuilder_path = self.config.get("llmbuilder_path", "llmbuilder")
        self.default_output_dir = self.config.get("default_output_dir", "./models")
        self.auto_start = self.config.get("auto_start", True)
    
    def trigger_training(self, dataset_path: str, model_config: ModelConfig,
                        output_dir: str = None) -> TrainingJob:
        """
        Trigger model training with LLMBuilder.
        
        Args:
            dataset_path: Path to the dataset file
            model_config: Configuration for the model
            output_dir: Output directory for training artifacts
            
        Returns:
            TrainingJob object representing the training job
        """
        job_id = str(uuid4())
        output_dir = output_dir or os.path.join(self.default_output_dir, job_id)
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            dataset_path=dataset_path,
            model_config=model_config,
            output_dir=output_dir
        )
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create LLMBuilder configuration
            config_path = self._create_training_config(model_config, dataset_path, output_dir)
            
            # Start training if auto_start is enabled
            if self.auto_start:
                self._start_training_process(job, config_path)
            
            return job
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            return job
    
    def _create_training_config(self, model_config: ModelConfig, 
                              dataset_path: str, output_dir: str) -> str:
        """Create LLMBuilder configuration file."""
        config = {
            "model": {
                "vocab_size": model_config.vocab_size,
                "embedding_dim": model_config.embedding_dim,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "max_seq_length": model_config.max_seq_length,
                "dropout": model_config.dropout
            },
            "training": {
                "learning_rate": model_config.learning_rate,
                "batch_size": model_config.batch_size,
                "num_epochs": model_config.num_epochs,
                "save_every": model_config.save_every,
                "eval_every": model_config.eval_every
            },
            "data": {
                "dataset_path": dataset_path,
                "block_size": model_config.max_seq_length
            },
            "output": {
                "output_dir": output_dir,
                "checkpoint_dir": os.path.join(output_dir, "checkpoints"),
                "tokenizer_dir": os.path.join(output_dir, "tokenizer")
            }
        }
        
        config_path = os.path.join(output_dir, "training_config.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _start_training_process(self, job: TrainingJob, config_path: str):
        """Start the training process."""
        job.status = JobStatus.RUNNING
        job.start_time = datetime.now()
        
        # This would typically start a subprocess or submit to a job queue
        # For now, we'll simulate the process
        print(f"Starting training job {job.job_id} with config {config_path}")
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job."""
        # This would typically query a job database or process manager
        # For now, return None (not implemented)
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        # This would typically send a cancellation signal to the process
        # For now, return False (not implemented)
        return False


class ModelTracker:
    """Tracks dataset-to-model performance correlations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model tracker.
        
        Args:
            config: Configuration for tracking operations
        """
        self.config = config or {}
        self.tracking_db_path = self.config.get("tracking_db_path", "./model_tracking.json")
        self.tracking_data = self._load_tracking_data()
    
    def track_performance(self, dataset_id: str, model_id: str, 
                         metrics: Dict[str, float]) -> None:
        """
        Track performance metrics for a dataset-model pair.
        
        Args:
            dataset_id: ID of the dataset used for training
            model_id: ID of the trained model
            metrics: Performance metrics dictionary
        """
        if dataset_id not in self.tracking_data:
            self.tracking_data[dataset_id] = {}
        
        self.tracking_data[dataset_id][model_id] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_tracking_data()
    
    def get_performance_history(self, dataset_id: str) -> Dict[str, Any]:
        """Get performance history for a dataset."""
        return self.tracking_data.get(dataset_id, {})
    
    def get_correlation_report(self, dataset_id: str) -> CorrelationReport:
        """
        Generate correlation report for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            CorrelationReport with analysis
        """
        history = self.get_performance_history(dataset_id)
        
        # Calculate correlations (simplified implementation)
        correlations = {}
        insights = []
        
        if len(history) > 1:
            # Example correlation analysis
            losses = [data["metrics"].get("loss", 0) for data in history.values()]
            if losses:
                avg_loss = sum(losses) / len(losses)
                correlations["average_loss"] = avg_loss
                insights.append(f"Average loss across {len(losses)} models: {avg_loss:.4f}")
        
        return CorrelationReport(
            dataset_id=dataset_id,
            model_versions=[],  # Would be populated from actual model data
            correlations=correlations,
            insights=insights
        )
    
    def _load_tracking_data(self) -> Dict[str, Any]:
        """Load tracking data from storage."""
        if os.path.exists(self.tracking_db_path):
            try:
                with open(self.tracking_db_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_tracking_data(self):
        """Save tracking data to storage."""
        with open(self.tracking_db_path, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)


class VersionManager:
    """Manages dataset-model versioning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize version manager.
        
        Args:
            config: Configuration for version management
        """
        self.config = config or {}
        self.versions_db_path = self.config.get("versions_db_path", "./model_versions.json")
        self.versions_data = self._load_versions_data()
    
    def create_model_version(self, dataset_version: str, 
                           model_config: ModelConfig) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            dataset_version: Version of the dataset used
            model_config: Configuration of the model
            
        Returns:
            ModelVersion object
        """
        model_id = f"model_{uuid4().hex[:8]}"
        version = f"v{len(self.versions_data) + 1}"
        
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            dataset_version=dataset_version,
            model_config=model_config,
            training_job_id=""  # Would be set by training trigger
        )
        
        self.versions_data[model_id] = model_version.to_dict()
        self._save_versions_data()
        
        return model_version
    
    def get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get model version by ID."""
        data = self.versions_data.get(model_id)
        if data:
            # Convert back to ModelVersion object (simplified)
            return ModelVersion(
                model_id=data["model_id"],
                version=data["version"],
                dataset_version=data["dataset_version"],
                model_config=ModelConfig(**data["model_config"]),
                training_job_id=data["training_job_id"]
            )
        return None
    
    def list_versions_for_dataset(self, dataset_version: str) -> List[ModelVersion]:
        """List all model versions for a dataset version."""
        versions = []
        for data in self.versions_data.values():
            if data["dataset_version"] == dataset_version:
                versions.append(ModelVersion(
                    model_id=data["model_id"],
                    version=data["version"],
                    dataset_version=data["dataset_version"],
                    model_config=ModelConfig(**data["model_config"]),
                    training_job_id=data["training_job_id"]
                ))
        return versions
    
    def _load_versions_data(self) -> Dict[str, Any]:
        """Load versions data from storage."""
        if os.path.exists(self.versions_db_path):
            try:
                with open(self.versions_db_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_versions_data(self):
        """Save versions data to storage."""
        with open(self.versions_db_path, 'w') as f:
            json.dump(self.versions_data, f, indent=2)


class LLMBuilderConnector:
    """Main connector for LLMBuilder integration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLMBuilder connector.
        
        Args:
            config: Configuration for LLMBuilder integration
        """
        self.config = config or {}
        self.llmbuilder_root = self.config.get("llmbuilder_root", "./llmbuilder")
        self.data_dir = os.path.join(self.llmbuilder_root, "data", "clean")
        
        # Initialize components
        self.exporter = DatasetExporter(self.config.get("export", {}))
        self.training_trigger = TrainingTrigger(self.config.get("training", {}))
        self.model_tracker = ModelTracker(self.config.get("tracking", {}))
        self.version_manager = VersionManager(self.config.get("versioning", {}))
    
    def export_to_llmbuilder(self, dataset: Dataset, 
                           target_dir: str = None, format: ExportFormat = None) -> ExportResult:
        """
        Export dataset to LLMBuilder data directory.
        
        Args:
            dataset: Dataset to export
            target_dir: Target directory (defaults to LLMBuilder data/clean)
            
        Returns:
            ExportResult with export details
        """
        target_dir = target_dir or self.data_dir
        return self.exporter.export_dataset(dataset, target_dir, format)
    
    def trigger_training(self, dataset_path: str, 
                        model_config: ModelConfig) -> TrainingJob:
        """
        Trigger model training.
        
        Args:
            dataset_path: Path to dataset file
            model_config: Model configuration
            
        Returns:
            TrainingJob representing the training process
        """
        return self.training_trigger.trigger_training(dataset_path, model_config)
    
    def track_performance(self, dataset_id: str, model_id: str, 
                         metrics: Dict[str, float]) -> None:
        """Track model performance metrics."""
        self.model_tracker.track_performance(dataset_id, model_id, metrics)
    
    def create_model_version(self, dataset_version: str, 
                           model_config: ModelConfig) -> ModelVersion:
        """Create new model version."""
        return self.version_manager.create_model_version(dataset_version, model_config)
    
    def get_correlation_report(self, dataset_id: str) -> CorrelationReport:
        """Get dataset-model correlation report."""
        return self.model_tracker.get_correlation_report(dataset_id)
    
    def setup_llmbuilder_environment(self) -> bool:
        """
        Set up LLMBuilder directory structure.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Create directory structure
            directories = [
                self.llmbuilder_root,
                os.path.join(self.llmbuilder_root, "data"),
                os.path.join(self.llmbuilder_root, "data", "raw"),
                os.path.join(self.llmbuilder_root, "data", "clean"),
                os.path.join(self.llmbuilder_root, "models"),
                os.path.join(self.llmbuilder_root, "checkpoints"),
                os.path.join(self.llmbuilder_root, "logs")
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Failed to setup LLMBuilder environment: {e}")
            return False
    
    def validate_llmbuilder_installation(self) -> bool:
        """
        Validate that LLMBuilder is properly installed.
        
        Returns:
            True if LLMBuilder is available, False otherwise
        """
        try:
            # Try to import llmbuilder
            import llmbuilder
            return True
        except ImportError:
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of LLMBuilder integration.
        
        Returns:
            Dictionary with integration status information
        """
        return {
            "llmbuilder_installed": self.validate_llmbuilder_installation(),
            "environment_setup": os.path.exists(self.llmbuilder_root),
            "data_directory": os.path.exists(self.data_dir),
            "config": self.config
        }