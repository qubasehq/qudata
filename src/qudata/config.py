"""
Configuration management for QuData.

This module provides configuration loading, validation, and management
for the data processing pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with common validation."""
    
    model_config = ConfigDict(
        extra="forbid",  # Don't allow extra fields
        validate_assignment=True
    )


@dataclass
class PathConfig:
    """File path configuration."""
    raw_data: str = "data/raw"
    staging: str = "data/staging"
    processed: str = "data/processed"
    exports: str = "data/exports"


class IngestConfig(BaseConfig):
    """Configuration for data ingestion stage."""
    enabled: bool = True
    file_types: List[str] = Field(default=["txt", "md", "html", "pdf", "docx", "epub"])
    max_file_size: str = Field(default="100MB")
    parallel_workers: int = Field(default=4, ge=1, le=16)
    
    @field_validator('max_file_size')
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size format."""
        if not isinstance(v, str):
            raise ValueError("File size must be a string")
        
        # Convert size string to bytes for validation
        size_units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        
        try:
            if v[-2:].upper() in size_units:
                size_value = float(v[:-2])
                unit = v[-2:].upper()
            elif v[-1:].upper() == 'B':
                size_value = float(v[:-1])
                unit = 'B'
            else:
                raise ValueError("Invalid size format")
                
            if size_value <= 0:
                raise ValueError("File size must be positive")
                
            return v
        except (ValueError, IndexError):
            raise ValueError("File size must be in format like '100MB', '1GB', etc.")


class CleanConfig(BaseConfig):
    """Configuration for data cleaning stage."""
    enabled: bool = True
    normalize_unicode: bool = True
    remove_boilerplate: bool = True
    deduplicate: bool = True
    min_length: int = Field(default=100, ge=1)
    max_length: int = Field(default=100000, ge=1)
    
    @field_validator('max_length')
    @classmethod
    def validate_length_range(cls, v, info):
        """Ensure max_length > min_length."""
        if info.data and 'min_length' in info.data and v <= info.data['min_length']:
            raise ValueError("max_length must be greater than min_length")
        return v


class AnnotateConfig(BaseConfig):
    """Configuration for annotation stage."""
    enabled: bool = True
    taxonomy: bool = True
    ner: bool = True
    topics: bool = Field(default=False)
    safety: bool = Field(default=False)


class ScoreConfig(BaseConfig):
    """Configuration for quality scoring stage."""
    enabled: bool = True
    min_quality_score: float = Field(default=0.7, ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(default={
        "length": 0.2,
        "language": 0.2,
        "coherence": 0.3,
        "uniqueness": 0.2,
        "readability": 0.1
    })
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class PackConfig(BaseConfig):
    """Configuration for dataset packing stage."""
    enabled: bool = True
    formats: List[str] = Field(default=["jsonl", "chatml", "plain"])
    max_records_per_file: int = Field(default=10000, ge=1)
    
    @field_validator('formats')
    @classmethod
    def validate_formats(cls, v):
        """Validate supported formats."""
        supported = {"jsonl", "chatml", "plain", "alpaca"}
        invalid = set(v) - supported
        if invalid:
            raise ValueError(f"Unsupported formats: {invalid}. Supported: {supported}")
        return v


class ExportConfig(BaseConfig):
    """Configuration for export stage."""
    enabled: bool = True
    llmbuilder_path: Optional[str] = Field(default="../llmbuilder/data/cleaned")
    create_manifest: bool = True
    compress_output: bool = Field(default=False)


class PipelineConfig(BaseConfig):
    """Main pipeline configuration."""
    name: str = "qudata"
    version: str = "1.0.0"
    
    # Path configurations (optional)
    paths: Optional[PathConfig] = None
    
    # Stage configurations
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    annotate: AnnotateConfig = Field(default_factory=AnnotateConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    pack: PackConfig = Field(default_factory=PackConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    
    # Global settings
    parallel_processing: bool = True
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    checkpoint_enabled: bool = True
    
    @model_validator(mode='before')
    @classmethod
    def handle_nested_stages(cls, data):
        """Handle nested stage configurations from YAML."""
        if isinstance(data, dict) and 'stages' in data:
            # Extract nested stage configs to top level
            stages = data.pop('stages')
            for stage_name, stage_config in stages.items():
                if stage_name not in data:  # Don't override if already present
                    data[stage_name] = stage_config
        return data


class TaxonomyConfig(BaseConfig):
    """Configuration for taxonomy classification."""
    domains: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    categories: List[str] = Field(default_factory=list)
    topics: Dict[str, Any] = Field(default_factory=dict)


class QualityConfig(BaseConfig):
    """Configuration for quality assessment."""
    scoring: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Union[int, float]] = Field(default_factory=dict)
    filters: Dict[str, bool] = Field(default_factory=dict)


class CleansingRulesConfig(BaseConfig):
    """Configuration for text cleansing rules."""
    pii_patterns: Dict[str, str] = Field(default_factory=dict)
    boilerplate_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    allow_patterns: List[str] = Field(default_factory=list)
    deny_patterns: List[str] = Field(default_factory=list)
    normalization: Dict[str, bool] = Field(default_factory=dict)


class LabelsConfig(BaseConfig):
    """Configuration for label schemas."""
    instruction_following: Dict[str, Any] = Field(default_factory=dict)
    classification: Dict[str, List[str]] = Field(default_factory=dict)


class ConfigManager:
    """
    Configuration manager for loading and validating YAML configurations.
    
    Handles loading of all configuration files and provides validated
    configuration objects for the pipeline.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    def load_pipeline_config(self, config_file: str = "pipeline.yaml") -> PipelineConfig:
        """
        Load and validate pipeline configuration.
        
        Args:
            config_file: Pipeline configuration file name
            
        Returns:
            Validated PipelineConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            # Create default config if it doesn't exist
            default_config = PipelineConfig()
            self._save_config(config_path, default_config.model_dump())
            self._configs['pipeline'] = default_config
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                config_data = {}
            
            # Extract pipeline section if it exists
            if 'pipeline' in config_data:
                config_data = config_data['pipeline']
            
            pipeline_config = PipelineConfig(**config_data)
            self._configs['pipeline'] = pipeline_config
            
            return pipeline_config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading pipeline config: {e}")
    
    def load_taxonomy_config(self, config_file: str = "taxonomy.yaml") -> TaxonomyConfig:
        """Load and validate taxonomy configuration."""
        return self._load_config(config_file, TaxonomyConfig, 'taxonomy')
    
    def load_quality_config(self, config_file: str = "quality.yaml") -> QualityConfig:
        """Load and validate quality configuration."""
        return self._load_config(config_file, QualityConfig, 'quality')
    
    def load_cleansing_rules_config(self, config_file: str = "cleansing_rules.yaml") -> CleansingRulesConfig:
        """Load and validate cleansing rules configuration."""
        return self._load_config(config_file, CleansingRulesConfig, 'cleansing_rules')
    
    def load_labels_config(self, config_file: str = "labels.yaml") -> LabelsConfig:
        """Load and validate labels configuration."""
        return self._load_config(config_file, LabelsConfig, 'labels')
    
    def _load_config(self, config_file: str, config_class: type, config_key: str) -> Any:
        """
        Generic configuration loader.
        
        Args:
            config_file: Configuration file name
            config_class: Configuration class to instantiate
            config_key: Key to cache the config under
            
        Returns:
            Validated configuration object
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            # Create default config if it doesn't exist
            default_config = config_class()
            self._save_config(config_path, default_config.model_dump())
            self._configs[config_key] = default_config
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                config_data = {}
            
            config_obj = config_class(**config_data)
            self._configs[config_key] = config_obj
            
            return config_obj
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {config_key} config: {e}")
    
    def _save_config(self, config_path: Path, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save default config to {config_path}: {e}")
    
    def get_config(self, config_key: str) -> Optional[Any]:
        """Get cached configuration by key."""
        return self._configs.get(config_key)
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """
        Validate all configuration files.
        
        Returns:
            Dictionary mapping config names to validation status
        """
        results = {}
        
        config_loaders = {
            'pipeline': self.load_pipeline_config,
            'taxonomy': self.load_taxonomy_config,
            'quality': self.load_quality_config,
            'cleansing_rules': self.load_cleansing_rules_config,
            'labels': self.load_labels_config,
        }
        
        for config_name, loader in config_loaders.items():
            try:
                loader()
                results[config_name] = True
            except Exception as e:
                print(f"Validation failed for {config_name}: {e}")
                results[config_name] = False
        
        return results
    
    def get_stage_config(self, stage_name: str) -> Optional[BaseConfig]:
        """
        Get configuration for a specific pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Stage configuration object or None if not found
        """
        pipeline_config = self.get_config('pipeline')
        if not pipeline_config:
            pipeline_config = self.load_pipeline_config()
        
        return getattr(pipeline_config, stage_name, None)


def load_config(config_dir: str = "configs") -> ConfigManager:
    """
    Convenience function to create and return a ConfigManager.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Initialized ConfigManager instance
    """
    return ConfigManager(config_dir)


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "configs") -> ConfigManager:
    """
    Get global ConfigManager instance (singleton pattern).
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager