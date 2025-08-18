"""
Unit tests for configuration management.

Tests configuration loading, validation, and error handling.
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qudata.config import (
    ConfigManager, PipelineConfig, IngestConfig, CleanConfig,
    AnnotateConfig, ScoreConfig, PackConfig, ExportConfig,
    TaxonomyConfig, QualityConfig, CleansingRulesConfig, LabelsConfig,
    load_config, get_config_manager
)


class TestConfigModels:
    """Test configuration model validation."""
    
    def test_ingest_config_defaults(self):
        """Test IngestConfig default values."""
        config = IngestConfig()
        
        assert config.enabled is True
        assert config.file_types == ["txt", "md", "html", "pdf", "docx", "epub"]
        assert config.max_file_size == "100MB"
        assert config.parallel_workers == 4
    
    def test_ingest_config_validation(self):
        """Test IngestConfig validation."""
        # Valid config
        config = IngestConfig(
            enabled=True,
            file_types=["txt", "pdf"],
            max_file_size="50MB",
            parallel_workers=2
        )
        assert config.parallel_workers == 2
        
        # Invalid parallel workers
        with pytest.raises(ValueError):
            IngestConfig(parallel_workers=0)
        
        with pytest.raises(ValueError):
            IngestConfig(parallel_workers=20)
        
        # Invalid file size format
        with pytest.raises(ValueError):
            IngestConfig(max_file_size="invalid")
        
        with pytest.raises(ValueError):
            IngestConfig(max_file_size="0MB")
        
        # Valid file size formats
        valid_sizes = ["100B", "50KB", "10MB", "1GB", "1.5GB"]
        for size in valid_sizes:
            config = IngestConfig(max_file_size=size)
            assert config.max_file_size == size
    
    def test_clean_config_validation(self):
        """Test CleanConfig validation."""
        # Valid config
        config = CleanConfig(min_length=50, max_length=1000)
        assert config.min_length == 50
        assert config.max_length == 1000
        
        # Invalid: max_length <= min_length
        with pytest.raises(ValueError):
            CleanConfig(min_length=1000, max_length=500)
        
        with pytest.raises(ValueError):
            CleanConfig(min_length=100, max_length=100)
    
    def test_score_config_validation(self):
        """Test ScoreConfig validation."""
        # Valid config with weights summing to 1.0
        weights = {
            "length": 0.3,
            "language": 0.2,
            "coherence": 0.3,
            "uniqueness": 0.1,
            "readability": 0.1
        }
        config = ScoreConfig(weights=weights)
        assert config.weights == weights
        
        # Invalid: weights don't sum to 1.0
        invalid_weights = {
            "length": 0.5,
            "language": 0.3,
            "coherence": 0.3  # Sum = 1.1
        }
        with pytest.raises(ValueError):
            ScoreConfig(weights=invalid_weights)
        
        # Invalid quality score range
        with pytest.raises(ValueError):
            ScoreConfig(min_quality_score=-0.1)
        
        with pytest.raises(ValueError):
            ScoreConfig(min_quality_score=1.1)
    
    def test_pack_config_validation(self):
        """Test PackConfig validation."""
        # Valid formats
        config = PackConfig(formats=["jsonl", "chatml"])
        assert config.formats == ["jsonl", "chatml"]
        
        # Invalid format
        with pytest.raises(ValueError):
            PackConfig(formats=["invalid_format"])
        
        # Mixed valid and invalid
        with pytest.raises(ValueError):
            PackConfig(formats=["jsonl", "invalid"])
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()
        
        assert config.name == "qudata"
        assert config.version == "1.0.0"
        assert isinstance(config.ingest, IngestConfig)
        assert isinstance(config.clean, CleanConfig)
        assert isinstance(config.annotate, AnnotateConfig)
        assert isinstance(config.score, ScoreConfig)
        assert isinstance(config.pack, PackConfig)
        assert isinstance(config.export, ExportConfig)
        assert config.parallel_processing is True
        assert config.log_level == "INFO"
        assert config.checkpoint_enabled is True
    
    def test_pipeline_config_validation(self):
        """Test PipelineConfig validation."""
        # Valid log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = PipelineConfig(log_level=level)
            assert config.log_level == level
        
        # Invalid log level
        with pytest.raises(ValueError):
            PipelineConfig(log_level="INVALID")


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_init(self):
        """Test ConfigManager initialization."""
        # Valid config directory
        manager = ConfigManager(str(self.config_dir))
        assert manager.config_dir == self.config_dir
        
        # Invalid config directory
        with pytest.raises(FileNotFoundError):
            ConfigManager("/nonexistent/directory")
    
    def test_load_pipeline_config_default(self):
        """Test loading default pipeline config when file doesn't exist."""
        manager = ConfigManager(str(self.config_dir))
        config = manager.load_pipeline_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.name == "qudata"
        
        # Check that default config file was created
        config_file = self.config_dir / "pipeline.yaml"
        assert config_file.exists()
    
    def test_load_pipeline_config_from_file(self):
        """Test loading pipeline config from existing file."""
        # Create test config file
        config_data = {
            "pipeline": {
                "name": "test-pipeline",
                "version": "2.0.0",
                "log_level": "DEBUG",
                "ingest": {
                    "enabled": False,
                    "max_file_size": "50MB"
                }
            }
        }
        
        config_file = self.config_dir / "pipeline.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(str(self.config_dir))
        config = manager.load_pipeline_config()
        
        assert config.name == "test-pipeline"
        assert config.version == "2.0.0"
        assert config.log_level == "DEBUG"
        assert config.ingest.enabled is False
        assert config.ingest.max_file_size == "50MB"
    
    def test_load_pipeline_config_without_pipeline_key(self):
        """Test loading config when data is at root level."""
        config_data = {
            "name": "root-level-config",
            "version": "1.5.0",
            "log_level": "WARNING"
        }
        
        config_file = self.config_dir / "pipeline.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(str(self.config_dir))
        config = manager.load_pipeline_config()
        
        assert config.name == "root-level-config"
        assert config.version == "1.5.0"
        assert config.log_level == "WARNING"
    
    def test_load_invalid_yaml(self):
        """Test handling of invalid YAML files."""
        config_file = self.config_dir / "pipeline.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = ConfigManager(str(self.config_dir))
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            manager.load_pipeline_config()
    
    def test_load_invalid_config_data(self):
        """Test handling of invalid configuration data."""
        config_data = {
            "pipeline": {
                "name": "test",
                "log_level": "INVALID_LEVEL"  # Invalid log level
            }
        }
        
        config_file = self.config_dir / "pipeline.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(str(self.config_dir))
        
        with pytest.raises(ValueError, match="Error loading pipeline config"):
            manager.load_pipeline_config()
    
    def test_load_taxonomy_config(self):
        """Test loading taxonomy configuration."""
        config_data = {
            "domains": {
                "technology": {
                    "keywords": ["software", "programming"],
                    "patterns": ["\\\\b(python|javascript)\\\\b"]
                }
            },
            "categories": ["documentation", "tutorial"]
        }
        
        config_file = self.config_dir / "taxonomy.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(str(self.config_dir))
        config = manager.load_taxonomy_config()
        
        assert isinstance(config, TaxonomyConfig)
        assert "technology" in config.domains
        assert config.categories == ["documentation", "tutorial"]
    
    def test_load_quality_config(self):
        """Test loading quality configuration."""
        config_data = {
            "scoring": {
                "weights": {
                    "length": 0.3,
                    "coherence": 0.7
                }
            },
            "thresholds": {
                "min_length": 50
            }
        }
        
        config_file = self.config_dir / "quality.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(str(self.config_dir))
        config = manager.load_quality_config()
        
        assert isinstance(config, QualityConfig)
        assert config.scoring["weights"]["length"] == 0.3
        assert config.thresholds["min_length"] == 50
    
    def test_get_cached_config(self):
        """Test getting cached configuration."""
        manager = ConfigManager(str(self.config_dir))
        
        # Load config (should cache it)
        config1 = manager.load_pipeline_config()
        
        # Get cached config
        cached_config = manager.get_config('pipeline')
        
        assert cached_config is config1
        assert isinstance(cached_config, PipelineConfig)
    
    def test_get_stage_config(self):
        """Test getting stage-specific configuration."""
        manager = ConfigManager(str(self.config_dir))
        
        # Load pipeline config first
        manager.load_pipeline_config()
        
        # Get stage configs
        ingest_config = manager.get_stage_config('ingest')
        clean_config = manager.get_stage_config('clean')
        invalid_config = manager.get_stage_config('nonexistent')
        
        assert isinstance(ingest_config, IngestConfig)
        assert isinstance(clean_config, CleanConfig)
        assert invalid_config is None
    
    def test_validate_all_configs(self):
        """Test validation of all configuration files."""
        # Create valid config files
        configs = {
            "pipeline.yaml": {"name": "test", "version": "1.0.0"},
            "taxonomy.yaml": {"domains": {}, "categories": []},
            "quality.yaml": {"scoring": {}, "thresholds": {}},
            "cleansing_rules.yaml": {"pii_patterns": {}, "normalization": {}},
            "labels.yaml": {"instruction_following": {}, "classification": {}}
        }
        
        for filename, data in configs.items():
            config_file = self.config_dir / filename
            with open(config_file, 'w') as f:
                yaml.dump(data, f)
        
        manager = ConfigManager(str(self.config_dir))
        results = manager.validate_all_configs()
        
        assert all(results.values())  # All should be True
        assert len(results) == 5
    
    def test_validate_all_configs_with_errors(self):
        """Test validation with some invalid configs."""
        # Create one valid and one invalid config
        valid_config = self.config_dir / "pipeline.yaml"
        with open(valid_config, 'w') as f:
            yaml.dump({"name": "test"}, f)
        
        invalid_config = self.config_dir / "quality.yaml"
        with open(invalid_config, 'w') as f:
            f.write("invalid: yaml: [")
        
        manager = ConfigManager(str(self.config_dir))
        results = manager.validate_all_configs()
        
        assert results['pipeline'] is True
        assert results['quality'] is False


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_load_config_function(self):
        """Test load_config convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            
            manager = load_config(str(config_dir))
            assert isinstance(manager, ConfigManager)
            assert manager.config_dir == config_dir
    
    def test_get_config_manager_singleton(self):
        """Test get_config_manager singleton behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            
            # Clear global instance
            import qudata.config
            qudata.config._config_manager = None
            
            # First call should create instance
            manager1 = get_config_manager(str(config_dir))
            
            # Second call should return same instance
            manager2 = get_config_manager(str(config_dir))
            
            assert manager1 is manager2


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_workflow(self):
        """Test complete configuration loading workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()
            
            # Create comprehensive config
            pipeline_config = {
                "pipeline": {
                    "name": "integration-test",
                    "version": "1.0.0",
                    "ingest": {
                        "enabled": True,
                        "file_types": ["txt", "pdf"],
                        "max_file_size": "200MB",
                        "parallel_workers": 8
                    },
                    "clean": {
                        "enabled": True,
                        "min_length": 200,
                        "max_length": 50000
                    },
                    "score": {
                        "enabled": True,
                        "min_quality_score": 0.8,
                        "weights": {
                            "length": 0.25,
                            "language": 0.25,
                            "coherence": 0.25,
                            "uniqueness": 0.25
                        }
                    }
                }
            }
            
            config_file = config_dir / "pipeline.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(pipeline_config, f)
            
            # Load and validate
            manager = ConfigManager(str(config_dir))
            config = manager.load_pipeline_config()
            
            # Verify all settings
            assert config.name == "integration-test"
            assert config.ingest.file_types == ["txt", "pdf"]
            assert config.ingest.parallel_workers == 8
            assert config.clean.min_length == 200
            assert config.score.min_quality_score == 0.8
            assert config.score.weights["length"] == 0.25
            
            # Test stage config access
            ingest_config = manager.get_stage_config('ingest')
            assert ingest_config.max_file_size == "200MB"


if __name__ == "__main__":
    pytest.main([__file__])