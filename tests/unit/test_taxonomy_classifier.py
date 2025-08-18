"""
Unit tests for TaxonomyClassifier.

Tests the rule-based domain categorization functionality including
keyword matching, pattern matching, and configuration handling.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.qudata.annotate.taxonomy import TaxonomyClassifier, CategoryResult, TaxonomyConfig
from src.qudata.models import ProcessingError, ErrorSeverity


class TestTaxonomyConfig:
    """Test TaxonomyConfig class."""
    
    def test_from_dict_basic(self):
        """Test creating TaxonomyConfig from dictionary."""
        config_dict = {
            'domains': {
                'technology': {
                    'keywords': ['software', 'programming'],
                    'patterns': ['\\bpython\\b']
                }
            },
            'categories': ['documentation', 'tutorial'],
            'topics': {'auto_extract': True},
            'default_domain': 'general',
            'min_confidence': 0.5
        }
        
        config = TaxonomyConfig.from_dict(config_dict)
        
        assert config.domains == config_dict['domains']
        assert config.categories == config_dict['categories']
        assert config.topics == config_dict['topics']
        assert config.default_domain == 'general'
        assert config.min_confidence == 0.5
    
    def test_from_dict_defaults(self):
        """Test TaxonomyConfig with default values."""
        config_dict = {
            'domains': {},
            'categories': [],
            'topics': {}
        }
        
        config = TaxonomyConfig.from_dict(config_dict)
        
        assert config.default_domain == 'uncategorized'
        assert config.min_confidence == 0.3


class TestTaxonomyClassifier:
    """Test TaxonomyClassifier class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample taxonomy configuration for testing."""
        return {
            'domains': {
                'technology': {
                    'keywords': ['software', 'programming', 'AI', 'machine learning', 'computer'],
                    'patterns': ['\\b(python|javascript|react|nodejs)\\b']
                },
                'science': {
                    'keywords': ['research', 'study', 'experiment', 'analysis', 'data'],
                    'patterns': ['\\b(hypothesis|methodology|results)\\b']
                },
                'business': {
                    'keywords': ['company', 'market', 'revenue', 'strategy', 'customer'],
                    'patterns': ['\\b(profit|loss|investment|ROI)\\b']
                }
            },
            'categories': ['documentation', 'tutorial', 'reference'],
            'topics': {'auto_extract': True, 'min_topic_size': 10},
            'default_domain': 'uncategorized',
            'min_confidence': 0.3
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name
    
    def test_init_with_config_file(self, temp_config_file):
        """Test initializing classifier with config file."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        assert classifier.config is not None
        assert 'technology' in classifier.config.domains
        assert len(classifier._compiled_patterns) > 0
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_init_missing_config_file(self):
        """Test initializing with missing config file."""
        with pytest.raises(ProcessingError) as exc_info:
            TaxonomyClassifier(config_path="nonexistent.yaml")
        
        assert exc_info.value.error_type == "ConfigurationError"
        assert exc_info.value.severity == ErrorSeverity.CRITICAL
    
    @patch('builtins.open', mock_open(read_data="invalid: yaml: content: ["))
    @patch('pathlib.Path.exists', return_value=True)
    def test_init_invalid_yaml(self, mock_exists):
        """Test initializing with invalid YAML."""
        with pytest.raises(ProcessingError) as exc_info:
            TaxonomyClassifier(config_path="invalid.yaml")
        
        assert exc_info.value.error_type == "ConfigurationError"
    
    def test_categorize_content_technology(self, temp_config_file):
        """Test categorizing technology content."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        content = """
        This is a tutorial about Python programming and software development.
        We'll cover machine learning algorithms and AI concepts.
        """
        
        result = classifier.categorize_content(content)
        
        assert result.domain == 'technology'
        assert result.confidence > 0
        assert 'programming' in result.matched_keywords
        assert 'software' in result.matched_keywords
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_categorize_content_science(self, temp_config_file):
        """Test categorizing science content."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        content = """
        This research study presents our methodology for data analysis.
        The experiment shows significant results in our hypothesis testing.
        """
        
        result = classifier.categorize_content(content)
        
        assert result.domain == 'science'
        assert result.confidence > 0
        assert 'research' in result.matched_keywords
        assert 'study' in result.matched_keywords
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_categorize_content_pattern_matching(self, temp_config_file):
        """Test categorization with pattern matching."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        content = """
        Here's a Python script that uses JavaScript for the frontend.
        The nodejs backend handles the API calls.
        """
        
        result = classifier.categorize_content(content)
        
        assert result.domain == 'technology'
        assert len(result.matched_patterns) > 0
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_categorize_content_empty(self, temp_config_file):
        """Test categorizing empty content."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        result = classifier.categorize_content("")
        
        assert result.domain == 'uncategorized'
        assert result.confidence == 0.0
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_categorize_content_low_confidence(self, temp_config_file):
        """Test categorizing content with low confidence."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        # Content with no matching keywords - make it long to reduce confidence
        content = "This is a very long text about general topics that don't match any specific area very well. " * 20
        
        result = classifier.categorize_content(content)
        
        # Should fall back to uncategorized due to no matches or low confidence
        assert result.domain == 'uncategorized'
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_categorize_content_multiple_domains(self, temp_config_file):
        """Test content that matches multiple domains."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        content = """
        This research study analyzes software development practices in companies.
        We examine programming methodologies and business strategies.
        """
        
        result = classifier.categorize_content(content)
        
        # Should pick the domain with highest score
        assert result.domain in ['technology', 'science', 'business']
        assert result.confidence > 0
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_get_available_domains(self, temp_config_file):
        """Test getting available domains."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        domains = classifier.get_available_domains()
        
        assert 'technology' in domains
        assert 'science' in domains
        assert 'business' in domains
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_get_available_categories(self, temp_config_file):
        """Test getting available categories."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        categories = classifier.get_available_categories()
        
        assert 'documentation' in categories
        assert 'tutorial' in categories
        assert 'reference' in categories
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_validate_config_valid(self, temp_config_file):
        """Test validating a valid configuration."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        assert classifier.validate_config() is True
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_validate_config_no_domains(self):
        """Test validating config with no domains."""
        config_dict = {
            'domains': {},
            'categories': [],
            'topics': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        classifier = TaxonomyClassifier(config_path=config_path)
        
        with pytest.raises(ProcessingError) as exc_info:
            classifier.validate_config()
        
        assert exc_info.value.error_type == "ConfigurationError"
        assert "No domains defined" in exc_info.value.message
        
        # Clean up
        Path(config_path).unlink()
    
    def test_validate_config_invalid_domain(self):
        """Test validating config with invalid domain structure."""
        config_dict = {
            'domains': {
                'technology': "invalid_structure"  # Should be dict
            },
            'categories': [],
            'topics': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        classifier = TaxonomyClassifier(config_path=config_path)
        
        with pytest.raises(ProcessingError) as exc_info:
            classifier.validate_config()
        
        assert exc_info.value.error_type == "ConfigurationError"
        assert "must be a dictionary" in exc_info.value.message
        
        # Clean up
        Path(config_path).unlink()
    
    def test_validate_config_domain_no_keywords_or_patterns(self):
        """Test validating domain with no keywords or patterns."""
        config_dict = {
            'domains': {
                'technology': {}  # No keywords or patterns
            },
            'categories': [],
            'topics': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        classifier = TaxonomyClassifier(config_path=config_path)
        
        with pytest.raises(ProcessingError) as exc_info:
            classifier.validate_config()
        
        assert exc_info.value.error_type == "ConfigurationError"
        assert "must have either keywords or patterns" in exc_info.value.message
        
        # Clean up
        Path(config_path).unlink()
    
    def test_reload_config(self, temp_config_file):
        """Test reloading configuration."""
        classifier = TaxonomyClassifier(config_path=temp_config_file)
        
        original_domains = len(classifier.config.domains)
        
        # Modify config file
        new_config = {
            'domains': {
                'new_domain': {
                    'keywords': ['test'],
                    'patterns': []
                }
            },
            'categories': [],
            'topics': {}
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(new_config, f)
        
        classifier.reload_config()
        
        assert len(classifier.config.domains) != original_domains
        assert 'new_domain' in classifier.config.domains
        
        # Clean up
        Path(temp_config_file).unlink()


class TestCategoryResult:
    """Test CategoryResult class."""
    
    def test_category_result_creation(self):
        """Test creating CategoryResult."""
        result = CategoryResult(
            domain='technology',
            confidence=0.8,
            matched_keywords=['python', 'programming'],
            matched_patterns=['\\bpython\\b']
        )
        
        assert result.domain == 'technology'
        assert result.confidence == 0.8
        assert 'python' in result.matched_keywords
        assert '\\bpython\\b' in result.matched_patterns
    
    def test_category_result_defaults(self):
        """Test CategoryResult with default values."""
        result = CategoryResult(domain='science', confidence=0.5)
        
        assert result.domain == 'science'
        assert result.confidence == 0.5
        assert result.matched_keywords == []
        assert result.matched_patterns == []