"""
Taxonomy classification - rule-based categorization.

This module provides domain and topic categorization functionality using
configurable rules from YAML configuration files.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..models import Document, DocumentMetadata, ProcessingError, ErrorSeverity
from ..config import ConfigManager


@dataclass
class CategoryResult:
    """Result of content categorization."""
    domain: str
    confidence: float
    matched_keywords: List[str] = None
    matched_patterns: List[str] = None
    
    def __post_init__(self):
        if self.matched_keywords is None:
            self.matched_keywords = []
        if self.matched_patterns is None:
            self.matched_patterns = []


@dataclass
class TaxonomyConfig:
    """Configuration for taxonomy classification."""
    domains: Dict[str, Dict[str, Any]]
    categories: List[str]
    topics: Dict[str, Any]
    default_domain: str = "uncategorized"
    min_confidence: float = 0.3
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaxonomyConfig':
        """Create TaxonomyConfig from dictionary."""
        return cls(
            domains=config_dict.get('domains', {}),
            categories=config_dict.get('categories', []),
            topics=config_dict.get('topics', {}),
            default_domain=config_dict.get('default_domain', 'uncategorized'),
            min_confidence=config_dict.get('min_confidence', 0.3)
        )


class TaxonomyClassifier:
    """Rule-based domain and topic categorization system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the taxonomy classifier.
        
        Args:
            config_path: Path to taxonomy configuration file. If None, uses default.
        """
        self.config_path = config_path or "configs/taxonomy.yaml"
        self.config = self._load_config()
        self._compiled_patterns = self._compile_patterns()
    
    def _load_config(self) -> TaxonomyConfig:
        """Load taxonomy configuration from YAML file."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Taxonomy config file not found: {self.config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            return TaxonomyConfig.from_dict(config_dict)
        
        except Exception as e:
            raise ProcessingError(
                stage="taxonomy_init",
                error_type="ConfigurationError",
                message=f"Failed to load taxonomy configuration: {str(e)}",
                severity=ErrorSeverity.CRITICAL
            )
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for each domain."""
        compiled = {}
        for domain, domain_config in self.config.domains.items():
            # Ensure domain_config is a dictionary
            if not isinstance(domain_config, dict):
                continue
            patterns = domain_config.get('patterns', [])
            compiled[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled
    
    def categorize_content(self, content: str) -> CategoryResult:
        """
        Categorize content by domain using keyword and pattern matching.
        
        Args:
            content: Text content to categorize
            
        Returns:
            CategoryResult with domain classification and confidence score
        """
        if not content or not content.strip():
            return CategoryResult(
                domain=self.config.default_domain,
                confidence=0.0
            )
        
        content_lower = content.lower()
        domain_scores = {}
        
        # Score each domain based on keyword and pattern matches
        for domain, domain_config in self.config.domains.items():
            score, matched_keywords, matched_patterns = self._score_domain(
                content_lower, content, domain, domain_config
            )
            
            if score > 0:
                domain_scores[domain] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'matched_patterns': matched_patterns
                }
        
        # Find the best matching domain
        if not domain_scores:
            return CategoryResult(
                domain=self.config.default_domain,
                confidence=0.0
            )
        
        best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d]['score'])
        best_score = domain_scores[best_domain]['score']
        
        # Normalize confidence score (simple approach: score / content_length * 10)
        content_words = len(content.split())
        confidence = min(1.0, best_score / max(1, content_words) * 10)
        
        # Check if confidence meets minimum threshold
        if confidence < self.config.min_confidence:
            return CategoryResult(
                domain=self.config.default_domain,
                confidence=confidence
            )
        
        return CategoryResult(
            domain=best_domain,
            confidence=confidence,
            matched_keywords=domain_scores[best_domain]['matched_keywords'],
            matched_patterns=domain_scores[best_domain]['matched_patterns']
        )
    
    def _score_domain(self, content_lower: str, content: str, domain: str, 
                     domain_config: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """
        Score how well content matches a specific domain.
        
        Args:
            content_lower: Lowercase version of content for keyword matching
            content: Original content for pattern matching
            domain: Domain name
            domain_config: Configuration for this domain
            
        Returns:
            Tuple of (score, matched_keywords, matched_patterns)
        """
        score = 0.0
        matched_keywords = []
        matched_patterns = []
        
        # Score based on keyword matches
        keywords = domain_config.get('keywords', [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            if count > 0:
                score += count * 1.0  # Each keyword match adds 1.0 to score
                matched_keywords.append(keyword)
        
        # Score based on pattern matches
        patterns = self._compiled_patterns.get(domain, [])
        for i, pattern in enumerate(patterns):
            matches = pattern.findall(content)
            if matches:
                score += len(matches) * 2.0  # Pattern matches are weighted higher
                matched_patterns.append(domain_config.get('patterns', [])[i])
        
        return score, matched_keywords, matched_patterns
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return list(self.config.domains.keys())
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        return self.config.categories
    
    def validate_config(self) -> bool:
        """
        Validate the taxonomy configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ProcessingError: If configuration is invalid
        """
        if not self.config.domains:
            raise ProcessingError(
                stage="taxonomy_validation",
                error_type="ConfigurationError",
                message="No domains defined in taxonomy configuration",
                severity=ErrorSeverity.HIGH
            )
        
        # Validate each domain has required fields
        for domain, domain_config in self.config.domains.items():
            if not isinstance(domain_config, dict):
                raise ProcessingError(
                    stage="taxonomy_validation",
                    error_type="ConfigurationError",
                    message=f"Domain '{domain}' configuration must be a dictionary",
                    severity=ErrorSeverity.HIGH
                )
            
            # Check for keywords or patterns
            keywords = domain_config.get('keywords', [])
            patterns = domain_config.get('patterns', [])
            
            if not keywords and not patterns:
                raise ProcessingError(
                    stage="taxonomy_validation",
                    error_type="ConfigurationError",
                    message=f"Domain '{domain}' must have either keywords or patterns defined",
                    severity=ErrorSeverity.MEDIUM
                )
        
        return True
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self._compiled_patterns = self._compile_patterns()
        self.validate_config()
