"""
Boilerplate removal for cleaning common web and document artifacts.

This module provides functionality to detect and remove boilerplate content
such as navigation elements, advertisements, headers, footers, and other
non-content elements that are commonly found in web pages and documents.
"""

import re
import yaml
from typing import Dict, List, Set, Any, Optional, Pattern
from dataclasses import dataclass, field
from pathlib import Path
from ..models import ProcessingError, ErrorSeverity


@dataclass
class BoilerplatePattern:
    """Represents a boilerplate pattern for detection."""
    name: str
    pattern: str
    pattern_type: str = "regex"  # regex, keyword, css_selector
    case_sensitive: bool = False
    whole_word: bool = False
    
    def __post_init__(self):
        """Compile regex pattern if needed."""
        if self.pattern_type == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            if self.whole_word:
                self.pattern = r'\b' + self.pattern + r'\b'
            self._compiled_pattern = re.compile(self.pattern, flags)
        else:
            self._compiled_pattern = None
    
    def matches(self, text: str) -> bool:
        """Check if this pattern matches the given text."""
        if self.pattern_type == "regex":
            return bool(self._compiled_pattern.search(text))
        elif self.pattern_type == "keyword":
            search_text = text if self.case_sensitive else text.lower()
            search_pattern = self.pattern if self.case_sensitive else self.pattern.lower()
            
            if self.whole_word:
                return bool(re.search(r'\b' + re.escape(search_pattern) + r'\b', search_text))
            else:
                return search_pattern in search_text
        else:
            return False


@dataclass
class BoilerplateRemovalResult:
    """Result of boilerplate removal process."""
    original_text: str
    cleaned_text: str
    removed_patterns: List[str] = field(default_factory=list)
    removed_content: List[str] = field(default_factory=list)
    
    def get_removal_ratio(self) -> float:
        """Get ratio of content removed."""
        if not self.original_text:
            return 0.0
        original_len = len(self.original_text)
        cleaned_len = len(self.cleaned_text)
        return (original_len - cleaned_len) / original_len
    
    def get_removed_count(self) -> int:
        """Get number of boilerplate patterns removed."""
        return len(self.removed_patterns)


class BoilerplateRemover:
    """
    Engine for detecting and removing boilerplate content.
    
    Supports configurable patterns for different types of boilerplate content
    including navigation, advertisements, legal text, and other common artifacts.
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize BoilerplateRemover.
        
        Args:
            config: Configuration dictionary with boilerplate patterns
            config_file: Path to YAML configuration file
        """
        self.config = config or {}
        
        # Load configuration from file if provided
        if config_file:
            self._load_config_file(config_file)
        
        # Removal settings
        boilerplate_config = self.config.get('boilerplate_removal', {})
        self.aggressive_removal = boilerplate_config.get('aggressive_removal', 
                                                        self.config.get('aggressive_removal', False))
        self.preserve_structure = boilerplate_config.get('preserve_structure', 
                                                        self.config.get('preserve_structure', True))
        self.min_content_length = boilerplate_config.get('min_content_length', 
                                                        self.config.get('min_content_length', 50))
        
        # Content type specific settings
        self.remove_navigation = boilerplate_config.get('remove_navigation', 
                                                       self.config.get('remove_navigation', True))
        self.remove_ads = boilerplate_config.get('remove_ads', 
                                                self.config.get('remove_ads', True))
        self.remove_cookies = boilerplate_config.get('remove_cookies', 
                                                    self.config.get('remove_cookies', True))
        self.remove_social = boilerplate_config.get('remove_social', 
                                                   self.config.get('remove_social', True))
        self.remove_legal = boilerplate_config.get('remove_legal', 
                                                  self.config.get('remove_legal', True))
        self.remove_headers_footers = boilerplate_config.get('remove_headers_footers', 
                                                            self.config.get('remove_headers_footers', True))
        
        # Initialize patterns
        self.patterns: List[BoilerplatePattern] = []
        self._load_patterns()
    
    def _load_config_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with existing config
                    self.config.update(file_config)
        except Exception as e:
            raise ProcessingError(
                stage="boilerplate_config",
                error_type="ConfigurationError",
                message=f"Failed to load boilerplate config: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
    
    def _load_patterns(self):
        """Load boilerplate patterns from configuration."""
        self.patterns = []
        
        # Load patterns from config
        boilerplate_patterns = self.config.get('boilerplate_patterns', {})
        
        # Load all pattern categories
        pattern_categories = [
            ('navigation', self.remove_navigation),
            ('headers_footers', self.remove_headers_footers),
            ('ads', self.remove_ads),
            ('cookies', self.remove_cookies),
            ('social', self.remove_social),
            ('legal', self.remove_legal),
            ('comments', self.config.get('remove_comments', True)),
            ('pagination', self.config.get('remove_pagination', True)),
            ('search', self.config.get('remove_search', True)),
            ('forms', self.config.get('remove_forms', False))
        ]
        
        for category, should_remove in pattern_categories:
            if should_remove and category in boilerplate_patterns:
                patterns_list = boilerplate_patterns[category]
                for pattern in patterns_list:
                    self.patterns.append(BoilerplatePattern(
                        name=f"{category}_{pattern.replace(' ', '_')}",
                        pattern=pattern,
                        pattern_type="keyword",
                        case_sensitive=False
                    ))
        
        # Load deny patterns (low-quality content)
        deny_patterns = self.config.get('deny_patterns', {})
        for category, patterns_list in deny_patterns.items():
            for pattern in patterns_list:
                self.patterns.append(BoilerplatePattern(
                    name=f"deny_{category}_{len(self.patterns)}",
                    pattern=pattern,
                    pattern_type="regex" if pattern.startswith('\\') or '^' in pattern else "keyword",
                    case_sensitive=False
                ))
        
        # Load boilerplate removal custom patterns
        boilerplate_config = self.config.get('boilerplate_removal', {})
        custom_patterns = boilerplate_config.get('custom_patterns', [])
        
        for pattern_config in custom_patterns:
            if isinstance(pattern_config, dict):
                # Extract pattern configuration
                pattern_data = {
                    'name': pattern_config.get('name', f"custom_{len(self.patterns)}"),
                    'pattern': pattern_config.get('pattern', ''),
                    'pattern_type': pattern_config.get('pattern_type', 'keyword'),
                    'case_sensitive': pattern_config.get('case_sensitive', False),
                    'whole_word': pattern_config.get('whole_word', False)
                }
                self.patterns.append(BoilerplatePattern(**pattern_data))
        
        # Load legacy custom patterns for backward compatibility
        legacy_custom = self.config.get('custom_patterns', [])
        for pattern_config in legacy_custom:
            if isinstance(pattern_config, dict):
                self.patterns.append(BoilerplatePattern(**pattern_config))
            elif isinstance(pattern_config, str):
                self.patterns.append(BoilerplatePattern(
                    name=f"legacy_custom_{len(self.patterns)}",
                    pattern=pattern_config,
                    pattern_type="keyword"
                ))
    
    def remove_boilerplate(self, text: str) -> BoilerplateRemovalResult:
        """
        Remove boilerplate content from text.
        
        Args:
            text: Input text to clean
            
        Returns:
            BoilerplateRemovalResult with cleaned text and removal statistics
        """
        if not text or not isinstance(text, str):
            return BoilerplateRemovalResult(
                original_text=text or "",
                cleaned_text=""
            )
        
        result = BoilerplateRemovalResult(
            original_text=text,
            cleaned_text=text
        )
        
        try:
            # Process text line by line for better control
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line_cleaned = self._process_line(line, result)
                if line_cleaned is not None:  # None means line was removed
                    cleaned_lines.append(line_cleaned)
            
            # Join lines back together
            result.cleaned_text = '\n'.join(cleaned_lines)
            
            # Post-processing cleanup
            result.cleaned_text = self._post_process_text(result.cleaned_text)
            
        except Exception as e:
            raise ProcessingError(
                stage="boilerplate_removal",
                error_type="BoilerplateRemovalError",
                message=f"Failed to remove boilerplate: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return result
    
    def _process_line(self, line: str, result: BoilerplateRemovalResult) -> Optional[str]:
        """
        Process a single line for boilerplate removal.
        
        Args:
            line: Line to process
            result: Result object to update with removal information
            
        Returns:
            Cleaned line or None if line should be removed entirely
        """
        original_line = line
        
        # Skip empty lines
        if not line.strip():
            return line
        
        # Check if entire line should be removed
        for pattern in self.patterns:
            if pattern.matches(line):
                result.removed_patterns.append(pattern.name)
                result.removed_content.append(original_line)
                
                # In aggressive mode, remove entire line
                if self.aggressive_removal:
                    return None
                
                # Otherwise, try to clean the line
                line = self._clean_line_with_pattern(line, pattern)
        
        # Check if cleaned line is too short (but allow some short lines)
        if len(line.strip()) < 3:  # Only remove very short lines
            return None
        
        return line
    
    def _clean_line_with_pattern(self, line: str, pattern: BoilerplatePattern) -> str:
        """Clean a line by removing or replacing pattern matches."""
        if pattern.pattern_type == "regex":
            return pattern._compiled_pattern.sub('', line)
        elif pattern.pattern_type == "keyword":
            # Replace keyword with empty string
            if pattern.case_sensitive:
                return line.replace(pattern.pattern, '')
            else:
                # Case-insensitive replacement
                return re.sub(
                    re.escape(pattern.pattern), 
                    '', 
                    line, 
                    flags=re.IGNORECASE
                )
        return line
    
    def _post_process_text(self, text: str) -> str:
        """Post-process cleaned text to improve formatting."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        
        # Remove lines that are mostly punctuation or symbols
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
            
            # Check if line is mostly non-alphanumeric (be more lenient)
            if len(stripped) >= 3:  # Keep lines with at least 3 characters
                alphanumeric_count = sum(1 for c in stripped if c.isalnum())
                if len(stripped) < 10 or alphanumeric_count / len(stripped) >= 0.2:  # At least 20% alphanumeric or short lines
                    cleaned_lines.append(line)
        
        # Join and final cleanup
        text = '\n'.join(cleaned_lines)
        text = text.strip()
        
        return text
    
    def detect_boilerplate_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Detect boilerplate patterns in text without removing them.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping pattern names to matched content
        """
        detected = {}
        
        for pattern in self.patterns:
            matches = []
            
            if pattern.pattern_type == "regex":
                for match in pattern._compiled_pattern.finditer(text):
                    matches.append(match.group())
            elif pattern.pattern_type == "keyword":
                lines = text.split('\n')
                for line in lines:
                    if pattern.matches(line):
                        matches.append(line.strip())
            
            if matches:
                detected[pattern.name] = matches
        
        return detected
    
    def add_custom_pattern(self, name: str, pattern: str, 
                          pattern_type: str = "keyword", **kwargs):
        """Add a custom boilerplate pattern."""
        custom_pattern = BoilerplatePattern(
            name=name,
            pattern=pattern,
            pattern_type=pattern_type,
            **kwargs
        )
        self.patterns.append(custom_pattern)
    
    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name."""
        original_count = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.name != name]
        return len(self.patterns) < original_count
    
    def get_pattern_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about boilerplate patterns in text."""
        detected = self.detect_boilerplate_patterns(text)
        
        stats = {
            'total_patterns_detected': len(detected),
            'total_matches': sum(len(matches) for matches in detected.values()),
            'patterns_by_type': {},
            'most_common_patterns': []
        }
        
        # Group by pattern type
        for pattern_name, matches in detected.items():
            # Extract pattern type from name
            pattern_type = pattern_name.split('_')[0] if '_' in pattern_name else 'custom'
            if pattern_type not in stats['patterns_by_type']:
                stats['patterns_by_type'][pattern_type] = 0
            stats['patterns_by_type'][pattern_type] += len(matches)
        
        # Most common patterns
        pattern_counts = [(name, len(matches)) for name, matches in detected.items()]
        pattern_counts.sort(key=lambda x: x[1], reverse=True)
        stats['most_common_patterns'] = pattern_counts[:10]
        
        return stats
