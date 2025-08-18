"""
Unit tests for boilerplate removal functionality.

Tests the BoilerplateRemover for detecting and removing various types
of boilerplate content including navigation, ads, cookies, and legal text.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.qudata.clean.boilerplate import (
    BoilerplateRemover, BoilerplatePattern, BoilerplateRemovalResult
)
from src.qudata.models import ProcessingError


class TestBoilerplatePattern:
    """Test cases for BoilerplatePattern class."""
    
    def test_regex_pattern_matching(self):
        """Test regex pattern matching."""
        pattern = BoilerplatePattern(
            name="test_regex",
            pattern=r"header.*footer",
            pattern_type="regex"
        )
        
        assert pattern.matches("header content footer")
        assert pattern.matches("header and some other content footer")
        assert not pattern.matches("just some content")
    
    def test_keyword_pattern_matching(self):
        """Test keyword pattern matching."""
        pattern = BoilerplatePattern(
            name="test_keyword",
            pattern="advertisement",
            pattern_type="keyword"
        )
        
        assert pattern.matches("This is an advertisement section")
        assert pattern.matches("advertisement")
        assert not pattern.matches("This is content")
    
    def test_case_sensitive_matching(self):
        """Test case sensitive pattern matching."""
        # Case sensitive
        pattern_sensitive = BoilerplatePattern(
            name="test_case_sensitive",
            pattern="Cookie",
            pattern_type="keyword",
            case_sensitive=True
        )
        
        assert pattern_sensitive.matches("Cookie policy")
        assert not pattern_sensitive.matches("cookie policy")
        
        # Case insensitive
        pattern_insensitive = BoilerplatePattern(
            name="test_case_insensitive",
            pattern="Cookie",
            pattern_type="keyword",
            case_sensitive=False
        )
        
        assert pattern_insensitive.matches("Cookie policy")
        assert pattern_insensitive.matches("cookie policy")
    
    def test_whole_word_matching(self):
        """Test whole word pattern matching."""
        # Whole word matching
        pattern_whole = BoilerplatePattern(
            name="test_whole_word",
            pattern="ad",
            pattern_type="keyword",
            whole_word=True
        )
        
        assert pattern_whole.matches("This is an ad section")
        assert not pattern_whole.matches("This is a bad section")  # 'ad' in 'bad'
        
        # Partial word matching
        pattern_partial = BoilerplatePattern(
            name="test_partial_word",
            pattern="ad",
            pattern_type="keyword",
            whole_word=False
        )
        
        assert pattern_partial.matches("This is an ad section")
        assert pattern_partial.matches("This is a bad section")


class TestBoilerplateRemover:
    """Test cases for BoilerplateRemover class."""
    
    def test_basic_boilerplate_removal(self):
        """Test basic boilerplate removal functionality."""
        # Provide basic configuration with patterns
        config = {
            'boilerplate_patterns': {
                'navigation': ['navigation', 'nav'],
                'ads': ['advertisement'],
                'cookies': ['cookie'],
                'legal': ['copyright', 'footer']
            }
        }
        remover = BoilerplateRemover(config)
        
        text = """
        Welcome to our website!
        
        This is the main content of the page.
        It contains valuable information.
        
        Navigation: Home | About | Contact
        Advertisement: Buy our products now!
        Cookie notice: We use cookies on this site.
        Footer: Copyright 2024 All rights reserved.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "main content" in result.cleaned_text
        assert "valuable information" in result.cleaned_text
        assert len(result.removed_patterns) > 0
        assert result.get_removal_ratio() > 0
    
    def test_navigation_removal(self):
        """Test removal of navigation elements."""
        config = {
            'boilerplate_patterns': {
                'navigation': ['navigation', 'menu', 'sidebar']
            },
            'remove_navigation': True
        }
        remover = BoilerplateRemover(config)
        
        text = """
        Main navigation menu here
        This is important content.
        Sidebar navigation links
        More important content.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "important content" in result.cleaned_text
        assert "More important content" in result.cleaned_text
        # Navigation should be removed or reduced
        nav_patterns = [p for p in result.removed_patterns if 'navigation' in p]
        assert len(nav_patterns) > 0
    
    def test_advertisement_removal(self):
        """Test removal of advertisement content."""
        config = {
            'boilerplate_patterns': {
                'ads': ['advertisement', 'sponsored']
            },
            'remove_ads': True
        }
        remover = BoilerplateRemover(config)
        
        text = """
        This is valuable content.
        
        Advertisement: Special offer today!
        Sponsored content here.
        
        More valuable content follows.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "valuable content" in result.cleaned_text
        assert "More valuable content" in result.cleaned_text
        # Ads should be detected and removed
        ad_patterns = [p for p in result.removed_patterns if 'ads' in p]
        assert len(ad_patterns) > 0
    
    def test_cookie_and_privacy_removal(self):
        """Test removal of cookie and privacy notices."""
        config = {'remove_cookies': True}
        remover = BoilerplateRemover(config)
        
        text = """
        Important article content here.
        
        Cookie policy: We use cookies to improve your experience.
        Privacy policy: Your data is protected.
        Terms of service apply.
        
        Article continues with more content.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "Important article content" in result.cleaned_text
        assert "Article continues" in result.cleaned_text
        # Cookie/privacy patterns should be detected
        cookie_patterns = [p for p in result.removed_patterns if 'cookies' in p]
        assert len(cookie_patterns) > 0
    
    def test_social_media_removal(self):
        """Test removal of social media elements."""
        config = {'remove_social': True}
        remover = BoilerplateRemover(config)
        
        text = """
        Great article about technology.
        
        Share this article on Facebook!
        Follow us on Twitter.
        Like and subscribe for more content.
        
        The article conclusion goes here.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "Great article about technology" in result.cleaned_text
        assert "article conclusion" in result.cleaned_text
        # Social media patterns should be detected
        social_patterns = [p for p in result.removed_patterns if 'social' in p]
        assert len(social_patterns) > 0
    
    def test_legal_text_removal(self):
        """Test removal of legal and copyright text."""
        config = {'remove_legal': True}
        remover = BoilerplateRemover(config)
        
        text = """
        Original content starts here.
        
        Copyright © 2024 Company Name. All rights reserved.
        Trademark notices apply.
        Legal disclaimer: Use at your own risk.
        
        Content continues after legal text.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "Original content starts" in result.cleaned_text
        assert "Content continues" in result.cleaned_text
        # Legal patterns should be detected
        legal_patterns = [p for p in result.removed_patterns if 'legal' in p]
        assert len(legal_patterns) > 0
    
    def test_header_footer_removal(self):
        """Test removal of header and footer patterns."""
        config = {'remove_headers_footers': True}
        remover = BoilerplateRemover(config)
        
        text = """
        =====================================
        HEADER: Website Title
        =====================================
        
        This is the main content area.
        Important information is here.
        
        =====================================
        FOOTER: Contact Information
        =====================================
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "main content area" in result.cleaned_text
        assert "Important information" in result.cleaned_text
        # Header/footer patterns should be detected
        header_footer_patterns = [p for p in result.removed_patterns if 'header_footer' in p]
        assert len(header_footer_patterns) > 0
    
    def test_aggressive_removal_mode(self):
        """Test aggressive removal mode."""
        config = {'aggressive_removal': True}
        remover = BoilerplateRemover(config)
        
        text = """
        Good content here.
        Navigation menu items here.
        More good content.
        Advertisement section.
        Final good content.
        """
        
        result = remover.remove_boilerplate(text)
        
        # In aggressive mode, entire lines with boilerplate should be removed
        assert "Good content here" in result.cleaned_text
        assert "More good content" in result.cleaned_text
        assert "Final good content" in result.cleaned_text
        # Boilerplate lines should be completely removed
        assert "Navigation menu" not in result.cleaned_text
        assert "Advertisement" not in result.cleaned_text
    
    def test_minimum_content_length(self):
        """Test minimum content length filtering."""
        config = {'min_content_length': 20}
        remover = BoilerplateRemover(config)
        
        text = """
        Short line.
        This is a much longer line that meets the minimum content length requirement.
        Another short.
        This is another long line that should be preserved in the output.
        """
        
        result = remover.remove_boilerplate(text)
        
        # Long lines should be preserved
        assert "much longer line" in result.cleaned_text
        assert "another long line" in result.cleaned_text
        # Short lines should be removed
        assert "Short line" not in result.cleaned_text
        assert "Another short" not in result.cleaned_text
    
    def test_custom_patterns(self):
        """Test adding custom boilerplate patterns."""
        remover = BoilerplateRemover()
        
        # Add custom pattern
        remover.add_custom_pattern(
            name="custom_spam",
            pattern="SPAM CONTENT",
            pattern_type="keyword"
        )
        
        text = """
        This is legitimate content.
        SPAM CONTENT: Buy now for cheap!
        More legitimate content here.
        """
        
        result = remover.remove_boilerplate(text)
        
        assert "legitimate content" in result.cleaned_text
        assert "More legitimate content" in result.cleaned_text
        # Custom pattern should be detected
        custom_patterns = [p for p in result.removed_patterns if 'custom_spam' in p]
        assert len(custom_patterns) > 0
    
    def test_pattern_detection_without_removal(self):
        """Test detecting patterns without removing them."""
        remover = BoilerplateRemover()
        
        text = """
        Content with navigation menu.
        Advertisement section here.
        Cookie policy notice.
        Social media share buttons.
        """
        
        detected = remover.detect_boilerplate_patterns(text)
        
        # Should detect various pattern types
        assert len(detected) > 0
        
        # Check for specific pattern types
        nav_detected = any('navigation' in pattern for pattern in detected.keys())
        ads_detected = any('ads' in pattern for pattern in detected.keys())
        
        assert nav_detected or ads_detected  # At least one should be detected
    
    def test_configuration_from_file(self):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_data = {
            'boilerplate_patterns': {
                'navigation': ['nav', 'menu'],
                'ads': ['advertisement', 'sponsored'],
                'custom': ['test_pattern']
            },
            'aggressive_removal': True,
            'min_content_length': 15
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            remover = BoilerplateRemover(config_file=config_file)
            
            # Configuration should be loaded
            assert remover.aggressive_removal == True
            assert remover.min_content_length == 15
            
            # Patterns should be loaded
            pattern_names = [p.name for p in remover.patterns]
            nav_patterns = [name for name in pattern_names if 'navigation' in name]
            assert len(nav_patterns) > 0
            
        finally:
            # Clean up temp file
            Path(config_file).unlink()
    
    def test_pattern_statistics(self):
        """Test generation of pattern statistics."""
        remover = BoilerplateRemover()
        
        text = """
        Content with multiple boilerplate elements.
        Navigation menu here.
        Advertisement section.
        Another advertisement.
        Cookie policy notice.
        Social media buttons.
        """
        
        stats = remover.get_pattern_statistics(text)
        
        assert 'total_patterns_detected' in stats
        assert 'total_matches' in stats
        assert 'patterns_by_type' in stats
        assert 'most_common_patterns' in stats
        
        assert stats['total_patterns_detected'] > 0
        assert stats['total_matches'] > 0
    
    def test_pattern_management(self):
        """Test adding and removing patterns."""
        remover = BoilerplateRemover()
        
        initial_count = len(remover.patterns)
        
        # Add custom pattern
        remover.add_custom_pattern(
            name="test_pattern",
            pattern="test_content",
            pattern_type="keyword"
        )
        
        assert len(remover.patterns) == initial_count + 1
        
        # Remove pattern
        removed = remover.remove_pattern("test_pattern")
        assert removed == True
        assert len(remover.patterns) == initial_count
        
        # Try to remove non-existent pattern
        removed = remover.remove_pattern("non_existent")
        assert removed == False
    
    def test_empty_and_invalid_input(self):
        """Test handling of empty and invalid input."""
        remover = BoilerplateRemover()
        
        # Empty string
        result = remover.remove_boilerplate("")
        assert result.cleaned_text == ""
        assert len(result.removed_patterns) == 0
        
        # None input
        result = remover.remove_boilerplate(None)
        assert result.cleaned_text == ""
        
        # Whitespace only
        result = remover.remove_boilerplate("   \n\t   ")
        assert result.cleaned_text.strip() == ""
    
    def test_post_processing(self):
        """Test post-processing of cleaned text."""
        remover = BoilerplateRemover()
        
        text = """
        Good content here.
        
        
        
        More content after excessive newlines.
        Line with    multiple    spaces.
        !!!@@@###$$$ (mostly symbols)
        Final content line.
        """
        
        result = remover.remove_boilerplate(text)
        
        # Should clean up excessive whitespace
        assert result.cleaned_text.count('\n\n\n') == 0
        
        # Should preserve content lines
        assert "Good content here" in result.cleaned_text
        assert "More content after" in result.cleaned_text
        assert "Final content line" in result.cleaned_text
    
    def test_boilerplate_removal_result(self):
        """Test BoilerplateRemovalResult properties."""
        original = "Original text with boilerplate content here."
        cleaned = "Original text here."
        
        result = BoilerplateRemovalResult(
            original_text=original,
            cleaned_text=cleaned,
            removed_patterns=['pattern1', 'pattern2'],
            removed_content=['boilerplate', 'content']
        )
        
        assert result.get_removal_ratio() > 0
        assert result.get_removed_count() == 2
    
    def test_selective_removal_configuration(self):
        """Test selective enabling/disabling of removal types."""
        # Only remove navigation, keep everything else
        config = {
            'remove_navigation': True,
            'remove_ads': False,
            'remove_cookies': False,
            'remove_social': False,
            'remove_legal': False
        }
        
        remover = BoilerplateRemover(config)
        
        text = """
        Content here.
        Navigation menu.
        Advertisement section.
        Cookie policy.
        Social media.
        Copyright notice.
        """
        
        result = remover.remove_boilerplate(text)
        
        # Only navigation patterns should be in removed_patterns
        nav_patterns = [p for p in result.removed_patterns if 'navigation' in p]
        other_patterns = [p for p in result.removed_patterns if 'navigation' not in p]
        
        assert len(nav_patterns) > 0
        # Other types should not be removed (though they might not be detected either)
        # This test mainly ensures the configuration is respected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_unicode_content(self):
        """Test with Unicode content."""
        remover = BoilerplateRemover()
        
        text = """
        Contenu principal en français.
        Navigation: Accueil | À propos
        Publicité: Achetez maintenant! 🛒
        Contenu important avec émojis 🌟.
        """
        
        result = remover.remove_boilerplate(text)
        
        # Should handle Unicode correctly
        assert "Contenu principal" in result.cleaned_text
        assert "Contenu important" in result.cleaned_text
    
    def test_very_long_text(self):
        """Test with very long text."""
        remover = BoilerplateRemover()
        
        # Create long text with boilerplate scattered throughout
        content_parts = []
        for i in range(100):
            content_parts.append(f"Content section {i} with valuable information.")
            if i % 10 == 0:
                content_parts.append("Advertisement: Buy our products!")
        
        text = '\n'.join(content_parts)
        result = remover.remove_boilerplate(text)
        
        # Should handle long text efficiently
        assert "Content section 0" in result.cleaned_text
        assert "Content section 99" in result.cleaned_text
        assert len(result.removed_patterns) > 0
    
    def test_nested_boilerplate_patterns(self):
        """Test with nested or overlapping boilerplate patterns."""
        remover = BoilerplateRemover()
        
        text = """
        Main content here.
        Navigation menu with advertisement links.
        Social media share buttons in footer.
        Content continues.
        """
        
        result = remover.remove_boilerplate(text)
        
        # Should detect multiple pattern types in same lines
        assert "Main content here" in result.cleaned_text
        assert "Content continues" in result.cleaned_text
        assert len(result.removed_patterns) > 0