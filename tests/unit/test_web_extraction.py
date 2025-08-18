"""
Unit tests for WebExtractor (HTML files).

Tests the extraction of content, tables, images, and structure from HTML files
with various formatting and embedded objects using BeautifulSoup4 and readability.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.qudata.ingest.web import WebExtractor
from src.qudata.models import ProcessingError, ErrorSeverity


class TestWebExtractor:
    """Test cases for WebExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = WebExtractor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_html(self, content: str, filename: str = "test.html") -> str:
        """Create a temporary HTML file with given content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_supports_format(self):
        """Test format support detection."""
        assert self.extractor.supports_format('html')
        assert self.extractor.supports_format('htm')
        assert self.extractor.supports_format('xhtml')
        assert self.extractor.supports_format('HTML')
        assert not self.extractor.supports_format('pdf')
        assert not self.extractor.supports_format('txt')
    
    def test_basic_html_extraction(self):
        """Test basic HTML content extraction."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Document</title>
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a test paragraph with some content.</p>
            <h2>Subheading</h2>
            <p>Another paragraph with more content.</p>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "Test Document" in result.content
        assert "Main Heading" in result.content
        assert "test paragraph" in result.content
        assert "Subheading" in result.content
        
        # Check structure
        assert result.structure is not None
        assert len(result.structure.headings) >= 2
        assert result.structure.paragraphs >= 2
    
    def test_html_with_tables(self):
        """Test HTML table extraction."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Table Test</title>
        </head>
        <body>
            <h1>Data Table</h1>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Age</th>
                        <th>City</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>John</td>
                        <td>25</td>
                        <td>New York</td>
                    </tr>
                    <tr>
                        <td>Jane</td>
                        <td>30</td>
                        <td>London</td>
                    </tr>
                </tbody>
            </table>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert len(result.tables) == 1
        
        table = result.tables[0]
        assert table.headers == ['Name', 'Age', 'City']
        assert len(table.rows) == 2
        assert table.rows[0] == ['John', '25', 'New York']
        assert table.rows[1] == ['Jane', '30', 'London']
    
    def test_html_with_images(self):
        """Test HTML image metadata extraction."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Test</title>
        </head>
        <body>
            <h1>Images</h1>
            <img src="image1.jpg" alt="First image" width="300" height="200">
            <img src="image2.png" alt="Second image" title="A nice picture">
            <img src="https://example.com/image3.gif" alt="Remote image">
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert len(result.images) == 3
        
        # Check first image
        img1 = result.images[0]
        assert "image1.jpg" in img1.path
        assert img1.alt_text == "First image"
        assert img1.width == 300
        assert img1.height == 200
        
        # Check second image
        img2 = result.images[1]
        assert "image2.png" in img2.path
        assert img2.alt_text == "Second image"
        assert img2.caption == "A nice picture"
        
        # Check third image (remote)
        img3 = result.images[2]
        assert img3.path == "https://example.com/image3.gif"
        assert img3.alt_text == "Remote image"
    
    def test_html_with_lists(self):
        """Test HTML list extraction."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>List Test</title>
        </head>
        <body>
            <h1>Lists</h1>
            <ul>
                <li>First item</li>
                <li>Second item</li>
                <li>Third item</li>
            </ul>
            <ol>
                <li>Numbered first</li>
                <li>Numbered second</li>
            </ol>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "- First item" in result.content
        assert "- Second item" in result.content
        assert "1. Numbered first" in result.content
        assert "2. Numbered second" in result.content
        
        # Check structure
        assert result.structure.lists >= 5  # 3 + 2 list items
    
    def test_html_noise_removal(self):
        """Test removal of noise elements."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Noise Test</title>
            <script>alert('This should be removed');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav>Navigation menu</nav>
            <header>Site header</header>
            <main>
                <h1>Main Content</h1>
                <p>This is the actual content.</p>
                <div class="ads">Advertisement content</div>
                <div id="social-share">Share buttons</div>
            </main>
            <aside class="sidebar">Sidebar content</aside>
            <footer>Site footer</footer>
            <!-- This is a comment -->
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "Main Content" in result.content
        assert "actual content" in result.content
        
        # These should be removed
        assert "alert(" not in result.content
        assert "color: red" not in result.content
        assert "Navigation menu" not in result.content
        assert "Site header" not in result.content
        assert "Advertisement content" not in result.content
        assert "Share buttons" not in result.content
        assert "Sidebar content" not in result.content
        assert "Site footer" not in result.content
        assert "This is a comment" not in result.content
    
    def test_html_with_code_blocks(self):
        """Test HTML code block extraction."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Test</title>
        </head>
        <body>
            <h1>Code Examples</h1>
            <p>Here's some inline <code>code</code> in a paragraph.</p>
            <pre>
def hello_world():
    print("Hello, World!")
            </pre>
            <blockquote>
                This is a quoted text.
                It spans multiple lines.
            </blockquote>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "`code`" in result.content
        assert "```" in result.content
        assert "def hello_world():" in result.content
        assert "> This is a quoted text." in result.content
        
        # Check structure
        assert result.structure.code_blocks >= 1
    
    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        html_content = """
        <html>
        <head>
            <title>Malformed Test
        </head>
        <body>
            <h1>Missing closing tag
            <p>Unclosed paragraph
            <div>Nested without closing
                <span>More nesting
        </body>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        # Should still extract content despite malformed HTML
        assert result is not None
        assert "Malformed Test" in result.content
        assert "Missing closing tag" in result.content
        assert "Unclosed paragraph" in result.content
    
    def test_empty_html(self):
        """Test handling of empty HTML."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title></title>
        </head>
        <body>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert result.content.strip() == "" or "Untitled Document" in result.content
        assert result.structure.paragraphs == 0
        assert result.structure.headings == [] or result.structure.headings == ["Untitled Document"]
    
    def test_large_file_rejection(self):
        """Test rejection of files that are too large."""
        # Create extractor with small max file size
        extractor = WebExtractor({'max_file_size': 100})  # 100 bytes
        
        # Create large HTML content
        large_content = "<html><body>" + "x" * 200 + "</body></html>"
        file_path = self.create_temp_html(large_content)
        
        with pytest.raises(ProcessingError) as exc_info:
            extractor.extract(file_path)
        
        assert exc_info.value.error_type == "FileTooLarge"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with pytest.raises(ProcessingError) as exc_info:
            self.extractor.extract("nonexistent.html")
        
        assert exc_info.value.error_type == "FileNotFound"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_permission_error(self):
        """Test handling of permission errors."""
        file_path = self.create_temp_html("<html><body>Test</body></html>")
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ProcessingError) as exc_info:
                self.extractor.extract(file_path)
            
            assert exc_info.value.error_type == "PermissionError"
            assert exc_info.value.severity == ErrorSeverity.HIGH
    
    def test_encoding_error(self):
        """Test handling of encoding errors."""
        # Create file with invalid UTF-8
        file_path = os.path.join(self.temp_dir, "invalid_encoding.html")
        with open(file_path, 'wb') as f:
            f.write(b'<html><body>\xff\xfe Invalid UTF-8</body></html>')
        
        # Should handle encoding error gracefully
        result = self.extractor.extract(file_path)
        assert result is not None
        # Content might be garbled but extraction should not fail
    
    def test_configuration_options(self):
        """Test various configuration options."""
        config = {
            'extract_tables': False,
            'extract_images': False,
            'use_readability': False,
            'preserve_links': False,
            'remove_scripts': False,
            'remove_styles': False
        }
        
        extractor = WebExtractor(config)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Config Test</title>
            <script>var x = 1;</script>
            <style>body { margin: 0; }</style>
        </head>
        <body>
            <h1>Test</h1>
            <p>Content with <a href="http://example.com">link</a></p>
            <table><tr><td>Table data</td></tr></table>
            <img src="test.jpg" alt="Test image">
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = extractor.extract(file_path)
        
        assert result is not None
        assert len(result.tables) == 0  # Tables disabled
        assert len(result.images) == 0  # Images disabled
        
        # Scripts and styles should be preserved
        assert "var x = 1;" in result.content or "margin: 0;" in result.content
    
    @patch('src.qudata.ingest.web.HAS_READABILITY', False)
    def test_without_readability(self):
        """Test extraction without readability library."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Readability Test</title>
        </head>
        <body>
            <h1>Main Content</h1>
            <p>This should still be extracted.</p>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "Main Content" in result.content
        assert "still be extracted" in result.content
    
    def test_table_without_headers(self):
        """Test table extraction without explicit headers."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <table>
                <tr>
                    <td>Row 1, Col 1</td>
                    <td>Row 1, Col 2</td>
                </tr>
                <tr>
                    <td>Row 2, Col 1</td>
                    <td>Row 2, Col 2</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert len(result.tables) == 1
        
        table = result.tables[0]
        assert table.headers == []  # No headers
        assert len(table.rows) == 2
        assert table.rows[0] == ['Row 1, Col 1', 'Row 1, Col 2']
    
    def test_table_with_caption(self):
        """Test table extraction with caption."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <table>
                <caption>Sales Data</caption>
                <tr>
                    <th>Product</th>
                    <th>Sales</th>
                </tr>
                <tr>
                    <td>Widget A</td>
                    <td>100</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert len(result.tables) == 1
        
        table = result.tables[0]
        assert table.caption == "Sales Data"
        assert table.headers == ['Product', 'Sales']
        assert table.rows == [['Widget A', '100']]
    
    def test_complex_nested_structure(self):
        """Test extraction from complex nested HTML structure."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Complex Structure</title>
        </head>
        <body>
            <article>
                <header>
                    <h1>Article Title</h1>
                    <p>Article subtitle</p>
                </header>
                <section>
                    <h2>Section 1</h2>
                    <div>
                        <p>Paragraph in div</p>
                        <ul>
                            <li>List item 1</li>
                            <li>List item 2</li>
                        </ul>
                    </div>
                </section>
                <section>
                    <h2>Section 2</h2>
                    <blockquote>
                        <p>Quoted paragraph</p>
                    </blockquote>
                </section>
            </article>
        </body>
        </html>
        """
        
        file_path = self.create_temp_html(html_content)
        result = self.extractor.extract(file_path)
        
        assert result is not None
        assert "Article Title" in result.content
        assert "Section 1" in result.content
        assert "Section 2" in result.content
        assert "Paragraph in div" in result.content
        assert "- List item 1" in result.content
        assert "> Quoted paragraph" in result.content
        
        # Check structure counts
        assert result.structure.paragraphs >= 3
        assert len(result.structure.headings) >= 3
        assert result.structure.lists >= 2


if __name__ == '__main__':
    pytest.main([__file__])