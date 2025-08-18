# Utility Functions and Helpers

The utils module provides common utility functions and helper classes used throughout the QuData system.

## Components

### HTML Utilities (`html.py`)
- HTML parsing and cleaning functions
- Tag removal and content extraction
- HTML entity decoding and normalization
- Safe HTML processing with security considerations

### I/O Utilities (`io.py`)
- File system operations and path handling
- Safe file reading and writing with encoding detection
- Directory traversal and file discovery
- Temporary file management and cleanup

### Text Utilities (`text.py`)
- Text processing and normalization functions
- String manipulation and formatting helpers
- Encoding detection and conversion
- Text similarity and comparison utilities

## Usage Examples

### HTML Processing
```python
from qudata.utils.html import clean_html, extract_text, remove_tags

# Clean HTML content
clean_content = clean_html(html_string, remove_scripts=True)

# Extract plain text from HTML
text = extract_text(html_string, preserve_links=False)

# Remove specific tags
content = remove_tags(html_string, tags=['script', 'style', 'nav'])
```

### File I/O Operations
```python
from qudata.utils.io import safe_read_file, safe_write_file, detect_encoding

# Safe file reading with encoding detection
content = safe_read_file("document.txt")

# Write file with proper encoding
safe_write_file("output.txt", content, encoding="utf-8")

# Detect file encoding
encoding = detect_encoding("mystery_file.txt")
```

### Text Processing
```python
from qudata.utils.text import normalize_text, calculate_similarity, clean_whitespace

# Normalize text content
normalized = normalize_text(text, remove_accents=True, lowercase=True)

# Calculate text similarity
similarity = calculate_similarity(text1, text2, method="cosine")

# Clean whitespace and formatting
clean_text = clean_whitespace(text, normalize_spaces=True)
```

## HTML Utilities

### Functions Available
- `clean_html(html, remove_scripts=True, remove_styles=True)`: Clean HTML content
- `extract_text(html, preserve_links=False)`: Extract plain text from HTML
- `remove_tags(html, tags)`: Remove specific HTML tags
- `decode_entities(text)`: Decode HTML entities
- `is_valid_html(html)`: Validate HTML structure
- `extract_links(html)`: Extract all links from HTML
- `extract_images(html)`: Extract image information

### Security Features
- XSS prevention through tag sanitization
- Safe HTML parsing with error handling
- Malicious content detection and removal
- Configurable whitelist/blacklist for tags and attributes

## I/O Utilities

### File Operations
- `safe_read_file(path, encoding=None)`: Safe file reading with encoding detection
- `safe_write_file(path, content, encoding="utf-8")`: Safe file writing
- `detect_encoding(path)`: Automatic encoding detection
- `ensure_directory(path)`: Create directory if it doesn't exist
- `get_file_info(path)`: Get comprehensive file information
- `find_files(directory, pattern)`: Find files matching pattern

### Path Handling
- Cross-platform path normalization
- Safe path joining and validation
- Temporary file and directory management
- File extension detection and validation

## Text Utilities

### Text Processing
- `normalize_text(text, options)`: Comprehensive text normalization
- `clean_whitespace(text)`: Clean and normalize whitespace
- `remove_control_chars(text)`: Remove control characters
- `fix_encoding_issues(text)`: Fix common encoding problems
- `detect_language(text)`: Language detection
- `calculate_readability(text)`: Readability scoring

### Similarity and Comparison
- `calculate_similarity(text1, text2, method)`: Text similarity calculation
- `find_duplicates(texts, threshold)`: Duplicate detection
- `extract_keywords(text, count)`: Keyword extraction
- `tokenize_text(text, method)`: Text tokenization

## Configuration

Utilities can be configured through the main configuration:

```yaml
utils:
  html:
    remove_scripts: true
    remove_styles: true
    preserve_links: false
    allowed_tags: ["p", "div", "span", "h1", "h2", "h3"]
  io:
    default_encoding: "utf-8"
    temp_dir: "/tmp/qudata"
    max_file_size: "100MB"
  text:
    normalize_unicode: true
    remove_accents: false
    similarity_method: "cosine"
    min_similarity_threshold: 0.8
```

## Error Handling

All utility functions include comprehensive error handling:
- Graceful degradation for malformed input
- Detailed error logging and reporting
- Fallback mechanisms for encoding issues
- Safe defaults for missing or invalid parameters

## Performance Considerations

- Efficient algorithms for text processing
- Memory-conscious file handling for large files
- Caching for expensive operations
- Parallel processing where applicable
- Streaming support for large datasets