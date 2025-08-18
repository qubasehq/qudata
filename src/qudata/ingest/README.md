# Data Ingestion Module

The ingestion module provides comprehensive data extraction capabilities from multiple sources and formats. It serves as the entry point for all data processing pipelines in QuData.

## Overview

This module handles:
- **File Format Detection**: Automatic detection of file types using signatures and extensions
- **Multi-Format Extraction**: Support for PDF, DOCX, HTML, TXT, CSV, JSON, and more
- **Web Scraping**: Intelligent content extraction from web pages with rate limiting
- **API Integration**: REST and GraphQL endpoint connectivity
- **Database Extraction**: SQL and NoSQL database connectivity
- **OCR Processing**: Text extraction from images and scanned documents
- **Streaming Data**: Real-time processing of RSS feeds, logs, and Kafka streams

## Core Components

### File Type Detection

```python
from qudata.ingest import FileTypeDetector

detector = FileTypeDetector()
file_type = detector.detect_file_type("document.pdf")
# Returns: "pdf"
```

**Supported Formats:**
- Text: `.txt`, `.md`, `.rst`
- Documents: `.pdf`, `.docx`, `.odt`, `.rtf`
- Web: `.html`, `.htm`, `.xml`
- Structured: `.csv`, `.json`, `.jsonl`, `.yaml`
- Archives: `.zip`, `.tar`, `.gz`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff` (for OCR)

### Text Extractors

#### Plain Text Extractor
```python
from qudata.ingest import PlainTextExtractor

extractor = PlainTextExtractor()
result = extractor.extract("document.txt")
print(result.content)  # Extracted text content
```

#### PDF Extractor
```python
from qudata.ingest import PDFExtractor

extractor = PDFExtractor({
    'extract_tables': True,
    'preserve_layout': True
})
result = extractor.extract("document.pdf")
print(result.content)  # Text content
print(result.tables)   # Extracted tables
```

**Features:**
- Text extraction with layout preservation
- Table detection and extraction
- Metadata extraction (author, creation date, etc.)
- Handling of password-protected PDFs
- Error recovery for corrupted files

#### Document Extractor (DOCX/ODT)
```python
from qudata.ingest import DocumentExtractor

extractor = DocumentExtractor({
    'extract_images': True,
    'preserve_formatting': True
})
result = extractor.extract("document.docx")
```

**Features:**
- Text extraction with formatting preservation
- Table and image extraction
- Header/footer handling
- Comment and revision extraction
- Style information preservation

#### Web Extractor
```python
from qudata.ingest import WebExtractor

extractor = WebExtractor({
    'remove_navigation': True,
    'extract_main_content': True
})
result = extractor.extract("page.html")
```

**Features:**
- Main content extraction using readability algorithms
- Navigation and boilerplate removal
- Link and image extraction
- Metadata extraction from HTML meta tags
- Support for various encodings

#### Structured Data Extractor
```python
from qudata.ingest import StructuredExtractor

extractor = StructuredExtractor()
result = extractor.extract("data.csv")
# Handles CSV, JSON, JSONL, YAML formats
```

**Features:**
- Automatic delimiter detection for CSV
- Nested JSON structure handling
- Schema inference and validation
- Error handling for malformed data
- Large file streaming support

### Web Scraping

```python
from qudata.ingest import WebScraper, RateLimiter

scraper = WebScraper({
    'rate_limiter': RateLimiter(requests_per_minute=60),
    'cache_responses': True,
    'extract_links': True
})

content = scraper.scrape_url("https://example.com")
```

**Features:**
- Respectful crawling with rate limiting
- Response caching and deduplication
- JavaScript rendering support (optional)
- Sitemap parsing
- Robot.txt compliance
- Content extraction with readability

### API Integration

```python
from qudata.ingest import APIClient, AuthConfig

auth = AuthConfig(
    auth_type="bearer",
    token="your-api-token"
)

client = APIClient(auth_config=auth)
data = client.get("https://api.example.com/data")
```

**Supported Authentication:**
- Bearer tokens
- API keys
- Basic authentication
- OAuth 2.0
- Custom headers

**Features:**
- REST and GraphQL support
- Automatic pagination handling
- Rate limiting and retry logic
- Response caching
- Error handling and logging

### OCR Processing

```python
from qudata.ingest import OCRProcessor, ImagePreprocessor

preprocessor = ImagePreprocessor()
ocr = OCRProcessor({
    'language': 'eng',
    'confidence_threshold': 0.6
})

# Process scanned document
result = ocr.extract_from_image("scanned_page.png")
print(result.text)
print(result.confidence)
```

**Features:**
- Image preprocessing for better OCR accuracy
- Multi-language support
- Confidence scoring
- Batch processing
- PDF with embedded images support
- Error handling for poor quality images

### Streaming Data Processing

```python
from qudata.ingest import StreamProcessor, RSSFeedReader

# RSS Feed processing
rss_reader = RSSFeedReader({
    'update_interval': 3600,  # 1 hour
    'max_entries': 100
})

for item in rss_reader.read_feed("https://example.com/feed.xml"):
    print(item.title, item.content)
```

**Supported Streams:**
- RSS/Atom feeds
- Log files (various formats)
- Kafka streams
- Database change streams
- File system monitoring

## Configuration

### Basic Configuration

```yaml
# configs/ingest.yaml
ingest:
  file_detection:
    use_magic_numbers: true
    fallback_to_extension: true
  
  pdf:
    extract_tables: true
    preserve_layout: true
    password_attempts: 3
  
  web:
    user_agent: "QuData/1.0"
    timeout: 30
    max_redirects: 5
  
  ocr:
    language: "eng"
    confidence_threshold: 0.6
    preprocessing: true
```

### Advanced Configuration

```python
from qudata.ingest import FileTypeDetector

# Custom file type detection
detector = FileTypeDetector({
    'custom_extensions': {
        '.myformat': 'custom_format'
    },
    'magic_number_overrides': {
        b'\x50\x4B': 'zip'  # Custom ZIP detection
    }
})
```

## Error Handling

The ingestion module provides comprehensive error handling:

```python
from qudata.ingest import PDFExtractor
from qudata.models import ProcessingError

extractor = PDFExtractor()
try:
    result = extractor.extract("corrupted.pdf")
except ProcessingError as e:
    print(f"Error: {e.message}")
    print(f"Stage: {e.stage}")
    print(f"Severity: {e.severity}")
```

**Error Types:**
- `FileNotFoundError`: File doesn't exist
- `UnsupportedFormatError`: Format not supported
- `CorruptedFileError`: File is corrupted or unreadable
- `AuthenticationError`: API authentication failed
- `RateLimitError`: Rate limit exceeded
- `NetworkError`: Network connectivity issues

## Performance Considerations

### Memory Management
- Large files are processed in chunks
- Streaming support for CSV and JSON files
- Configurable batch sizes for bulk processing

### Parallel Processing
```python
from qudata.ingest import FileTypeDetector
import concurrent.futures

detector = FileTypeDetector()
files = ["file1.pdf", "file2.docx", "file3.html"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(detector.detect_file_type, files))
```

### Caching
- Response caching for web scraping
- File signature caching for repeated detection
- Configurable cache TTL and size limits

## Examples

### Basic File Processing
```python
from qudata.ingest import FileTypeDetector, PDFExtractor, DocumentExtractor

detector = FileTypeDetector()
extractors = {
    'pdf': PDFExtractor(),
    'docx': DocumentExtractor()
}

def process_file(file_path):
    file_type = detector.detect_file_type(file_path)
    extractor = extractors.get(file_type)
    
    if extractor:
        return extractor.extract(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Process a file
result = process_file("document.pdf")
print(result.content)
```

### Batch Web Scraping
```python
from qudata.ingest import WebScraper, RateLimiter

scraper = WebScraper({
    'rate_limiter': RateLimiter(requests_per_minute=30),
    'cache_responses': True
})

urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

for url in urls:
    try:
        content = scraper.scrape_url(url)
        print(f"Scraped {len(content.text)} characters from {url}")
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
```

### API Data Collection
```python
from qudata.ingest import APIClient, AuthConfig

auth = AuthConfig(auth_type="bearer", token="your-token")
client = APIClient(auth_config=auth)

# Paginated API data collection
all_data = []
page = 1

while True:
    response = client.get(f"https://api.example.com/data?page={page}")
    data = response.json()
    
    if not data.get('results'):
        break
        
    all_data.extend(data['results'])
    page += 1

print(f"Collected {len(all_data)} records")
```

## Testing

The module includes comprehensive tests:

```bash
# Run ingestion tests
pytest tests/unit/test_*_extraction.py
pytest tests/integration/test_ingestion_integration.py

# Run specific extractor tests
pytest tests/unit/test_pdf_extraction.py -v
pytest tests/unit/test_web_scraping.py -v
```

## Dependencies

**Core Dependencies:**
- `pdfplumber`: PDF text extraction
- `python-docx`: DOCX document processing
- `beautifulsoup4`: HTML parsing and web scraping
- `requests`: HTTP client for APIs and web scraping
- `pytesseract`: OCR processing
- `opencv-python`: Image preprocessing
- `feedparser`: RSS/Atom feed parsing

**Optional Dependencies:**
- `scrapy`: Advanced web scraping (install with `pip install qudata[web]`)
- `selenium`: JavaScript rendering (install with `pip install qudata[web]`)
- `kafka-python`: Kafka stream processing

## Troubleshooting

### Common Issues

**PDF Extraction Fails:**
```python
# Check if PDF is password protected
from qudata.ingest import PDFExtractor

extractor = PDFExtractor({'password_attempts': 3})
try:
    result = extractor.extract("protected.pdf")
except ProcessingError as e:
    if "password" in e.message.lower():
        print("PDF is password protected")
```

**OCR Low Accuracy:**
```python
# Enable image preprocessing
from qudata.ingest import OCRProcessor, ImagePreprocessor

preprocessor = ImagePreprocessor({
    'enhance_contrast': True,
    'denoise': True,
    'deskew': True
})

ocr = OCRProcessor({
    'preprocessor': preprocessor,
    'confidence_threshold': 0.8
})
```

**Web Scraping Blocked:**
```python
# Use custom user agent and delays
from qudata.ingest import WebScraper, RateLimiter

scraper = WebScraper({
    'user_agent': 'Mozilla/5.0 (compatible; QuData/1.0)',
    'rate_limiter': RateLimiter(
        requests_per_minute=10,
        delay_between_requests=2
    )
})
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `detector.py`: File type detection
- `files.py`: Plain text extraction
- `pdf.py`: PDF processing
- `document.py`: Document format processing
- `web.py`: Web content extraction
- `structured.py`: Structured data parsing
- `scraper.py`: Web scraping
- `api.py`: API integration
- `ocr.py`: OCR processing
- `stream.py`: Streaming data processing