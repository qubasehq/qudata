# Data Formats and Export Guide

This guide covers the various input and output formats supported by QuData, including configuration options and best practices for each format.

## Table of Contents

- [Input Formats](#input-formats)
- [Output Formats](#output-formats)
- [Format Configuration](#format-configuration)
- [Quality Considerations](#quality-considerations)
- [Best Practices](#best-practices)

## Input Formats

QuData supports a wide variety of input formats for maximum flexibility in data ingestion.

### Document Formats

#### PDF Files
- **Extensions**: `.pdf`
- **Features**: Text extraction, table detection, image extraction, OCR for scanned PDFs
- **Limitations**: Complex layouts may affect extraction quality

```python
from qudata.ingest import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract("document.pdf")

print(f"Text content: {len(result.content)} characters")
print(f"Tables found: {len(result.tables)}")
print(f"Images found: {len(result.images)}")
```

**Configuration:**
```yaml
ingest:
  pdf:
    extract_tables: true
    extract_images: true
    ocr_fallback: true
    ocr_languages: ["eng", "spa"]
    preserve_layout: false
```

#### Microsoft Word Documents
- **Extensions**: `.docx`, `.doc`
- **Features**: Text extraction, formatting preservation, table extraction, embedded objects
- **Limitations**: Complex formatting may be simplified

```python
from qudata.ingest import DocumentExtractor

extractor = DocumentExtractor()
result = extractor.extract("document.docx")

print(f"Title: {result.metadata.title}")
print(f"Author: {result.metadata.author}")
print(f"Content: {result.content}")
```

**Configuration:**
```yaml
ingest:
  docx:
    preserve_formatting: true
    extract_tables: true
    extract_images: false
    include_headers_footers: false
```

#### HTML and Web Content
- **Extensions**: `.html`, `.htm`
- **Features**: Content extraction, link preservation, metadata extraction, boilerplate removal
- **Limitations**: JavaScript-generated content not supported

```python
from qudata.ingest import WebExtractor

extractor = WebExtractor()

# From file
result = extractor.extract("webpage.html")

# From URL
result = extractor.extract_from_url("https://example.com/article")

print(f"Title: {result.metadata.title}")
print(f"Clean content: {result.content}")
```

**Configuration:**
```yaml
ingest:
  html:
    remove_scripts: true
    remove_styles: true
    preserve_links: false
    extract_metadata: true
    readability_threshold: 0.7
```

#### Plain Text Files
- **Extensions**: `.txt`, `.md`, `.rst`
- **Features**: Direct text ingestion, encoding detection, metadata extraction from headers
- **Limitations**: No structural information

```python
from qudata.ingest import PlainTextExtractor

extractor = PlainTextExtractor()
result = extractor.extract("document.txt")

print(f"Encoding: {result.metadata.encoding}")
print(f"Content: {result.content}")
```

### Structured Data Formats

#### CSV Files
- **Extensions**: `.csv`
- **Features**: Column-based data extraction, header detection, data type inference
- **Use Cases**: Tabular data, survey responses, structured datasets

```python
from qudata.ingest import StructuredExtractor

extractor = StructuredExtractor()
result = extractor.extract("data.csv")

# Access structured data
for row in result.structured_data:
    print(f"Row: {row}")
```

**Configuration:**
```yaml
ingest:
  csv:
    delimiter: ","
    quote_char: '"'
    encoding: "utf-8"
    skip_blank_lines: true
    infer_types: true
```

#### JSON Files
- **Extensions**: `.json`, `.jsonl`
- **Features**: Hierarchical data extraction, schema inference, nested object handling
- **Use Cases**: API responses, configuration files, structured datasets

```python
result = extractor.extract("data.json")

# Access JSON structure
json_data = result.structured_data
print(f"Keys: {list(json_data.keys())}")
```

**Configuration:**
```yaml
ingest:
  json:
    flatten_nested: false
    max_depth: 10
    extract_text_fields: true
    text_field_patterns: ["content", "text", "description"]
```

### Image and OCR Formats

#### Image Files
- **Extensions**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`
- **Features**: OCR text extraction, image preprocessing, confidence scoring
- **Use Cases**: Scanned documents, screenshots, diagrams with text

```python
from qudata.ingest import OCRProcessor

processor = OCRProcessor()
result = processor.extract("scanned_document.png")

print(f"Extracted text: {result.content}")
print(f"OCR confidence: {result.metadata.ocr_confidence}")
```

**Configuration:**
```yaml
ingest:
  ocr:
    languages: ["eng", "spa", "fra"]
    confidence_threshold: 0.8
    preprocessing:
      enhance_contrast: true
      remove_noise: true
      deskew: true
    tesseract_config: "--psm 6"
```

### Database Sources

#### SQL Databases
- **Supported**: PostgreSQL, MySQL, SQLite
- **Features**: Query-based extraction, schema introspection, batch processing
- **Use Cases**: Content management systems, application databases

```python
from qudata.ingest import DatabaseExtractor

extractor = DatabaseExtractor()
result = extractor.extract_from_query(
    connection_string="postgresql://user:pass@localhost/db",
    query="SELECT title, content FROM articles WHERE published = true"
)
```

**Configuration:**
```yaml
ingest:
  database:
    connections:
      - name: "content_db"
        type: "postgresql"
        host: "localhost"
        database: "content"
        username: "user"
        password: "password"
    batch_size: 1000
    timeout: 30
```

#### NoSQL Databases
- **Supported**: MongoDB, Elasticsearch
- **Features**: Document-based extraction, flexible schema handling
- **Use Cases**: Content repositories, search indexes

```python
result = extractor.extract_from_mongodb(
    connection_string="mongodb://localhost:27017/content",
    collection="articles",
    query={"status": "published"}
)
```

### Web Sources

#### Web Scraping
- **Features**: URL-based extraction, sitemap crawling, rate limiting
- **Use Cases**: News articles, blog posts, documentation sites

```python
from qudata.ingest import WebScraper

scraper = WebScraper()

# Single URL
result = scraper.scrape_url("https://example.com/article")

# Multiple URLs
urls = ["https://example.com/page1", "https://example.com/page2"]
results = scraper.scrape_urls(urls)

# Sitemap crawling
results = scraper.scrape_sitemap("https://example.com/sitemap.xml")
```

**Configuration:**
```yaml
ingest:
  web_scraping:
    rate_limit: 10  # requests per second
    timeout: 30
    user_agent: "QuData/1.0"
    respect_robots_txt: true
    max_pages: 1000
```

#### RSS Feeds
- **Features**: Feed parsing, content extraction, update tracking
- **Use Cases**: News feeds, blog updates, content syndication

```python
from qudata.ingest import RSSExtractor

extractor = RSSExtractor()
result = extractor.extract_feed("https://example.com/feed.xml")

for item in result.items:
    print(f"Title: {item.title}")
    print(f"Content: {item.content}")
```

## Output Formats

QuData generates training-ready datasets in multiple formats optimized for different use cases.

### JSONL Format

JSON Lines format for general-purpose LLM training.

**Structure:**
```json
{"text": "Document content", "metadata": {"source": "file.pdf", "quality": 0.85}, "labels": ["category1"]}
{"text": "Another document", "metadata": {"source": "file2.pdf", "quality": 0.92}, "labels": ["category2"]}
```

**Usage:**
```python
from qudata.pack import JSONLFormatter

formatter = JSONLFormatter()
formatter.export_to_file(
    documents=processed_docs,
    output_path="training_data.jsonl",
    fields=["text", "metadata", "quality_score", "labels"]
)
```

**Configuration:**
```yaml
export:
  jsonl:
    fields: ["text", "metadata", "quality_score", "labels"]
    filter_low_quality: true
    min_quality_score: 0.7
    max_tokens_per_line: 8192
    include_empty_lines: false
```

### ChatML Format

Conversational format optimized for chat model training.

**Structure:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
  ],
  "metadata": {
    "source": "ml_textbook.pdf",
    "quality_score": 0.89,
    "topics": ["machine learning", "AI"]
  }
}
```

**Usage:**
```python
from qudata.pack import ChatMLFormatter

formatter = ChatMLFormatter()
chatml_data = formatter.format_documents(
    documents=processed_docs,
    system_message="You are a helpful assistant.",
    include_metadata=True
)
```

**Configuration:**
```yaml
export:
  chatml:
    system_message: "You are a helpful assistant."
    include_metadata: true
    max_tokens_per_message: 4096
    conversation_format: "qa"  # or "instruction", "dialogue"
```

### Parquet Format

Columnar format optimized for analytics and large-scale processing.

**Structure:**
- Efficient storage and querying
- Schema evolution support
- Compression and encoding optimizations

**Usage:**
```python
from qudata.pack import ParquetFormatter

formatter = ParquetFormatter()
formatter.export_to_file(
    documents=processed_docs,
    output_path="training_data.parquet",
    compression="snappy"
)
```

**Configuration:**
```yaml
export:
  parquet:
    compression: "snappy"  # or "gzip", "lz4", "brotli"
    row_group_size: 50000
    page_size: 1024
    schema_validation: true
```

### Plain Text Format

Human-readable format for inspection and debugging.

**Structure:**
```
=== Document 1 ===
Source: document.pdf
Quality: 0.85
Topics: technology, AI
Language: en

Content goes here...

---

=== Document 2 ===
...
```

**Usage:**
```python
from qudata.pack import PlainTextFormatter

formatter = PlainTextFormatter()
text_output = formatter.format_documents(
    documents=processed_docs,
    separator="\n---\n",
    include_headers=True
)
```

**Configuration:**
```yaml
export:
  plain_text:
    separator: "\n---\n"
    include_headers: true
    include_metadata: true
    max_line_length: 120
    wrap_text: true
```

### Custom Formats

Create custom export formats for specific requirements.

```python
from qudata.pack import BaseFormatter

class CustomFormatter(BaseFormatter):
    def format_document(self, document):
        return {
            "id": document.id,
            "content": document.content,
            "custom_field": self.extract_custom_data(document)
        }
    
    def export_to_file(self, documents, output_path):
        formatted_data = [self.format_document(doc) for doc in documents]
        # Custom export logic here
```

## Format Configuration

### Global Export Settings

```yaml
export:
  output_dir: "/data/processed"
  formats: ["jsonl", "chatml", "parquet"]
  
  # Quality filtering
  filter_low_quality: true
  min_quality_score: 0.7
  
  # Dataset splitting
  splits:
    enabled: true
    ratios: [0.8, 0.1, 0.1]  # train, validation, test
    stratify_by: "domain"
    random_seed: 42
  
  # Metadata inclusion
  include_metadata: true
  metadata_fields: ["source", "quality_score", "language", "topics"]
```

### Format-Specific Settings

#### JSONL Configuration
```yaml
export:
  jsonl:
    fields: ["text", "metadata", "quality_score"]
    filter_empty: true
    max_tokens_per_line: 8192
    encoding: "utf-8"
    compression: "gzip"
```

#### ChatML Configuration
```yaml
export:
  chatml:
    system_message: "You are a helpful assistant."
    conversation_format: "qa"
    include_metadata: true
    max_tokens_per_message: 4096
    role_mapping:
      instruction: "user"
      response: "assistant"
```

#### Parquet Configuration
```yaml
export:
  parquet:
    compression: "snappy"
    row_group_size: 50000
    page_size: 1024
    use_dictionary: true
    write_statistics: true
```

## Quality Considerations

### Input Quality Factors

#### PDF Quality
- **Text-based PDFs**: Highest quality, direct text extraction
- **Scanned PDFs**: OCR-dependent, may have errors
- **Complex layouts**: Tables and multi-column layouts may be challenging

#### Web Content Quality
- **Clean articles**: High quality with proper content extraction
- **Complex pages**: Navigation and ads may affect quality
- **Dynamic content**: JavaScript-generated content not captured

#### OCR Quality
- **Image resolution**: Higher resolution improves accuracy
- **Text clarity**: Clear, high-contrast text works best
- **Language support**: Accuracy varies by language

### Output Quality Control

#### Quality Scoring
```python
# Check quality distribution
quality_scores = [doc.quality_score for doc in documents]
print(f"Average quality: {sum(quality_scores) / len(quality_scores):.2f}")

# Filter by quality
high_quality = [doc for doc in documents if doc.quality_score >= 0.8]
print(f"High quality documents: {len(high_quality)}")
```

#### Content Validation
```python
from qudata.validation import DatasetValidator

validator = DatasetValidator()
result = validator.validate_dataset(documents)

if not result.is_valid:
    for issue in result.issues:
        print(f"{issue.severity}: {issue.message}")
```

## Best Practices

### Input Processing

1. **Format Selection**
   - Prefer text-based formats over image-based when possible
   - Use structured formats (JSON, CSV) for tabular data
   - Consider OCR quality for scanned documents

2. **Quality Preprocessing**
   - Clean HTML content before processing
   - Validate structured data schemas
   - Check encoding for text files

3. **Batch Processing**
   - Process similar formats together
   - Use parallel processing for large datasets
   - Monitor memory usage with large files

### Output Generation

1. **Format Choice**
   - Use JSONL for general training datasets
   - Use ChatML for conversational models
   - Use Parquet for analytics and large datasets

2. **Quality Control**
   - Set appropriate quality thresholds
   - Review sample outputs manually
   - Validate export formats

3. **Dataset Management**
   - Use consistent naming conventions
   - Version control configuration files
   - Document format specifications

### Performance Optimization

1. **Memory Management**
   ```yaml
   performance:
     streaming_mode: true
     batch_size: 100
     max_memory_usage: "8GB"
   ```

2. **Parallel Processing**
   ```yaml
   ingest:
     parallel_processing: true
     max_workers: 8
   ```

3. **Caching**
   ```yaml
   performance:
     caching:
       enabled: true
       cache_dir: "/tmp/qudata_cache"
   ```

### Troubleshooting

#### Common Issues

1. **Encoding Problems**
   ```python
   # Check for encoding issues
   try:
       content = file.read().decode('utf-8')
   except UnicodeDecodeError:
       # Try alternative encodings
       content = file.read().decode('latin-1')
   ```

2. **Memory Issues**
   ```python
   # Monitor memory usage
   import psutil
   memory_percent = psutil.virtual_memory().percent
   if memory_percent > 80:
       print("Warning: High memory usage")
   ```

3. **Quality Issues**
   ```python
   # Investigate low quality scores
   low_quality = [doc for doc in documents if doc.quality_score < 0.5]
   for doc in low_quality[:5]:
       print(f"File: {doc.source_path}")
       print(f"Issues: {doc.quality_issues}")
   ```

This guide provides comprehensive coverage of all supported formats and their optimal usage patterns. For specific format requirements or custom implementations, refer to the API documentation and configuration examples.