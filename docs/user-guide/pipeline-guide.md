# Pipeline Processing Guide

This guide covers how to use QuData's main processing pipeline to transform raw documents into high-quality training datasets.

## Overview

The QuData pipeline consists of several stages:

1. **Ingestion** - Extract content from various file formats
2. **Cleaning** - Remove noise, duplicates, and low-quality content
3. **Annotation** - Add metadata, categories, and entity tags
4. **Scoring** - Assess content quality and relevance
5. **Export** - Generate training-ready datasets in multiple formats

## Quick Start

### Basic Pipeline Usage

```python
from qudata import QuDataPipeline, load_config

# Load configuration
config = load_config("configs/pipeline.yaml")

# Initialize pipeline
pipeline = QuDataPipeline(config)

# Process a directory of documents
result = pipeline.process_directory(
    input_path="/path/to/raw/documents",
    output_path="/path/to/processed/data"
)

print(f"Processed {result.processed_count} documents")
print(f"Average quality score: {result.average_quality:.2f}")
```

### Processing Individual Files

```python
# Process a single file
result = pipeline.process_file("/path/to/document.pdf")

if result.success:
    document = result.document
    print(f"Title: {document.metadata.title}")
    print(f"Quality: {document.quality_score}")
    print(f"Categories: {document.metadata.topics}")
else:
    print("Processing failed:")
    for error in result.errors:
        print(f"  - {error.message}")
```

## Configuration

### Basic Configuration

Create a `pipeline.yaml` file:

```yaml
ingest:
  formats: ["pdf", "docx", "txt", "html"]
  max_file_size: "100MB"
  parallel_processing: true

clean:
  remove_duplicates: true
  min_quality_score: 0.7
  language_filter: ["en"]

annotate:
  enable_ner: true
  enable_classification: true

export:
  formats: ["jsonl", "chatml"]
  output_dir: "/data/processed"
```

### Advanced Configuration Options

#### Ingestion Settings

```yaml
ingest:
  formats: ["pdf", "docx", "txt", "html", "csv", "json"]
  max_file_size: "500MB"
  parallel_processing: true
  max_workers: 8
  
  # OCR settings for images and scanned PDFs
  ocr_enabled: true
  ocr_languages: ["eng", "spa"]
  ocr_confidence_threshold: 0.8
  
  # Web scraping settings
  web_scraping:
    enabled: true
    rate_limit: 10
    timeout: 30
    user_agent: "QuData/1.0"
```

#### Cleaning Settings

```yaml
clean:
  # Duplicate detection
  remove_duplicates: true
  similarity_threshold: 0.9
  
  # Quality filtering
  min_quality_score: 0.7
  max_quality_score: 1.0
  
  # Language filtering
  language_filter: ["en", "es", "fr"]
  language_confidence_threshold: 0.8
  
  # Content cleaning
  remove_boilerplate: true
  remove_html_tags: true
  normalize_whitespace: true
  
  # Text normalization
  unicode_normalization: "NFKC"
  remove_accents: false
  lowercase: false
```

#### Annotation Settings

```yaml
annotate:
  # Named Entity Recognition
  enable_ner: true
  ner_model: "en_core_web_sm"
  ner_confidence_threshold: 0.8
  
  # Content classification
  enable_classification: true
  taxonomy_file: "configs/taxonomy.yaml"
  
  # Topic modeling
  enable_topic_modeling: true
  topic_model: "bertopic"
  num_topics: 20
  
  # Quality scoring
  quality_scoring:
    weights:
      length: 0.2
      language: 0.3
      coherence: 0.3
      uniqueness: 0.2
```

#### Export Settings

```yaml
export:
  formats: ["jsonl", "chatml", "parquet"]
  output_dir: "/data/processed"
  
  # JSONL format settings
  jsonl:
    fields: ["content", "metadata", "quality_score"]
    filter_low_quality: true
    min_quality_score: 0.7
  
  # ChatML format settings
  chatml:
    system_message: "You are a helpful assistant."
    include_metadata: true
    max_tokens_per_message: 4096
  
  # Dataset splitting
  splits:
    enabled: true
    ratios: [0.8, 0.1, 0.1]  # train, val, test
    stratify_by: "domain"
    random_seed: 42
```

## Processing Stages

### 1. Ingestion Stage

The ingestion stage extracts content from various file formats:

#### Supported Formats

- **PDF**: Text extraction with table and image handling
- **DOCX**: Microsoft Word documents with formatting preservation
- **HTML**: Web pages with content extraction and cleaning
- **TXT/MD**: Plain text and Markdown files
- **CSV/JSON**: Structured data formats
- **Images**: OCR text extraction from images and scanned PDFs

#### Ingestion Examples

```python
from qudata.ingest import PDFExtractor, DocumentExtractor, WebExtractor

# Extract from PDF
pdf_extractor = PDFExtractor()
pdf_result = pdf_extractor.extract("/path/to/document.pdf")
print(f"Extracted {len(pdf_result.content)} characters")

# Extract from DOCX
doc_extractor = DocumentExtractor()
doc_result = doc_extractor.extract("/path/to/document.docx")
print(f"Title: {doc_result.metadata.title}")

# Extract from web page
web_extractor = WebExtractor()
web_result = web_extractor.extract_from_url("https://example.com/article")
print(f"Clean content: {web_result.content[:200]}...")
```

### 2. Cleaning Stage

The cleaning stage removes noise and improves content quality:

#### Cleaning Operations

- **Duplicate Removal**: Exact and near-duplicate detection
- **Boilerplate Removal**: Headers, footers, navigation elements
- **Language Filtering**: Keep only specified languages
- **Quality Filtering**: Remove low-quality content
- **Text Normalization**: Unicode, whitespace, encoding fixes

#### Cleaning Examples

```python
from qudata.clean import ComprehensiveCleaningPipeline

cleaner = ComprehensiveCleaningPipeline()

# Clean a single document
cleaned_doc = cleaner.clean_document(document)
print(f"Quality improved from {document.quality_score:.2f} to {cleaned_doc.quality_score:.2f}")

# Clean multiple documents
cleaned_docs = cleaner.clean_documents(document_list)
high_quality = [doc for doc in cleaned_docs if doc.quality_score >= 0.8]
```

### 3. Annotation Stage

The annotation stage adds metadata and semantic information:

#### Annotation Features

- **Named Entity Recognition**: People, organizations, locations
- **Content Classification**: Domain and topic categorization
- **Metadata Extraction**: Title, author, date, source
- **Quality Scoring**: Multi-dimensional quality assessment

#### Annotation Examples

```python
from qudata.annotate import TaxonomyClassifier, MetadataExtractor

# Classify content
classifier = TaxonomyClassifier()
categories = classifier.classify(document.content)
print(f"Categories: {categories}")

# Extract entities
extractor = MetadataExtractor()
entities = extractor.extract_entities(document.content)
for entity in entities:
    print(f"{entity.text} ({entity.label}): {entity.confidence:.2f}")
```

### 4. Export Stage

The export stage generates training-ready datasets:

#### Export Formats

- **JSONL**: JSON Lines format for general training
- **ChatML**: Conversational format for chat models
- **Parquet**: Columnar format for analytics
- **Plain Text**: Human-readable format for inspection

#### Export Examples

```python
from qudata.export import ContentSegmenter
from qudata.pack import JSONLFormatter, ChatMLFormatter

# Segment content for instruction tuning
segmenter = ContentSegmenter()
segments = segmenter.segment_document(document, format="instruction")

# Export to JSONL
jsonl_formatter = JSONLFormatter()
jsonl_formatter.export_to_file(
    documents=processed_docs,
    output_path="training_data.jsonl",
    fields=["content", "metadata", "quality_score"]
)

# Export to ChatML
chatml_formatter = ChatMLFormatter()
chatml_data = chatml_formatter.format_documents(
    processed_docs,
    system_message="You are a helpful assistant."
)
```

## Monitoring and Quality Control

### Quality Metrics

Monitor processing quality with built-in metrics:

```python
# Check pipeline results
result = pipeline.process_directory("/data/raw")

print(f"Processing Statistics:")
print(f"  Total files: {result.total_count}")
print(f"  Successful: {result.processed_count}")
print(f"  Failed: {result.failed_count}")
print(f"  Average quality: {result.average_quality:.2f}")
print(f"  Processing time: {result.total_time:.1f}s")

# Quality distribution
quality_scores = [doc.quality_score for doc in result.documents]
print(f"Quality distribution:")
print(f"  High (>0.8): {sum(1 for q in quality_scores if q > 0.8)}")
print(f"  Medium (0.6-0.8): {sum(1 for q in quality_scores if 0.6 <= q <= 0.8)}")
print(f"  Low (<0.6): {sum(1 for q in quality_scores if q < 0.6)}")
```

### Error Handling

Handle processing errors gracefully:

```python
result = pipeline.process_directory("/data/raw")

# Check for errors
if result.errors:
    print("Processing errors encountered:")
    for error in result.errors:
        print(f"  {error.file_path}: {error.message}")
        if error.suggestion:
            print(f"    Suggestion: {error.suggestion}")

# Process successful documents only
successful_docs = [doc for doc in result.documents if doc is not None]
```

## Performance Optimization

### Parallel Processing

Enable parallel processing for better performance:

```yaml
ingest:
  parallel_processing: true
  max_workers: 8  # Adjust based on CPU cores

clean:
  parallel_processing: true
  batch_size: 100  # Process in batches
```

### Memory Management

For large datasets, use streaming processing:

```python
from qudata import QuDataPipeline

# Configure for large datasets
config.performance = {
    "streaming_mode": True,
    "batch_size": 50,
    "max_memory_usage": "8GB"
}

pipeline = QuDataPipeline(config)

# Process in streaming mode
for batch_result in pipeline.process_directory_streaming("/large/dataset"):
    print(f"Processed batch: {batch_result.processed_count} documents")
    # Process or save batch results immediately
```

### Caching

Enable caching for repeated operations:

```yaml
performance:
  caching:
    enabled: true
    cache_dir: "/tmp/qudata_cache"
    cache_size: "1GB"
    
  # Cache expensive operations
  cache_language_detection: true
  cache_ner_results: true
  cache_quality_scores: true
```

## Troubleshooting

### Common Issues

#### Low Quality Scores
```python
# Investigate low quality scores
low_quality_docs = [doc for doc in result.documents if doc.quality_score < 0.5]

for doc in low_quality_docs[:5]:  # Check first 5
    print(f"File: {doc.source_path}")
    print(f"Quality: {doc.quality_score}")
    print(f"Length: {len(doc.content)} characters")
    print(f"Language: {doc.metadata.language}")
    print("---")
```

#### Processing Failures
```python
# Check processing failures
failed_files = [error.file_path for error in result.errors]
print(f"Failed to process {len(failed_files)} files:")

# Group errors by type
from collections import Counter
error_types = Counter(error.error_type for error in result.errors)
for error_type, count in error_types.most_common():
    print(f"  {error_type}: {count} files")
```

#### Memory Issues
```python
# Monitor memory usage
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Check memory before and after processing
monitor_memory()
result = pipeline.process_directory("/data")
monitor_memory()
```

## Best Practices

### 1. Configuration Management

- Use version control for configuration files
- Test configurations with small datasets first
- Document configuration changes and their impact

### 2. Quality Control

- Set appropriate quality thresholds for your use case
- Regularly review low-quality documents
- Monitor quality trends over time

### 3. Performance

- Start with small batches to tune performance
- Monitor resource usage during processing
- Use parallel processing for CPU-intensive tasks

### 4. Data Management

- Organize input data in logical directory structures
- Keep raw and processed data separate
- Implement backup strategies for important datasets

### 5. Monitoring

- Log processing statistics and errors
- Set up alerts for quality degradation
- Regular quality audits of processed datasets

This guide provides a comprehensive overview of using the QuData pipeline. For specific use cases or advanced configurations, refer to the API documentation and configuration examples.