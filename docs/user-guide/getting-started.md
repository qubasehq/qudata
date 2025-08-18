# Getting Started with QuData

QuData is a comprehensive data processing pipeline designed to transform raw multi-format data into high-quality datasets optimized for LLM training. This guide will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- 10GB+ free disk space for processing

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/qubasehq/qudata.git
cd qudata

# Install in development mode
pip install -e .

# Verify installation
qudata --version
```

### Optional Dependencies

```bash
# For advanced ML features
pip install -e ".[ml]"

# For web scraping capabilities
pip install -e ".[web]"

# For development tools
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Your Data

Create a directory structure for your data:

```bash
mkdir -p data/{raw,processed,exports}
```

Place your raw files in `data/raw/`:
- PDF documents
- DOCX files
- HTML pages
- Text files
- CSV/JSON data

### 2. Basic Configuration

Create a basic configuration file:

```bash
# Copy default configuration
cp configs/pipeline.yaml configs/my_pipeline.yaml
```

Edit `configs/my_pipeline.yaml`:

```yaml
pipeline:
  input_directory: "data/raw"
  output_directory: "data/processed"
  export_directory: "data/exports"

ingest:
  file_types: ["pdf", "docx", "txt", "html"]
  max_file_size: "100MB"

clean:
  remove_duplicates: true
  normalize_text: true
  min_quality_score: 0.6

export:
  formats: ["jsonl", "parquet"]
  split_data: true
```

### 3. Run Your First Pipeline

```bash
# Process all files in data/raw
qudata process --config configs/my_pipeline.yaml

# Or use the default configuration
qudata process --input data/raw --output data/processed
```

### 4. Export for Training

```bash
# Export processed data in JSONL format
qudata export --format jsonl --input data/processed --output data/exports/training.jsonl

# Export with train/validation/test splits
qudata export --format jsonl --input data/processed --output data/exports --split
```

## Basic Usage Examples

### Processing Individual Files

```python
from qudata import QuDataPipeline

# Initialize pipeline
pipeline = QuDataPipeline()

# Process specific files
files = ["document1.pdf", "document2.docx", "data.csv"]
dataset = pipeline.process_files(files)

print(f"Processed {len(dataset.documents)} documents")
print(f"Average quality: {dataset.quality_metrics.overall_score:.2f}")
```

### Batch Processing

```python
from qudata import QuDataPipeline

# Initialize with custom configuration
pipeline = QuDataPipeline(config_path="configs/my_pipeline.yaml")

# Process entire directory
result = pipeline.process_directory("data/raw", "data/processed")

if result.success:
    print(f"Successfully processed {result.documents_processed} documents")
    print(f"Processing time: {result.processing_time:.2f} seconds")
else:
    print("Processing failed:")
    for error in result.errors:
        print(f"  - {error.message}")
```

### Export to Different Formats

```python
from qudata import QuDataPipeline

pipeline = QuDataPipeline()
dataset = pipeline.process_files(["data.pdf", "document.docx"])

# Export to JSONL
jsonl_path = pipeline.export_dataset(dataset, "jsonl")
print(f"JSONL export: {jsonl_path}")

# Export to Parquet
parquet_path = pipeline.export_dataset(dataset, "parquet")
print(f"Parquet export: {parquet_path}")

# Export to CSV
csv_path = pipeline.export_dataset(dataset, "csv")
print(f"CSV export: {csv_path}")
```

## Configuration Overview

QuData uses YAML configuration files to control processing behavior. The main configuration sections are:

### Pipeline Configuration

```yaml
pipeline:
  input_directory: "data/raw"
  output_directory: "data/processed"
  parallel_processing: true
  max_workers: 4
```

### Ingestion Settings

```yaml
ingest:
  file_types: ["pdf", "docx", "txt", "html", "csv", "json"]
  max_file_size: "100MB"
  skip_corrupted: true
  extract_metadata: true
```

### Cleaning Options

```yaml
clean:
  normalize_text: true
  remove_duplicates: true
  similarity_threshold: 0.85
  remove_boilerplate: true
  fix_ocr_errors: true
  min_length: 50
  max_length: 10000
```

### Quality Control

```yaml
quality:
  min_score: 0.6
  dimensions:
    content: 0.4
    language: 0.3
    structure: 0.3
  auto_filter: true
```

### Export Settings

```yaml
export:
  formats: ["jsonl", "parquet"]
  include_metadata: true
  split_ratios:
    train: 0.8
    validation: 0.1
    test: 0.1
```

## Understanding the Processing Pipeline

QuData processes documents through several stages:

### 1. Ingestion
- **File Detection**: Automatically identifies file types
- **Content Extraction**: Extracts text from various formats
- **Metadata Extraction**: Captures document properties

### 2. Cleaning
- **Text Normalization**: Standardizes encoding and formatting
- **Deduplication**: Removes duplicate content
- **Boilerplate Removal**: Strips headers, footers, navigation
- **OCR Correction**: Fixes common OCR errors

### 3. Annotation
- **Domain Classification**: Categorizes content by topic
- **Entity Recognition**: Identifies people, places, organizations
- **Metadata Enhancement**: Adds author, date, source information

### 4. Quality Scoring
- **Multi-dimensional Assessment**: Evaluates content, language, structure
- **Filtering**: Removes low-quality documents
- **Scoring**: Assigns quality scores for ranking

### 5. Export
- **Format Generation**: Creates training-ready datasets
- **Data Splitting**: Generates train/validation/test splits
- **Validation**: Ensures export format compliance

## Monitoring Progress

### Command Line Progress

```bash
# Enable verbose output
qudata process --input data/raw --output data/processed --verbose

# Show detailed statistics
qudata process --input data/raw --output data/processed --stats
```

### Programmatic Monitoring

```python
from qudata import QuDataPipeline

pipeline = QuDataPipeline()

# Process with progress callback
def progress_callback(stage, progress, message):
    print(f"[{stage}] {progress:.1%}: {message}")

result = pipeline.process_directory(
    "data/raw", 
    "data/processed",
    progress_callback=progress_callback
)
```

## Common Workflows

### Academic Paper Processing

```yaml
# configs/academic_papers.yaml
pipeline:
  name: "academic_papers"

ingest:
  file_types: ["pdf"]
  extract_citations: true
  extract_figures: false

clean:
  remove_references: true
  remove_acknowledgments: true
  min_length: 1000  # Longer minimum for papers

annotate:
  extract_authors: true
  extract_keywords: true
  classify_domains: true

quality:
  min_score: 0.7  # Higher quality threshold
  check_academic_format: true
```

### Web Content Processing

```yaml
# configs/web_content.yaml
ingest:
  file_types: ["html"]
  web_scraping:
    enabled: true
    rate_limit: 60  # requests per minute
    extract_main_content: true

clean:
  remove_navigation: true
  remove_ads: true
  remove_comments: true
  html_to_text: true

quality:
  min_length: 200
  max_length: 5000
  check_readability: true
```

### Code Documentation Processing

```yaml
# configs/code_docs.yaml
ingest:
  file_types: ["md", "rst", "txt"]
  preserve_code_blocks: true

clean:
  normalize_code_formatting: true
  remove_duplicates: true
  preserve_structure: true

annotate:
  extract_code_languages: true
  classify_by_framework: true

export:
  formats: ["jsonl"]
  preserve_code_structure: true
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```yaml
# Reduce batch size and enable streaming
pipeline:
  batch_size: 100
  streaming_mode: true
  max_memory_usage: "2GB"
```

**Slow Processing:**
```yaml
# Enable parallel processing
pipeline:
  parallel_processing: true
  max_workers: 8  # Adjust based on CPU cores
```

**Poor Quality Results:**
```yaml
# Adjust quality thresholds
quality:
  min_score: 0.5  # Lower threshold
  auto_filter: false  # Manual review
```

**File Format Issues:**
```yaml
# Enable fallback extraction
ingest:
  fallback_extraction: true
  skip_corrupted: true
  log_extraction_errors: true
```

### Getting Help

- **Documentation**: Check the module-specific README files
- **Examples**: Look at files in the `examples/` directory
- **Configuration**: Review sample configs in `configs/`
- **Logs**: Check processing logs for detailed error information

## Next Steps

1. **Explore Advanced Features**: Learn about web scraping, API integration, and database connectivity
2. **Customize Processing**: Create custom extractors and cleaning rules
3. **Integrate with Training**: Set up LLMBuilder integration for automated training
4. **Monitor Quality**: Use the analysis and visualization tools
5. **Scale Up**: Configure distributed processing for large datasets

For more detailed information, see the specific module documentation:
- [Data Ingestion](../src/forge/ingest/README.md)
- [Text Cleaning](../src/forge/clean/README.md)
- [Content Annotation](../src/forge/annotate/README.md)
- [Data Analysis](../src/forge/analyze/README.md)
- [Export Formats](../src/forge/export/README.md)