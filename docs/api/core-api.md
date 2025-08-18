# QuData Core API Reference

This document provides comprehensive API reference for the QuData core components.

## Table of Contents

- [Pipeline API](#pipeline-api)
- [Configuration API](#configuration-api)
- [Data Models](#data-models)
- [Ingestion API](#ingestion-api)
- [Processing API](#processing-api)
- [Export API](#export-api)
- [Analysis API](#analysis-api)

## Pipeline API

### QuDataPipeline

The main pipeline class for orchestrating the entire data processing workflow.

```python
from qudata import QuDataPipeline, load_config

# Initialize pipeline
config = load_config("pipeline.yaml")
pipeline = QuDataPipeline(config)

# Process documents
result = pipeline.process_directory("/path/to/documents")
print(f"Processed {result.processed_count} documents")
```

#### Methods

##### `__init__(config: PipelineConfig)`
Initialize the pipeline with configuration.

**Parameters:**
- `config` (PipelineConfig): Pipeline configuration object

**Example:**
```python
from qudata import QuDataPipeline, PipelineConfig

config = PipelineConfig(
    ingest=IngestConfig(formats=["pdf", "docx", "txt"]),
    clean=CleanConfig(remove_duplicates=True),
    export=ExportConfig(formats=["jsonl", "chatml"])
)
pipeline = QuDataPipeline(config)
```

##### `process_directory(input_path: str, output_path: str = None) -> PipelineResult`
Process all documents in a directory.

**Parameters:**
- `input_path` (str): Path to input directory
- `output_path` (str, optional): Path to output directory

**Returns:**
- `PipelineResult`: Processing results with statistics and metadata

**Example:**
```python
result = pipeline.process_directory(
    input_path="/data/raw",
    output_path="/data/processed"
)

print(f"Success: {result.success}")
print(f"Processed: {result.processed_count}")
print(f"Failed: {result.failed_count}")
print(f"Quality Score: {result.average_quality}")
```

##### `process_file(file_path: str) -> ProcessingResult`
Process a single file.

**Parameters:**
- `file_path` (str): Path to the file to process

**Returns:**
- `ProcessingResult`: Individual file processing result

**Example:**
```python
result = pipeline.process_file("/path/to/document.pdf")

if result.success:
    document = result.document
    print(f"Title: {document.metadata.title}")
    print(f"Quality: {document.quality_score}")
    print(f"Content length: {len(document.content)}")
else:
    for error in result.errors:
        print(f"Error: {error.message}")
```

##### `process_documents(documents: List[Document]) -> List[ProcessingResult]`
Process a list of documents.

**Parameters:**
- `documents` (List[Document]): List of documents to process

**Returns:**
- `List[ProcessingResult]`: List of processing results

**Example:**
```python
documents = [
    Document(content="Sample text 1", metadata=DocumentMetadata()),
    Document(content="Sample text 2", metadata=DocumentMetadata())
]

results = pipeline.process_documents(documents)
successful_results = [r for r in results if r.success]
```

## Configuration API

### ConfigManager

Manages configuration loading, validation, and access.

```python
from qudata import ConfigManager, get_config_manager

# Get singleton config manager
config_manager = get_config_manager()

# Load configuration
config = config_manager.load_config("pipeline.yaml")

# Validate configuration
validation_result = config_manager.validate_config(config)
if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Config error: {error}")
```

#### Methods

##### `load_config(config_path: str) -> PipelineConfig`
Load configuration from file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `PipelineConfig`: Loaded configuration object

##### `validate_config(config: PipelineConfig) -> ValidationResult`
Validate configuration object.

**Parameters:**
- `config` (PipelineConfig): Configuration to validate

**Returns:**
- `ValidationResult`: Validation result with errors if any

### Configuration Classes

#### PipelineConfig
Main pipeline configuration container.

```python
from qudata import PipelineConfig, IngestConfig, CleanConfig

config = PipelineConfig(
    ingest=IngestConfig(
        formats=["pdf", "docx", "txt", "html"],
        max_file_size="100MB",
        parallel_processing=True
    ),
    clean=CleanConfig(
        remove_duplicates=True,
        min_quality_score=0.7,
        language_filter=["en", "es"]
    )
)
```

#### IngestConfig
Configuration for data ingestion.

**Attributes:**
- `formats` (List[str]): Supported file formats
- `max_file_size` (str): Maximum file size to process
- `parallel_processing` (bool): Enable parallel processing
- `ocr_enabled` (bool): Enable OCR for images and scanned PDFs

#### CleanConfig
Configuration for data cleaning.

**Attributes:**
- `remove_duplicates` (bool): Remove duplicate content
- `min_quality_score` (float): Minimum quality threshold
- `language_filter` (List[str]): Languages to keep
- `remove_boilerplate` (bool): Remove headers/footers

## Data Models

### Document

Core document data model.

```python
from qudata import Document, DocumentMetadata, DocumentStructure

document = Document(
    id="doc_001",
    source_path="/path/to/file.pdf",
    content="Document content here...",
    metadata=DocumentMetadata(
        title="Sample Document",
        author="John Doe",
        language="en",
        file_type="pdf"
    ),
    quality_score=0.85
)
```

#### Attributes

- `id` (str): Unique document identifier
- `source_path` (str): Original file path
- `content` (str): Extracted text content
- `metadata` (DocumentMetadata): Document metadata
- `structure` (DocumentStructure): Document structure information
- `quality_score` (float): Quality assessment score (0.0-1.0)
- `processing_timestamp` (datetime): When document was processed
- `version` (str): Processing version

#### Methods

##### `to_dict() -> Dict[str, Any]`
Convert document to dictionary.

```python
doc_dict = document.to_dict()
print(doc_dict["metadata"]["title"])
```

##### `from_dict(data: Dict[str, Any]) -> Document`
Create document from dictionary.

```python
document = Document.from_dict(doc_dict)
```

##### `get_word_count() -> int`
Get word count of document content.

```python
word_count = document.get_word_count()
print(f"Document has {word_count} words")
```

### DocumentMetadata

Document metadata container.

```python
from qudata import DocumentMetadata, Entity

metadata = DocumentMetadata(
    title="Research Paper",
    author="Dr. Smith",
    creation_date=datetime.now(),
    language="en",
    file_type="pdf",
    domain="academic",
    topics=["machine learning", "AI"],
    entities=[
        Entity(text="OpenAI", label="ORG", confidence=0.95),
        Entity(text="GPT-4", label="PRODUCT", confidence=0.90)
    ]
)
```

#### Attributes

- `title` (Optional[str]): Document title
- `author` (Optional[str]): Document author
- `creation_date` (Optional[datetime]): Creation timestamp
- `language` (str): Detected language code
- `file_type` (str): Original file format
- `domain` (str): Content domain/category
- `topics` (List[str]): Identified topics
- `entities` (List[Entity]): Named entities

### ProcessingResult

Result of document processing operation.

```python
from qudata import ProcessingResult, ProcessingError

result = ProcessingResult(
    success=True,
    document=processed_document,
    errors=[],
    warnings=["Low quality score"],
    processing_time=2.5,
    stage_results={
        "ingest": {"success": True, "time": 0.5},
        "clean": {"success": True, "time": 1.2},
        "annotate": {"success": True, "time": 0.8}
    }
)
```

#### Attributes

- `success` (bool): Whether processing succeeded
- `document` (Optional[Document]): Processed document if successful
- `errors` (List[ProcessingError]): Processing errors
- `warnings` (List[str]): Processing warnings
- `processing_time` (float): Total processing time in seconds
- `stage_results` (Dict[str, Any]): Per-stage processing results

## Ingestion API

### FileTypeDetector

Automatic file type detection.

```python
from qudata.ingest import FileTypeDetector

detector = FileTypeDetector()
file_type = detector.detect_file_type("/path/to/document.pdf")
print(f"Detected type: {file_type}")  # Output: pdf

# Check if format is supported
if detector.is_supported(file_type):
    print("Format is supported for processing")
```

#### Methods

##### `detect_file_type(file_path: str) -> str`
Detect file type from file path and content.

##### `is_supported(file_type: str) -> bool`
Check if file type is supported for processing.

##### `get_supported_formats() -> List[str]`
Get list of all supported file formats.

### Extractors

#### PDFExtractor

Extract content from PDF files.

```python
from qudata.ingest import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract("/path/to/document.pdf")

print(f"Content: {result.content}")
print(f"Tables: {len(result.tables)}")
print(f"Images: {len(result.images)}")
```

#### DocumentExtractor

Extract content from DOCX and other document formats.

```python
from qudata.ingest import DocumentExtractor

extractor = DocumentExtractor()
result = extractor.extract("/path/to/document.docx")

print(f"Title: {result.metadata.title}")
print(f"Author: {result.metadata.author}")
print(f"Content: {result.content[:200]}...")
```

#### WebExtractor

Extract content from HTML and web pages.

```python
from qudata.ingest import WebExtractor

extractor = WebExtractor()
result = extractor.extract_from_url("https://example.com/article")

print(f"Title: {result.metadata.title}")
print(f"Clean content: {result.content}")
```

## Processing API

### Cleaning Pipeline

```python
from qudata.clean import ComprehensiveCleaningPipeline

cleaner = ComprehensiveCleaningPipeline()
cleaned_document = cleaner.clean_document(document)

print(f"Original length: {len(document.content)}")
print(f"Cleaned length: {len(cleaned_document.content)}")
print(f"Quality score: {cleaned_document.quality_score}")
```

### Annotation Pipeline

```python
from qudata.annotate import TaxonomyClassifier, MetadataExtractor

# Classify content
classifier = TaxonomyClassifier()
categories = classifier.classify(document.content)
print(f"Categories: {categories}")

# Extract metadata
extractor = MetadataExtractor()
metadata = extractor.extract_metadata(document)
print(f"Entities: {metadata.entities}")
```

## Export API

### Format Exporters

```python
from qudata.export import ContentSegmenter
from qudata.pack import JSONLFormatter, ChatMLFormatter

# Segment content for training
segmenter = ContentSegmenter()
segments = segmenter.segment_document(document, format="instruction")

# Export to JSONL
jsonl_formatter = JSONLFormatter()
jsonl_formatter.export_to_file(
    documents=[document],
    output_path="training_data.jsonl"
)

# Export to ChatML
chatml_formatter = ChatMLFormatter()
chatml_data = chatml_formatter.format_documents([document])
```

## Analysis API

### Analysis Engine

```python
from qudata.analyze import AnalysisEngine

analyzer = AnalysisEngine()

# Analyze text statistics
stats = analyzer.analyze_text_statistics([document])
print(f"Total words: {stats.total_words}")
print(f"Unique tokens: {stats.unique_tokens}")

# Perform topic modeling
topics = analyzer.perform_topic_modeling(
    texts=[doc.content for doc in documents],
    method="bertopic"
)
print(f"Found {len(topics.topics)} topics")

# Analyze sentiment
sentiment = analyzer.analyze_sentiment([doc.content for doc in documents])
print(f"Average sentiment: {sentiment.average_polarity}")
```

## Error Handling

All API methods use consistent error handling patterns:

```python
from qudata import ProcessingError, ErrorSeverity

try:
    result = pipeline.process_file("document.pdf")
    if not result.success:
        for error in result.errors:
            if error.severity == ErrorSeverity.CRITICAL:
                print(f"Critical error: {error.message}")
                print(f"Stage: {error.stage}")
                print(f"Suggestion: {error.suggestion}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

## Async API Support

For high-throughput scenarios, async versions are available:

```python
import asyncio
from qudata import AsyncQuDataPipeline

async def process_documents_async():
    pipeline = AsyncQuDataPipeline(config)
    
    # Process multiple files concurrently
    tasks = [
        pipeline.process_file_async(f"document_{i}.pdf")
        for i in range(100)
    ]
    
    results = await asyncio.gather(*tasks)
    successful = [r for r in results if r.success]
    print(f"Successfully processed {len(successful)} documents")

# Run async processing
asyncio.run(process_documents_async())
```

## Configuration Examples

### Basic Configuration

```yaml
# pipeline.yaml
ingest:
  formats: ["pdf", "docx", "txt", "html"]
  max_file_size: "100MB"
  parallel_processing: true
  ocr_enabled: true

clean:
  remove_duplicates: true
  min_quality_score: 0.7
  language_filter: ["en"]
  remove_boilerplate: true

annotate:
  enable_ner: true
  enable_classification: true
  taxonomy_file: "configs/taxonomy.yaml"

export:
  formats: ["jsonl", "chatml"]
  output_dir: "/data/processed"
  split_ratios: [0.8, 0.1, 0.1]  # train, val, test
```

### Advanced Configuration

```yaml
# advanced-pipeline.yaml
ingest:
  formats: ["pdf", "docx", "txt", "html", "csv", "json"]
  max_file_size: "500MB"
  parallel_processing: true
  max_workers: 8
  ocr_enabled: true
  ocr_languages: ["eng", "spa", "fra"]
  
  web_scraping:
    enabled: true
    rate_limit: 10  # requests per second
    user_agent: "QuData/1.0"
    
  database:
    enabled: true
    connections:
      - type: "postgresql"
        host: "localhost"
        database: "content_db"
        
clean:
  remove_duplicates: true
  similarity_threshold: 0.9
  min_quality_score: 0.7
  max_quality_score: 1.0
  language_filter: ["en", "es", "fr"]
  remove_boilerplate: true
  
  text_normalization:
    unicode_normalization: "NFKC"
    remove_accents: false
    lowercase: false
    
  html_cleaning:
    remove_scripts: true
    remove_styles: true
    preserve_links: false

annotate:
  enable_ner: true
  enable_classification: true
  enable_topic_modeling: true
  
  ner_model: "en_core_web_sm"
  classification_model: "custom"
  taxonomy_file: "configs/taxonomy.yaml"
  
  quality_scoring:
    weights:
      length: 0.2
      language: 0.3
      coherence: 0.3
      uniqueness: 0.2

export:
  formats: ["jsonl", "chatml", "parquet"]
  output_dir: "/data/processed"
  
  jsonl:
    fields: ["content", "metadata", "quality_score"]
    filter_low_quality: true
    
  chatml:
    system_message: "You are a helpful assistant."
    include_metadata: true
    
  splits:
    enabled: true
    ratios: [0.8, 0.1, 0.1]
    stratify_by: "domain"
```

This API reference provides comprehensive documentation for all major components of the QuData system. Each section includes practical examples and configuration options to help developers integrate and use the system effectively.