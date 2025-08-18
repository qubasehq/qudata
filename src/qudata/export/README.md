# Export and Format Generation Module

The export module provides comprehensive dataset export capabilities, transforming processed documents into various formats optimized for different LLM training frameworks and analysis tools.

## Overview

This module handles:
- **Multi-Format Export**: JSONL, ChatML, Alpaca, Parquet, CSV formats
- **Content Segmentation**: Instruction-Context-Output format generation
- **Dataset Splitting**: Train/validation/test split creation
- **LLMBuilder Integration**: Direct export to training pipelines
- **Format Validation**: Export format compliance checking
- **Batch Processing**: Efficient large dataset export
- **Custom Formats**: Extensible format generation system

## Core Components

### Content Segmentation

```python
from qudata.export import ContentSegmenter

segmenter = ContentSegmenter({
    'format': 'instruction_context_output',
    'instruction_patterns': [
        r'Question:\s*(.*?)\n',
        r'Task:\s*(.*?)\n'
    ],
    'context_extraction': 'auto',
    'output_patterns': [
        r'Answer:\s*(.*?)(?:\n|$)',
        r'Response:\s*(.*?)(?:\n|$)'
    ]
})

segments = segmenter.segment_document(document)
for segment in segments:
    print(f"Instruction: {segment.instruction}")
    print(f"Context: {segment.context}")
    print(f"Output: {segment.output}")
```

**Segmentation Features:**
- **Pattern-based extraction**: Regex patterns for instruction/output detection
- **ML-based segmentation**: Transformer models for intelligent segmentation
- **Context preservation**: Maintain document structure and relationships
- **Quality filtering**: Remove low-quality or incomplete segments
- **Format validation**: Ensure proper instruction-following format

### Format Exporters

#### JSONL Exporter

```python
from qudata.export import JSONLExporter

exporter = JSONLExporter({
    'include_metadata': True,
    'flatten_structure': False,
    'custom_fields': ['quality_score', 'domain', 'language']
})

export_path = exporter.export_dataset(dataset, "output.jsonl")
```

**JSONL Format:**
```json
{"id": "doc_1", "text": "Document content", "metadata": {"quality": 0.85}}
{"id": "doc_2", "text": "Another document", "metadata": {"quality": 0.92}}
```

#### ChatML Exporter

```python
from qudata.export import ChatMLExporter

exporter = ChatMLExporter({
    'system_message': 'You are a helpful assistant.',
    'user_role': 'user',
    'assistant_role': 'assistant',
    'include_system': True
})

export_path = exporter.export_dataset(dataset, "output_chatml.jsonl")
```

**ChatML Format:**
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is machine learning?"},
  {"role": "assistant", "content": "Machine learning is..."}
]}
```

#### Alpaca Exporter

```python
from qudata.export import AlpacaExporter

exporter = AlpacaExporter({
    'instruction_field': 'instruction',
    'input_field': 'input',
    'output_field': 'output',
    'include_empty_input': False
})

export_path = exporter.export_dataset(dataset, "output_alpaca.json")
```

**Alpaca Format:**
```json
[
  {
    "instruction": "Explain the concept of...",
    "input": "Additional context...",
    "output": "The concept refers to..."
  }
]
```

#### Parquet Exporter

```python
from qudata.export import ParquetExporter

exporter = ParquetExporter({
    'compression': 'snappy',
    'include_index': False,
    'partition_by': ['domain', 'language'],
    'max_file_size': '100MB'
})

export_path = exporter.export_dataset(dataset, "output.parquet")
```

### Dataset Splitting

```python
from qudata.export import DatasetSplitter

splitter = DatasetSplitter({
    'train_ratio': 0.8,
    'validation_ratio': 0.1,
    'test_ratio': 0.1,
    'stratify_by': 'domain',  # Ensure balanced splits
    'random_seed': 42
})

splits = splitter.split_dataset(dataset)
print(f"Train: {len(splits.train)} documents")
print(f"Validation: {len(splits.validation)} documents")
print(f"Test: {len(splits.test)} documents")

# Export splits separately
for split_name, split_data in splits.items():
    exporter.export_dataset(split_data, f"{split_name}.jsonl")
```

### LLMBuilder Integration

```python
from qudata.export import LLMBuilderConnector

connector = LLMBuilderConnector({
    'llmbuilder_path': '/path/to/llmbuilder',
    'data_directory': 'data/clean',
    'trigger_training': True,
    'model_config': {
        'model_type': 'llama',
        'size': '7b',
        'training_steps': 1000
    }
})

# Export and trigger training
result = connector.export_to_llmbuilder(dataset, "my_dataset")
if result.success:
    print(f"Dataset exported to: {result.export_path}")
    print(f"Training job ID: {result.training_job_id}")
```

## Configuration

### Export Configuration

```yaml
# configs/export.yaml
export:
  formats:
    jsonl:
      enabled: true
      include_metadata: true
      custom_fields: ["quality_score", "domain"]
    
    chatml:
      enabled: true
      system_message: "You are a helpful assistant."
      include_system: true
    
    alpaca:
      enabled: true
      include_empty_input: false
    
    parquet:
      enabled: true
      compression: "snappy"
      partition_by: ["domain"]
  
  splitting:
    enabled: true
    train_ratio: 0.8
    validation_ratio: 0.1
    test_ratio: 0.1
    stratify_by: "domain"
    random_seed: 42
  
  llmbuilder:
    enabled: true
    auto_trigger: false
    data_directory: "data/clean"
```

### Advanced Configuration

```python
from qudata.export import ExportManager

# Custom export pipeline
config = {
    'export_formats': ['jsonl', 'parquet'],
    'quality_threshold': 0.7,
    'max_document_length': 4096,
    'batch_size': 1000,
    'parallel_processing': True,
    'validation': {
        'validate_format': True,
        'check_completeness': True,
        'verify_encoding': True
    }
}

manager = ExportManager(config)
```

## Advanced Features

### Custom Format Creation

```python
from qudata.export import BaseExporter

class CustomExporter(BaseExporter):
    def __init__(self, config):
        super().__init__(config)
        self.custom_field = config.get('custom_field', 'content')
    
    def export_document(self, document):
        return {
            'id': document.id,
            'content': document.content,
            'custom_data': self.extract_custom_data(document),
            'timestamp': document.metadata.creation_date.isoformat()
        }
    
    def export_dataset(self, dataset, output_path):
        with open(output_path, 'w') as f:
            for doc in dataset.documents:
                doc_data = self.export_document(doc)
                f.write(json.dumps(doc_data) + '\n')
        return output_path

# Use custom exporter
exporter = CustomExporter({'custom_field': 'special_content'})
```

### Batch Export Processing

```python
from qudata.export import ExportManager

manager = ExportManager({
    'batch_size': 5000,
    'parallel_workers': 4,
    'memory_limit': '4GB'
})

# Export large dataset efficiently
large_dataset = Dataset(documents=large_document_list)
results = manager.export_dataset_batch(
    large_dataset, 
    formats=['jsonl', 'parquet'],
    output_dir='exports/'
)
```

### Export Validation

```python
from qudata.export import ExportValidator

validator = ExportValidator({
    'check_format_compliance': True,
    'validate_encoding': True,
    'check_completeness': True,
    'verify_splits': True
})

# Validate exported files
validation_result = validator.validate_export(
    export_path="output.jsonl",
    format_type="jsonl",
    original_dataset=dataset
)

if not validation_result.is_valid:
    print("Validation errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

## Performance Optimization

### Streaming Export

```python
from qudata.export import JSONLExporter

exporter = JSONLExporter({
    'streaming_mode': True,
    'buffer_size': 1000,
    'compression': 'gzip'
})

# Stream large dataset export
def export_large_dataset(dataset, output_path):
    with exporter.stream_export(output_path) as stream:
        for document in dataset.documents:
            stream.write_document(document)
```

### Parallel Processing

```python
from qudata.export import ExportManager
import concurrent.futures

manager = ExportManager()

def export_chunk(chunk, format_type, output_dir):
    return manager.export_chunk(chunk, format_type, output_dir)

# Parallel export processing
chunks = manager.split_dataset_into_chunks(large_dataset, chunk_size=1000)
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(export_chunk, chunk, 'jsonl', 'output/')
        for chunk in chunks
    ]
    
    results = [future.result() for future in futures]
```

### Memory Management

```python
# Memory-efficient export for very large datasets
from qudata.export import JSONLExporter

exporter = JSONLExporter({
    'memory_efficient': True,
    'max_memory_usage': '2GB',
    'temp_directory': '/tmp/export'
})

# Process dataset in chunks
for chunk_path in exporter.export_dataset_chunked(large_dataset, chunk_size=10000):
    print(f"Exported chunk: {chunk_path}")
```

## Examples

### Basic Export Workflow

```python
from qudata.export import JSONLExporter, DatasetSplitter

# Load processed dataset
dataset = Dataset.load_from_directory("data/processed")

# Split dataset
splitter = DatasetSplitter({
    'train_ratio': 0.8,
    'validation_ratio': 0.1,
    'test_ratio': 0.1
})
splits = splitter.split_dataset(dataset)

# Export each split
exporter = JSONLExporter({'include_metadata': True})
for split_name, split_data in splits.items():
    output_path = f"exports/{split_name}.jsonl"
    exporter.export_dataset(split_data, output_path)
    print(f"Exported {split_name}: {len(split_data.documents)} documents")
```

### Multi-Format Export

```python
from qudata.export import ExportManager

manager = ExportManager({
    'formats': {
        'jsonl': {'include_metadata': True},
        'chatml': {'system_message': 'You are an AI assistant.'},
        'parquet': {'compression': 'snappy'},
        'csv': {'include_headers': True}
    }
})

# Export to multiple formats
results = manager.export_all_formats(dataset, output_dir="exports/")
for format_name, result in results.items():
    print(f"{format_name}: {result.file_path} ({result.file_size} bytes)")
```

### Content Segmentation Pipeline

```python
from qudata.export import ContentSegmenter, AlpacaExporter

# Configure segmentation
segmenter = ContentSegmenter({
    'format': 'instruction_context_output',
    'min_instruction_length': 10,
    'min_output_length': 20,
    'quality_threshold': 0.7
})

# Segment documents
segmented_documents = []
for document in dataset.documents:
    segments = segmenter.segment_document(document)
    for segment in segments:
        if segment.is_valid():
            segmented_documents.append(segment)

print(f"Generated {len(segmented_documents)} training segments")

# Export in Alpaca format
exporter = AlpacaExporter()
export_path = exporter.export_segments(segmented_documents, "training_data.json")
```

### LLMBuilder Integration

```python
from qudata.export import LLMBuilderConnector

connector = LLMBuilderConnector({
    'llmbuilder_path': '/path/to/llmbuilder',
    'auto_trigger_training': True,
    'model_config': {
        'model_type': 'llama',
        'size': '7b',
        'learning_rate': 2e-5,
        'batch_size': 4,
        'num_epochs': 3
    }
})

# Export and start training
result = connector.export_and_train(dataset, "my_custom_dataset")
if result.success:
    print(f"Training started: {result.training_job_id}")
    
    # Monitor training progress
    while not connector.is_training_complete(result.training_job_id):
        status = connector.get_training_status(result.training_job_id)
        print(f"Training progress: {status.progress}%")
        time.sleep(60)  # Check every minute
```

### Quality-Based Export

```python
from qudata.export import JSONLExporter

# Filter by quality before export
high_quality_docs = [
    doc for doc in dataset.documents 
    if doc.metadata.quality_score >= 0.8
]

medium_quality_docs = [
    doc for doc in dataset.documents 
    if 0.6 <= doc.metadata.quality_score < 0.8
]

# Export different quality tiers
exporter = JSONLExporter()
exporter.export_documents(high_quality_docs, "high_quality.jsonl")
exporter.export_documents(medium_quality_docs, "medium_quality.jsonl")

print(f"High quality: {len(high_quality_docs)} documents")
print(f"Medium quality: {len(medium_quality_docs)} documents")
```

## Testing

```bash
# Run export module tests
pytest tests/unit/test_content_segmentation.py
pytest tests/unit/test_export_formats.py
pytest tests/integration/test_llmbuilder_integration.py

# Test specific exporters
pytest tests/unit/test_jsonl_exporter.py -v
pytest tests/unit/test_chatml_exporter.py -v
```

## Dependencies

**Core Dependencies:**
- `pandas`: Data manipulation for export formats
- `pyarrow`: Parquet format support
- `jsonlines`: JSONL format handling

**Optional Dependencies:**
- `datasets`: HuggingFace datasets integration
- `transformers`: Advanced tokenization and formatting

## Troubleshooting

### Common Issues

**Large File Export Failures:**
```python
# Use streaming export
exporter = JSONLExporter({
    'streaming_mode': True,
    'buffer_size': 1000,
    'max_file_size': '1GB'
})
```

**Memory Issues:**
```python
# Enable memory-efficient mode
exporter = JSONLExporter({
    'memory_efficient': True,
    'temp_directory': '/tmp/export',
    'cleanup_temp_files': True
})
```

**Format Validation Errors:**
```python
# Enable detailed validation
validator = ExportValidator({
    'strict_validation': False,
    'auto_fix_errors': True,
    'log_validation_details': True
})
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `formats.py`: Format-specific exporters (JSONL, ChatML, Alpaca, etc.)
- `segmenter.py`: Content segmentation for training formats
- `llmbuilder.py`: LLMBuilder integration and automation
- `catalog.py`: Export catalog and metadata management