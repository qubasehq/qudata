# QuData API Documentation

This directory contains comprehensive API documentation for QuData, covering both the REST API and Python SDK interfaces.

## Available Documentation

### REST API
- **[REST API Reference](rest-api.md)**: Complete REST API documentation with endpoints, request/response formats, and examples
- **Base URL**: `https://api.qudata.com/v1`
- **Authentication**: API key-based authentication
- **Rate Limits**: Tiered rate limiting based on subscription

### Python SDK
The Python SDK provides a convenient interface for programmatic access to QuData functionality.

#### Installation
```bash
pip install qudata-sdk
```

#### Quick Start
```python
import qudata

# Initialize client
client = qudata.Client(api_key="your_api_key")

# Create and process dataset
dataset = client.datasets.create(
    name="My Dataset",
    files=["document1.pdf", "document2.docx"]
)

# Wait for processing
dataset.wait_for_completion()

# Export results
export = dataset.export(format="jsonl")
export.download("./output/")
```

## Core API Concepts

### Datasets
Datasets are collections of processed documents with associated metadata and quality metrics.

### Processing Jobs
Asynchronous jobs that handle data ingestion, cleaning, annotation, and quality scoring.

### Export Jobs
Jobs that convert processed datasets into various training formats (JSONL, ChatML, Alpaca, etc.).

### Configuration
YAML-based configuration system for customizing processing pipelines.

## Authentication

All API requests require authentication using an API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.qudata.com/v1/datasets
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "quality_threshold",
      "issue": "Value must be between 0 and 1"
    },
    "request_id": "req_123456789"
  }
}
```

## Rate Limiting

API requests are rate-limited based on your subscription tier:

- **Free**: 100 requests/hour
- **Pro**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## Webhooks

Configure webhooks to receive notifications about processing events:

```python
client.webhooks.create(
    url="https://your-app.com/webhook",
    events=["dataset.completed", "job.failed"],
    secret="your_webhook_secret"
)
```

## SDK Examples

### Dataset Management

```python
# List datasets
datasets = client.datasets.list(status="completed")

# Get dataset details
dataset = client.datasets.get("dataset_123")

# Update dataset
dataset.update(name="Updated Name")

# Delete dataset
dataset.delete()
```

### Processing Configuration

```python
# Create dataset with custom configuration
dataset = client.datasets.create(
    name="Academic Papers",
    files=["paper1.pdf", "paper2.pdf"],
    config={
        "ingest": {
            "file_types": ["pdf"],
            "extract_citations": True
        },
        "clean": {
            "min_quality_score": 0.8,
            "remove_duplicates": True
        },
        "export": {
            "formats": ["jsonl", "parquet"],
            "split_data": True
        }
    }
)
```

### Monitoring Progress

```python
# Monitor processing progress
dataset.on('progress', lambda progress: 
    print(f"Progress: {progress.percentage}%")
)

# Get detailed job status
job = client.jobs.get(dataset.processing_job_id)
print(f"Current stage: {job.progress.current_stage}")
print(f"Documents processed: {job.progress.documents_processed}")
```

### Export and Download

```python
# Export in multiple formats
exports = dataset.export_all([
    {"format": "jsonl", "split_data": True},
    {"format": "parquet", "include_metadata": True},
    {"format": "csv", "quality_threshold": 0.7}
])

# Download exports
for export in exports:
    export.download(f"./output/{export.format}/")
```

### Batch Operations

```python
# Process multiple datasets
datasets = []
for file_batch in file_batches:
    dataset = client.datasets.create(
        name=f"Batch {len(datasets)}",
        files=file_batch
    )
    datasets.append(dataset)

# Wait for all to complete
client.datasets.wait_for_completion(datasets)

# Bulk export
exports = client.datasets.bulk_export(
    datasets, 
    format="jsonl",
    merge=True
)
```

## Advanced Features

### Custom Processing Pipelines

```python
# Define custom processing pipeline
pipeline_config = {
    "stages": [
        {"name": "custom_extractor", "class": "MyCustomExtractor"},
        {"name": "custom_cleaner", "class": "MyCustomCleaner"},
        {"name": "custom_annotator", "class": "MyCustomAnnotator"}
    ],
    "stage_configs": {
        "custom_extractor": {"param1": "value1"},
        "custom_cleaner": {"param2": "value2"}
    }
}

dataset = client.datasets.create(
    name="Custom Pipeline",
    files=files,
    pipeline_config=pipeline_config
)
```

### Real-time Processing

```python
# Stream processing for real-time data
stream = client.streams.create(
    name="Real-time News",
    source_type="rss",
    source_config={
        "feeds": ["https://news.example.com/feed.xml"],
        "update_interval": 300  # 5 minutes
    }
)

# Process stream data
for batch in stream.process():
    print(f"Processed {len(batch.documents)} new documents")
```

### Integration with Training Frameworks

```python
# Direct integration with Hugging Face
dataset.export_to_huggingface(
    repo_name="my-org/my-dataset",
    format="parquet",
    private=True
)

# Integration with OpenAI fine-tuning
dataset.export_to_openai(
    format="jsonl",
    purpose="fine-tune"
)

# Custom training framework integration
dataset.export_to_custom(
    endpoint="https://my-training-platform.com/api/datasets",
    format="custom",
    auth_token="training_platform_token"
)
```

## Error Handling and Debugging

### Exception Handling

```python
from qudata.exceptions import (
    QuDataAPIError, 
    ProcessingError, 
    ValidationError,
    RateLimitError
)

try:
    dataset = client.datasets.create(name="Test", files=files)
    dataset.wait_for_completion()
    
except ValidationError as e:
    print(f"Configuration error: {e.message}")
    
except ProcessingError as e:
    print(f"Processing failed: {e.message}")
    print(f"Failed documents: {e.failed_documents}")
    
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}")
    
except QuDataAPIError as e:
    print(f"API error: {e.message}")
    print(f"Request ID: {e.request_id}")
```

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger('qudata').setLevel(logging.DEBUG)

# Get detailed processing information
dataset = client.datasets.get("dataset_123")
debug_info = dataset.get_debug_info()

print(f"Processing stages: {debug_info.stages}")
print(f"Error details: {debug_info.errors}")
print(f"Performance metrics: {debug_info.performance}")

# Validate configuration before processing
validation_result = client.config.validate(my_config)
if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Config error: {error}")
```

## Best Practices

### Configuration Management

```python
# Use environment-specific configurations
config = client.config.load_template("academic-papers")
config.update({
    "quality": {"min_score": 0.8},
    "export": {"formats": ["jsonl"]}
})

# Validate before use
if client.config.validate(config).is_valid:
    dataset = client.datasets.create(config=config)
```

### Resource Management

```python
# Use context managers for automatic cleanup
with client.datasets.create_session() as session:
    dataset = session.create(name="Temp Dataset", files=files)
    result = dataset.process()
    # Dataset automatically cleaned up on exit
```

### Performance Optimization

```python
# Batch operations for better performance
with client.batch_mode():
    datasets = []
    for file_batch in file_batches:
        dataset = client.datasets.create(files=file_batch)
        datasets.append(dataset)
    
    # All requests sent together
    results = client.execute_batch()
```

## Support and Resources

- **API Status**: https://status.qudata.com
- **Rate Limits**: Check current usage with `client.usage.get_current()`
- **Documentation**: https://docs.qudata.com
- **Support**: support@qudata.com
- **Community**: https://community.qudata.com

For detailed endpoint documentation, see [REST API Reference](rest-api.md).