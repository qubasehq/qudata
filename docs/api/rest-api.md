# REST API Documentation

QuData provides a comprehensive REST API for programmatic access to all data processing capabilities. This documentation covers all available endpoints, request/response formats, and usage examples.

## Base URL and Authentication

### Base URL
```
https://api.qudata.com/v1
```

### Authentication
QuData API uses API key authentication:

```bash
# Include API key in header
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.qudata.com/v1/datasets
```

## Core Endpoints

### Health Check

#### GET /health
Check API service health and status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "storage": "healthy"
  }
}
```

## Dataset Management

### List Datasets

#### GET /datasets
Retrieve list of all datasets.

**Parameters:**
- `page` (int, optional): Page number (default: 1)
- `limit` (int, optional): Items per page (default: 20, max: 100)
- `status` (string, optional): Filter by status (`processing`, `completed`, `failed`)
- `domain` (string, optional): Filter by domain

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.qudata.com/v1/datasets?page=1&limit=10&status=completed"
```

**Response:**
```json
{
  "datasets": [
    {
      "id": "dataset_123",
      "name": "Academic Papers Dataset",
      "description": "Collection of computer science papers",
      "status": "completed",
      "created_at": "2024-01-10T09:00:00Z",
      "updated_at": "2024-01-10T12:30:00Z",
      "document_count": 1500,
      "total_size": "2.5GB",
      "quality_score": 0.85,
      "domains": ["technology", "research"],
      "languages": ["en"]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 25,
    "pages": 3
  }
}
```

### Get Dataset Details

#### GET /datasets/{dataset_id}
Retrieve detailed information about a specific dataset.

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.qudata.com/v1/datasets/dataset_123"
```

**Response:**
```json
{
  "id": "dataset_123",
  "name": "Academic Papers Dataset",
  "description": "Collection of computer science papers",
  "status": "completed",
  "created_at": "2024-01-10T09:00:00Z",
  "updated_at": "2024-01-10T12:30:00Z",
  "metadata": {
    "document_count": 1500,
    "total_size": "2.5GB",
    "average_document_length": 3500,
    "quality_metrics": {
      "overall_score": 0.85,
      "content_quality": 0.88,
      "language_quality": 0.82,
      "structure_quality": 0.85
    },
    "domain_distribution": {
      "machine_learning": 45,
      "computer_vision": 30,
      "natural_language_processing": 25
    },
    "language_distribution": {
      "en": 95,
      "es": 3,
      "fr": 2
    }
  },
  "processing_stats": {
    "documents_processed": 1500,
    "documents_failed": 25,
    "processing_time": 3600,
    "stages_completed": [
      "ingestion",
      "cleaning",
      "annotation",
      "quality_scoring"
    ]
  }
}
```

### Create Dataset

#### POST /datasets
Create a new dataset from uploaded files or URLs.

**Request Body:**
```json
{
  "name": "My New Dataset",
  "description": "Dataset description",
  "config": {
    "ingest": {
      "file_types": ["pdf", "docx", "txt"],
      "max_file_size": "100MB"
    },
    "clean": {
      "remove_duplicates": true,
      "normalize_text": true,
      "min_quality_score": 0.6
    },
    "export": {
      "formats": ["jsonl", "parquet"]
    }
  },
  "sources": [
    {
      "type": "upload",
      "files": ["file1.pdf", "file2.docx"]
    },
    {
      "type": "url",
      "urls": ["https://example.com/document.pdf"]
    }
  ]
}
```

**Response:**
```json
{
  "id": "dataset_456",
  "name": "My New Dataset",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T11:30:00Z",
  "processing_job_id": "job_789"
}
```

### Update Dataset

#### PUT /datasets/{dataset_id}
Update dataset configuration or metadata.

**Request Body:**
```json
{
  "name": "Updated Dataset Name",
  "description": "Updated description",
  "config": {
    "clean": {
      "min_quality_score": 0.7
    }
  }
}
```

### Delete Dataset

#### DELETE /datasets/{dataset_id}
Delete a dataset and all associated data.

**Response:**
```json
{
  "message": "Dataset deleted successfully",
  "deleted_at": "2024-01-15T10:30:00Z"
}
```

## Document Management

### List Documents

#### GET /datasets/{dataset_id}/documents
Retrieve documents from a specific dataset.

**Parameters:**
- `page` (int, optional): Page number
- `limit` (int, optional): Items per page
- `quality_min` (float, optional): Minimum quality score
- `domain` (string, optional): Filter by domain
- `language` (string, optional): Filter by language

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.qudata.com/v1/datasets/dataset_123/documents?quality_min=0.8&limit=5"
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_001",
      "title": "Machine Learning Fundamentals",
      "content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {
        "source_path": "ml_fundamentals.pdf",
        "file_type": "pdf",
        "language": "en",
        "domain": "machine_learning",
        "author": "Dr. Jane Smith",
        "creation_date": "2023-12-01T00:00:00Z",
        "quality_score": 0.92,
        "word_count": 3500,
        "entities": [
          {"text": "machine learning", "type": "TECHNOLOGY"},
          {"text": "Dr. Jane Smith", "type": "PERSON"}
        ]
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 5,
    "total": 1500,
    "pages": 300
  }
}
```

### Get Document Details

#### GET /datasets/{dataset_id}/documents/{document_id}
Retrieve detailed information about a specific document.

**Response:**
```json
{
  "id": "doc_001",
  "title": "Machine Learning Fundamentals",
  "content": "Full document content here...",
  "metadata": {
    "source_path": "ml_fundamentals.pdf",
    "file_type": "pdf",
    "file_size": 2048576,
    "language": "en",
    "language_confidence": 0.98,
    "domain": "machine_learning",
    "domain_confidence": 0.95,
    "author": "Dr. Jane Smith",
    "creation_date": "2023-12-01T00:00:00Z",
    "processing_date": "2024-01-10T10:15:00Z",
    "quality_metrics": {
      "overall_score": 0.92,
      "content_quality": 0.95,
      "language_quality": 0.90,
      "structure_quality": 0.91
    },
    "text_stats": {
      "word_count": 3500,
      "sentence_count": 180,
      "paragraph_count": 25,
      "readability_score": 0.75
    },
    "entities": [
      {
        "text": "machine learning",
        "type": "TECHNOLOGY",
        "confidence": 0.95,
        "start": 0,
        "end": 16
      }
    ],
    "topics": [
      {
        "topic": "supervised_learning",
        "confidence": 0.85
      }
    ]
  },
  "processing_history": [
    {
      "stage": "ingestion",
      "status": "completed",
      "timestamp": "2024-01-10T10:00:00Z",
      "duration": 2.5
    },
    {
      "stage": "cleaning",
      "status": "completed",
      "timestamp": "2024-01-10T10:05:00Z",
      "duration": 1.2
    }
  ]
}
```

## Processing Jobs

### List Processing Jobs

#### GET /jobs
Retrieve list of processing jobs.

**Parameters:**
- `status` (string, optional): Filter by status
- `dataset_id` (string, optional): Filter by dataset

**Response:**
```json
{
  "jobs": [
    {
      "id": "job_789",
      "dataset_id": "dataset_456",
      "status": "processing",
      "created_at": "2024-01-15T10:30:00Z",
      "started_at": "2024-01-15T10:31:00Z",
      "estimated_completion": "2024-01-15T11:30:00Z",
      "progress": {
        "current_stage": "cleaning",
        "completed_stages": ["ingestion"],
        "documents_processed": 150,
        "total_documents": 500,
        "percentage": 30
      }
    }
  ]
}
```

### Get Job Status

#### GET /jobs/{job_id}
Get detailed status of a processing job.

**Response:**
```json
{
  "id": "job_789",
  "dataset_id": "dataset_456",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "estimated_completion": "2024-01-15T11:30:00Z",
  "progress": {
    "current_stage": "cleaning",
    "completed_stages": ["ingestion"],
    "documents_processed": 150,
    "total_documents": 500,
    "percentage": 30,
    "stage_progress": {
      "ingestion": 100,
      "cleaning": 60,
      "annotation": 0,
      "quality_scoring": 0
    }
  },
  "metrics": {
    "processing_rate": 2.5,
    "estimated_time_remaining": 3600,
    "memory_usage": "1.2GB",
    "cpu_usage": 75
  },
  "errors": [],
  "warnings": [
    {
      "message": "3 files skipped due to corruption",
      "timestamp": "2024-01-15T10:45:00Z"
    }
  ]
}
```

### Cancel Job

#### DELETE /jobs/{job_id}
Cancel a running processing job.

**Response:**
```json
{
  "message": "Job cancelled successfully",
  "cancelled_at": "2024-01-15T10:45:00Z"
}
```

## Export and Download

### Export Dataset

#### POST /datasets/{dataset_id}/export
Export dataset in specified format.

**Request Body:**
```json
{
  "format": "jsonl",
  "options": {
    "include_metadata": true,
    "quality_threshold": 0.7,
    "split_data": true,
    "split_ratios": {
      "train": 0.8,
      "validation": 0.1,
      "test": 0.1
    }
  }
}
```

**Response:**
```json
{
  "export_id": "export_123",
  "status": "processing",
  "estimated_completion": "2024-01-15T11:00:00Z",
  "download_urls": {
    "train": "https://api.qudata.com/v1/downloads/export_123/train.jsonl",
    "validation": "https://api.qudata.com/v1/downloads/export_123/validation.jsonl",
    "test": "https://api.qudata.com/v1/downloads/export_123/test.jsonl"
  }
}
```

### Get Export Status

#### GET /exports/{export_id}
Check status of export job.

**Response:**
```json
{
  "id": "export_123",
  "dataset_id": "dataset_123",
  "format": "jsonl",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:45:00Z",
  "file_info": {
    "train": {
      "url": "https://api.qudata.com/v1/downloads/export_123/train.jsonl",
      "size": "1.2GB",
      "document_count": 1200,
      "expires_at": "2024-01-22T10:45:00Z"
    },
    "validation": {
      "url": "https://api.qudata.com/v1/downloads/export_123/validation.jsonl",
      "size": "150MB",
      "document_count": 150,
      "expires_at": "2024-01-22T10:45:00Z"
    },
    "test": {
      "url": "https://api.qudata.com/v1/downloads/export_123/test.jsonl",
      "size": "150MB",
      "document_count": 150,
      "expires_at": "2024-01-22T10:45:00Z"
    }
  }
}
```

### Download Files

#### GET /downloads/{export_id}/{filename}
Download exported files.

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -o train.jsonl \
     "https://api.qudata.com/v1/downloads/export_123/train.jsonl"
```

## Analysis and Statistics

### Dataset Analysis

#### GET /datasets/{dataset_id}/analysis
Get comprehensive analysis of dataset.

**Response:**
```json
{
  "dataset_id": "dataset_123",
  "analysis_date": "2024-01-15T10:30:00Z",
  "overview": {
    "document_count": 1500,
    "total_size": "2.5GB",
    "average_quality": 0.85,
    "languages": ["en", "es", "fr"],
    "domains": ["technology", "research", "education"]
  },
  "quality_distribution": {
    "high_quality": 1200,
    "medium_quality": 250,
    "low_quality": 50
  },
  "content_analysis": {
    "average_length": 3500,
    "length_distribution": {
      "short": 150,
      "medium": 1200,
      "long": 150
    },
    "top_keywords": [
      {"keyword": "machine learning", "frequency": 450},
      {"keyword": "artificial intelligence", "frequency": 380}
    ]
  },
  "topic_analysis": {
    "topics": [
      {
        "id": "topic_1",
        "name": "Machine Learning",
        "keywords": ["learning", "algorithm", "model"],
        "document_count": 450,
        "coherence_score": 0.75
      }
    ]
  },
  "language_analysis": {
    "primary_language": "en",
    "language_distribution": {
      "en": 95.0,
      "es": 3.0,
      "fr": 2.0
    },
    "multilingual_documents": 25
  }
}
```

### Document Statistics

#### GET /datasets/{dataset_id}/documents/{document_id}/stats
Get detailed statistics for a specific document.

**Response:**
```json
{
  "document_id": "doc_001",
  "text_statistics": {
    "word_count": 3500,
    "unique_words": 1200,
    "sentence_count": 180,
    "paragraph_count": 25,
    "average_sentence_length": 19.4,
    "readability_score": 0.75,
    "complexity_score": 0.68
  },
  "quality_breakdown": {
    "overall_score": 0.92,
    "content_quality": 0.95,
    "language_quality": 0.90,
    "structure_quality": 0.91,
    "issues": []
  },
  "entity_statistics": {
    "total_entities": 45,
    "entity_types": {
      "PERSON": 12,
      "ORG": 8,
      "TECHNOLOGY": 15,
      "GPE": 10
    }
  },
  "topic_scores": [
    {
      "topic": "machine_learning",
      "score": 0.85
    },
    {
      "topic": "data_science",
      "score": 0.72
    }
  ]
}
```

## Configuration Management

### Get Configuration Templates

#### GET /config/templates
Retrieve available configuration templates.

**Response:**
```json
{
  "templates": [
    {
      "name": "academic_papers",
      "description": "Optimized for academic paper processing",
      "use_cases": ["research", "academic"],
      "config": {
        "ingest": {
          "file_types": ["pdf"],
          "extract_citations": true
        },
        "clean": {
          "min_quality_score": 0.7
        }
      }
    },
    {
      "name": "web_content",
      "description": "Optimized for web article processing",
      "use_cases": ["news", "blogs", "articles"],
      "config": {
        "ingest": {
          "file_types": ["html"],
          "extract_main_content": true
        }
      }
    }
  ]
}
```

### Validate Configuration

#### POST /config/validate
Validate a configuration before using it.

**Request Body:**
```json
{
  "config": {
    "ingest": {
      "file_types": ["pdf", "docx"],
      "max_file_size": "100MB"
    },
    "clean": {
      "remove_duplicates": true,
      "similarity_threshold": 0.85
    }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "warnings": [
    {
      "field": "clean.similarity_threshold",
      "message": "High similarity threshold may remove valid content"
    }
  ],
  "errors": []
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

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

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | API key is missing or invalid |
| `INSUFFICIENT_PERMISSIONS` | 403 | API key lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource doesn't exist |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `PROCESSING_ERROR` | 422 | Error during data processing |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Free tier**: 100 requests per hour
- **Pro tier**: 1,000 requests per hour
- **Enterprise tier**: 10,000 requests per hour

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## Webhooks

### Configure Webhooks

#### POST /webhooks
Set up webhooks for processing events.

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["dataset.completed", "job.failed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Events

- `dataset.created`: New dataset created
- `dataset.completed`: Dataset processing completed
- `dataset.failed`: Dataset processing failed
- `job.started`: Processing job started
- `job.completed`: Processing job completed
- `job.failed`: Processing job failed
- `export.completed`: Export job completed

### Webhook Payload Example

```json
{
  "event": "dataset.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "dataset_id": "dataset_123",
    "name": "My Dataset",
    "status": "completed",
    "document_count": 1500,
    "quality_score": 0.85
  }
}
```

## SDK Examples

### Python SDK

```python
import qudata

# Initialize client
client = qudata.Client(api_key="your_api_key")

# Create dataset
dataset = client.datasets.create(
    name="My Dataset",
    files=["document1.pdf", "document2.docx"],
    config={
        "clean": {"min_quality_score": 0.7},
        "export": {"formats": ["jsonl"]}
    }
)

# Wait for processing
dataset.wait_for_completion()

# Export dataset
export = dataset.export(format="jsonl", split_data=True)
export.download("./exports/")
```

### JavaScript SDK

```javascript
const QuData = require('qudata-js');

const client = new QuData.Client({
  apiKey: 'your_api_key'
});

// Create dataset
const dataset = await client.datasets.create({
  name: 'My Dataset',
  files: ['document1.pdf', 'document2.docx'],
  config: {
    clean: { minQualityScore: 0.7 },
    export: { formats: ['jsonl'] }
  }
});

// Monitor progress
dataset.on('progress', (progress) => {
  console.log(`Progress: ${progress.percentage}%`);
});

// Wait for completion
await dataset.waitForCompletion();

// Export and download
const exportJob = await dataset.export({
  format: 'jsonl',
  splitData: true
});

await exportJob.download('./exports/');
```

This REST API documentation provides comprehensive coverage of all QuData API endpoints with detailed examples and usage patterns for integration into various applications and workflows.