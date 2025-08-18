# Quality Scoring and Assessment

The score module provides comprehensive quality assessment capabilities for processed documents and datasets.

## Components

### QualityScorer (`quality.py`)
- Multi-dimensional quality scoring system
- Configurable quality metrics and weights
- Statistical analysis of content quality
- Threshold-based filtering and categorization

## Quality Dimensions

### Content Quality
- **Length Score**: Evaluates document length appropriateness
- **Language Score**: Assesses language detection confidence
- **Coherence Score**: Measures text coherence and readability
- **Uniqueness Score**: Detects duplicate and near-duplicate content
- **Structure Score**: Evaluates document structure and formatting

### Technical Quality
- **Encoding Quality**: Checks for encoding issues and corruption
- **Format Compliance**: Validates format-specific requirements
- **Metadata Completeness**: Assesses metadata richness and accuracy
- **Processing Success**: Tracks successful processing stages

## Usage Examples

### Basic Quality Scoring
```python
from qudata.score import QualityScorer

scorer = QualityScorer()
quality_score = scorer.score_document(document)

print(f"Overall Score: {quality_score.overall_score}")
print(f"Length Score: {quality_score.length_score}")
print(f"Language Score: {quality_score.language_score}")
```

### Batch Scoring
```python
scores = scorer.score_documents(document_list)
high_quality_docs = [
    doc for doc, score in zip(document_list, scores)
    if score.overall_score >= 0.8
]
```

### Custom Scoring Configuration
```python
config = {
    "weights": {
        "length": 0.2,
        "language": 0.3,
        "coherence": 0.3,
        "uniqueness": 0.2
    },
    "thresholds": {
        "min_length": 100,
        "max_length": 10000,
        "min_language_confidence": 0.8
    }
}

scorer = QualityScorer(config)
```

## Quality Metrics

### Overall Score Calculation
```
overall_score = (
    length_score * weight_length +
    language_score * weight_language +
    coherence_score * weight_coherence +
    uniqueness_score * weight_uniqueness
)
```

### Individual Metrics
- **Length Score**: `min(1.0, length / optimal_length)`
- **Language Score**: Language detection confidence (0.0-1.0)
- **Coherence Score**: Based on sentence structure and flow
- **Uniqueness Score**: `1.0 - similarity_to_existing_content`

## Configuration

Quality scoring can be configured through YAML:

```yaml
quality:
  scoring:
    weights:
      length: 0.2
      language: 0.3
      coherence: 0.3
      uniqueness: 0.2
    thresholds:
      min_length: 100
      max_length: 10000
      min_language_confidence: 0.8
      min_coherence_score: 0.6
      max_duplicate_similarity: 0.9
  filtering:
    enable: true
    min_overall_score: 0.7
    remove_duplicates: true
```

## Quality Reports

### Document-Level Reports
```python
report = scorer.generate_document_report(document)
print(report.summary)
print(report.recommendations)
```

### Dataset-Level Reports
```python
dataset_report = scorer.generate_dataset_report(documents)
print(f"Average Quality: {dataset_report.average_score}")
print(f"Quality Distribution: {dataset_report.score_distribution}")
```

## Integration

The scoring module integrates with:
- Cleaning pipeline for quality-based filtering
- Export pipeline for quality thresholds
- Validation system for quality benchmarks
- Visualization for quality dashboards
- Analysis engine for quality trends

## Performance Optimization

- Efficient similarity computation using hashing
- Cached language detection results
- Parallel processing for large document sets
- Memory-efficient streaming for huge datasets
- Configurable batch sizes for optimal performance