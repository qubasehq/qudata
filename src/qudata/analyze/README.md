# Data Analysis and Reporting Engine

The analysis module provides comprehensive analytics capabilities for understanding dataset characteristics, quality metrics, and content patterns to inform data processing decisions and model training strategies.

## Overview

This module handles:
- **Text Statistics**: Word counts, token distributions, vocabulary analysis
- **Topic Modeling**: Automatic topic discovery using LDA and BERTopic
- **Sentiment Analysis**: Polarity scoring and emotion detection
- **Language Analysis**: Language distribution and detection statistics
- **Quality Analysis**: Multi-dimensional quality scoring and assessment
- **Content Analysis**: Readability, complexity, and coherence metrics
- **Comparative Analysis**: Dataset comparison and trend analysis

## Core Components

### Text Analysis

```python
from qudata.analyze import TextAnalyzer

analyzer = TextAnalyzer({
    'include_readability': True,
    'include_complexity': True,
    'extract_keywords': True,
    'top_n_keywords': 20
})

stats = analyzer.analyze_text(document.content)
print(f"Word count: {stats.word_count}")
print(f"Unique tokens: {stats.unique_tokens}")
print(f"Readability score: {stats.readability_score}")
print(f"Top keywords: {stats.top_keywords}")
```

**Text Metrics:**
- **Basic counts**: Words, sentences, paragraphs, characters
- **Vocabulary**: Unique tokens, type-token ratio, lexical diversity
- **Readability**: Flesch-Kincaid, SMOG, ARI scores
- **Complexity**: Syntactic complexity, sentence length variation
- **Keywords**: TF-IDF based keyword extraction

### Topic Modeling

```python
from qudata.analyze import TopicModeler

modeler = TopicModeler({
    'algorithm': 'bertopic',  # or 'lda', 'nmf'
    'num_topics': 10,
    'min_topic_size': 5,
    'language': 'en',
    'embedding_model': 'all-MiniLM-L6-v2'
})

# Analyze document collection
documents = [doc.content for doc in document_collection]
topic_result = modeler.fit_transform(documents)

print(f"Number of topics found: {len(topic_result.topics)}")
for topic_id, topic in topic_result.topics.items():
    print(f"Topic {topic_id}: {topic.keywords}")
    print(f"Representative docs: {len(topic.documents)}")
```

**Topic Modeling Features:**
- **Multiple algorithms**: BERTopic, LDA, NMF
- **Hierarchical topics**: Topic relationships and subtopics
- **Dynamic topics**: Topic evolution over time
- **Topic visualization**: Interactive topic maps and charts
- **Document-topic assignment**: Probability distributions

### Sentiment Analysis

```python
from qudata.analyze import SentimentAnalyzer

analyzer = SentimentAnalyzer({
    'model': 'vader',  # or 'textblob', 'transformers'
    'include_emotions': True,
    'batch_size': 100
})

sentiment = analyzer.analyze_sentiment(document.content)
print(f"Polarity: {sentiment.polarity}")      # -1 to 1
print(f"Subjectivity: {sentiment.subjectivity}")  # 0 to 1
print(f"Emotions: {sentiment.emotions}")      # joy, anger, fear, etc.
```

**Sentiment Features:**
- **Polarity scoring**: Positive, negative, neutral classification
- **Subjectivity analysis**: Objective vs subjective content
- **Emotion detection**: Joy, anger, fear, sadness, surprise
- **Aspect-based sentiment**: Sentiment towards specific topics
- **Confidence scoring**: Reliability of sentiment predictions

### Language Analysis

```python
from qudata.analyze import LanguageAnalyzer

analyzer = LanguageAnalyzer({
    'detect_languages': True,
    'confidence_threshold': 0.8,
    'include_dialects': True
})

lang_stats = analyzer.analyze_languages(document_collection)
print(f"Primary language: {lang_stats.primary_language}")
print(f"Language distribution: {lang_stats.distribution}")
print(f"Multilingual documents: {lang_stats.multilingual_count}")
```

**Language Features:**
- **Language detection**: 95+ languages supported
- **Confidence scoring**: Detection reliability metrics
- **Dialect identification**: Regional language variants
- **Code-switching detection**: Mixed-language content
- **Language distribution**: Dataset language composition

### Quality Analysis

```python
from qudata.analyze import QualityAnalyzer

analyzer = QualityAnalyzer({
    'dimensions': ['content', 'structure', 'language', 'coherence'],
    'weights': {'content': 0.4, 'structure': 0.2, 'language': 0.2, 'coherence': 0.2},
    'thresholds': {'min_quality': 0.6, 'high_quality': 0.8}
})

quality = analyzer.analyze_quality(document)
print(f"Overall quality: {quality.overall_score}")
print(f"Content quality: {quality.content_score}")
print(f"Language quality: {quality.language_score}")
print(f"Issues found: {quality.issues}")
```

**Quality Dimensions:**
- **Content quality**: Informativeness, relevance, completeness
- **Structure quality**: Organization, formatting, coherence
- **Language quality**: Grammar, spelling, fluency
- **Technical quality**: Encoding, formatting, metadata completeness

### Comprehensive Analysis Engine

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine({
    'text_analysis': True,
    'topic_modeling': True,
    'sentiment_analysis': True,
    'language_analysis': True,
    'quality_analysis': True,
    'generate_report': True
})

# Analyze entire dataset
analysis_result = engine.analyze_dataset(dataset)
print(f"Dataset size: {analysis_result.dataset_stats.document_count}")
print(f"Average quality: {analysis_result.quality_stats.average_score}")
print(f"Top topics: {analysis_result.topic_stats.top_topics}")
```

## Configuration

### Analysis Configuration

```yaml
# configs/analysis.yaml
analysis:
  text_analysis:
    enabled: true
    include_readability: true
    include_complexity: true
    extract_keywords: true
    top_n_keywords: 50
  
  topic_modeling:
    enabled: true
    algorithm: "bertopic"
    num_topics: 15
    min_topic_size: 10
    embedding_model: "all-MiniLM-L6-v2"
  
  sentiment_analysis:
    enabled: true
    model: "vader"
    include_emotions: true
    confidence_threshold: 0.7
  
  language_analysis:
    enabled: true
    detect_languages: true
    confidence_threshold: 0.8
    include_dialects: false
  
  quality_analysis:
    enabled: true
    dimensions: ["content", "structure", "language", "coherence"]
    weights:
      content: 0.4
      structure: 0.2
      language: 0.2
      coherence: 0.2
    thresholds:
      min_quality: 0.6
      high_quality: 0.8
```

### Advanced Configuration

```python
from qudata.analyze import AnalysisEngine

# Custom analysis configuration
config = {
    'parallel_processing': True,
    'batch_size': 100,
    'cache_results': True,
    'output_formats': ['json', 'html', 'pdf'],
    'visualization': {
        'generate_charts': True,
        'chart_types': ['bar', 'pie', 'scatter', 'heatmap'],
        'interactive': True
    }
}

engine = AnalysisEngine(config)
```

## Analysis Reports

### Dataset Overview Report

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine()
result = engine.analyze_dataset(dataset)

# Generate comprehensive report
report = engine.generate_report(result, format='html')
print(f"Report saved to: {report.file_path}")
```

**Report Sections:**
- **Executive Summary**: Key findings and recommendations
- **Dataset Statistics**: Size, composition, quality metrics
- **Content Analysis**: Topics, sentiment, language distribution
- **Quality Assessment**: Quality scores and improvement suggestions
- **Visualizations**: Charts, graphs, and interactive plots

### Custom Reports

```python
from qudata.analyze import AnalysisEngine

# Custom report template
report_config = {
    'sections': [
        'dataset_overview',
        'quality_analysis',
        'topic_analysis',
        'recommendations'
    ],
    'visualizations': [
        'quality_distribution',
        'topic_clusters',
        'language_pie_chart'
    ],
    'format': 'pdf',
    'include_raw_data': False
}

engine = AnalysisEngine()
report = engine.generate_custom_report(dataset, report_config)
```

## Performance Optimization

### Batch Processing

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine({
    'batch_processing': True,
    'batch_size': 1000,
    'parallel_workers': 4,
    'memory_limit': '4GB'
})

# Process large dataset efficiently
large_dataset = Dataset(documents=large_document_list)
result = engine.analyze_dataset(large_dataset)
```

### Caching and Incremental Analysis

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine({
    'cache_enabled': True,
    'cache_directory': '.cache/analysis',
    'incremental_analysis': True,
    'cache_ttl': 86400  # 24 hours
})

# Only analyze new or changed documents
result = engine.analyze_dataset_incremental(dataset, previous_analysis)
```

### Memory Management

```python
# Streaming analysis for very large datasets
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine({
    'streaming_mode': True,
    'chunk_size': 1000,
    'memory_efficient': True
})

def analyze_large_dataset(dataset_path):
    for chunk_result in engine.analyze_dataset_stream(dataset_path):
        yield chunk_result
```

## Visualization

### Built-in Visualizations

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine({'generate_visualizations': True})
result = engine.analyze_dataset(dataset)

# Access generated visualizations
for viz in result.visualizations:
    print(f"Chart: {viz.title} ({viz.type})")
    print(f"File: {viz.file_path}")
```

**Chart Types:**
- **Quality Distribution**: Histogram of quality scores
- **Topic Clusters**: 2D/3D topic visualization
- **Language Distribution**: Pie chart of languages
- **Sentiment Timeline**: Sentiment trends over time
- **Word Clouds**: Visual keyword representation
- **Correlation Heatmaps**: Feature correlation analysis

### Custom Visualizations

```python
from qudata.analyze import AnalysisEngine
import plotly.graph_objects as go

engine = AnalysisEngine()
result = engine.analyze_dataset(dataset)

# Create custom visualization
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=result.quality_stats.scores,
    y=result.text_stats.word_counts,
    mode='markers',
    name='Quality vs Length'
))

fig.update_layout(
    title='Document Quality vs Word Count',
    xaxis_title='Quality Score',
    yaxis_title='Word Count'
)

fig.show()
```

## Examples

### Basic Dataset Analysis

```python
from qudata.analyze import AnalysisEngine
from qudata.models import Dataset

# Load dataset
dataset = Dataset.load_from_directory("data/processed")

# Initialize analysis engine
engine = AnalysisEngine({
    'text_analysis': True,
    'quality_analysis': True,
    'generate_report': True
})

# Perform analysis
result = engine.analyze_dataset(dataset)

# Print summary
print(f"Documents analyzed: {result.dataset_stats.document_count}")
print(f"Average quality: {result.quality_stats.average_score:.2f}")
print(f"Languages detected: {len(result.language_stats.languages)}")
print(f"Total words: {result.text_stats.total_words:,}")
```

### Topic Analysis Workflow

```python
from qudata.analyze import TopicModeler, AnalysisEngine

# Initialize topic modeler
modeler = TopicModeler({
    'algorithm': 'bertopic',
    'num_topics': 20,
    'min_topic_size': 15
})

# Extract document texts
documents = [doc.content for doc in dataset.documents]

# Perform topic modeling
topic_result = modeler.fit_transform(documents)

# Analyze topic quality
for topic_id, topic in topic_result.topics.items():
    coherence = modeler.calculate_coherence(topic_id)
    print(f"Topic {topic_id}: {topic.keywords[:5]}")
    print(f"Coherence: {coherence:.3f}")
    print(f"Documents: {len(topic.documents)}")
    print("---")
```

### Quality Assessment Pipeline

```python
from qudata.analyze import QualityAnalyzer

analyzer = QualityAnalyzer({
    'dimensions': ['content', 'language', 'structure'],
    'detailed_analysis': True
})

# Analyze document quality
quality_results = []
for document in dataset.documents:
    quality = analyzer.analyze_quality(document)
    quality_results.append({
        'document_id': document.id,
        'overall_score': quality.overall_score,
        'content_score': quality.content_score,
        'language_score': quality.language_score,
        'structure_score': quality.structure_score,
        'issues': quality.issues
    })

# Find low-quality documents
low_quality = [r for r in quality_results if r['overall_score'] < 0.6]
print(f"Low quality documents: {len(low_quality)}")

# Identify common issues
all_issues = []
for result in quality_results:
    all_issues.extend(result['issues'])

from collections import Counter
common_issues = Counter(all_issues).most_common(10)
print("Most common quality issues:")
for issue, count in common_issues:
    print(f"  {issue}: {count}")
```

### Comparative Analysis

```python
from qudata.analyze import AnalysisEngine

engine = AnalysisEngine()

# Analyze multiple datasets
dataset_v1 = Dataset.load_from_directory("data/v1")
dataset_v2 = Dataset.load_from_directory("data/v2")

result_v1 = engine.analyze_dataset(dataset_v1)
result_v2 = engine.analyze_dataset(dataset_v2)

# Compare results
comparison = engine.compare_analyses(result_v1, result_v2)
print(f"Quality improvement: {comparison.quality_change:.2f}")
print(f"Size change: {comparison.size_change}")
print(f"New topics: {comparison.new_topics}")
```

## Testing

```bash
# Run analysis module tests
pytest tests/unit/test_analysis_engine.py
pytest tests/unit/test_text_analyzer.py
pytest tests/unit/test_topic_modeler.py

# Run integration tests
pytest tests/integration/test_analysis_pipeline.py -v

# Run performance tests
pytest tests/benchmarks/test_analysis_performance.py
```

## Dependencies

**Core Dependencies:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms
- `nltk`: Natural language processing
- `textstat`: Readability metrics

**Optional Dependencies:**
- `bertopic`: Advanced topic modeling (install with `pip install qudata[ml]`)
- `transformers`: Transformer-based models (install with `pip install qudata[ml]`)
- `plotly`: Interactive visualizations
- `wordcloud`: Word cloud generation
- `seaborn`: Statistical visualizations

## Troubleshooting

### Common Issues

**Memory Issues with Large Datasets:**
```python
# Use streaming analysis
engine = AnalysisEngine({
    'streaming_mode': True,
    'chunk_size': 500,
    'memory_limit': '2GB'
})
```

**Slow Topic Modeling:**
```python
# Use faster algorithms or smaller models
modeler = TopicModeler({
    'algorithm': 'lda',  # Faster than BERTopic
    'num_topics': 10,    # Fewer topics
    'max_iter': 100      # Fewer iterations
})
```

**Poor Topic Quality:**
```python
# Adjust parameters
modeler = TopicModeler({
    'min_topic_size': 20,     # Larger minimum size
    'n_gram_range': (1, 2),   # Include bigrams
    'remove_stopwords': True,
    'min_df': 5,              # Minimum document frequency
    'max_df': 0.8             # Maximum document frequency
})
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `text_analyzer.py`: Text statistics and metrics
- `topic_modeler.py`: Topic modeling algorithms
- `sentiment_analyzer.py`: Sentiment and emotion analysis
- `language_analyzer.py`: Language detection and analysis
- `quality_analyzer.py`: Quality assessment and scoring
- `analysis_engine.py`: Comprehensive analysis orchestration