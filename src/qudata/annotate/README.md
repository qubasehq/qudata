# Annotation and Metadata Extraction Module

The annotation module provides intelligent document categorization, metadata extraction, and content tagging capabilities to enrich processed documents with structured information.

## Overview

This module handles:
- **Taxonomy Classification**: Domain and topic categorization using rule-based and ML approaches
- **Metadata Extraction**: Author, date, source, and document properties extraction
- **Named Entity Recognition**: Person, organization, location, and custom entity detection
- **Topic Modeling**: Automatic topic discovery and assignment
- **Content Safety**: Inappropriate content detection and filtering
- **Cross-Document Linking**: Related content identification and relationship mapping

## Core Components

### Taxonomy Classification

```python
from qudata.annotate import TaxonomyClassifier

classifier = TaxonomyClassifier({
    'taxonomy_file': 'configs/taxonomy.yaml',
    'confidence_threshold': 0.7,
    'max_categories': 3
})

result = classifier.classify_document(document)
print(result.primary_category)  # "technology"
print(result.categories)        # ["technology", "software", "ai"]
print(result.confidence_scores) # {"technology": 0.95, "software": 0.82}
```

**Classification Methods:**
- **Rule-based**: Keyword and pattern matching
- **ML-based**: TF-IDF and neural classifiers
- **Hybrid**: Combination of rule-based and ML approaches
- **Custom**: User-defined classification logic

### Metadata Extraction

```python
from qudata.annotate import MetadataExtractor

extractor = MetadataExtractor({
    'extract_dates': True,
    'extract_authors': True,
    'extract_sources': True,
    'date_formats': ['%Y-%m-%d', '%B %d, %Y']
})

metadata = extractor.extract_metadata(document)
print(metadata.author)        # "John Doe"
print(metadata.creation_date) # datetime(2024, 1, 15)
print(metadata.source_url)    # "https://example.com/article"
```

**Extracted Metadata:**
- **Author information**: Names, affiliations, contact details
- **Temporal data**: Creation, modification, publication dates
- **Source information**: URLs, citations, references
- **Document properties**: Language, format, size, encoding
- **Content metrics**: Word count, reading time, complexity

### Named Entity Recognition

```python
from qudata.annotate import EntityRecognizer

recognizer = EntityRecognizer({
    'model': 'en_core_web_sm',  # spaCy model
    'entity_types': ['PERSON', 'ORG', 'GPE', 'DATE'],
    'confidence_threshold': 0.8,
    'custom_entities': {
        'PRODUCT': ['iPhone', 'Android', 'Windows'],
        'TECHNOLOGY': ['AI', 'ML', 'blockchain']
    }
})

entities = recognizer.extract_entities(document)
for entity in entities:
    print(f"{entity.text} ({entity.label_}): {entity.confidence}")
```

**Entity Types:**
- **Standard**: PERSON, ORG, GPE, DATE, MONEY, PERCENT
- **Custom**: User-defined entity types and patterns
- **Domain-specific**: Technical terms, product names, etc.
- **Multilingual**: Support for multiple languages

### Topic Modeling

```python
from qudata.annotate import TopicModeler

modeler = TopicModeler({
    'algorithm': 'bertopic',  # or 'lda', 'nmf'
    'num_topics': 10,
    'min_topic_size': 5,
    'language': 'en'
})

# Train on document collection
topics = modeler.fit_transform(document_collection)

# Assign topics to new documents
topic_result = modeler.assign_topics(new_document)
print(topic_result.topics)      # [("AI/ML", 0.85), ("Technology", 0.72)]
print(topic_result.keywords)    # ["artificial", "intelligence", "machine"]
```

**Algorithms:**
- **BERTopic**: Transformer-based topic modeling
- **LDA**: Latent Dirichlet Allocation
- **NMF**: Non-negative Matrix Factorization
- **Custom**: User-defined topic modeling approaches

### Content Safety Detection

```python
from qudata.annotate import SafetyDetector

detector = SafetyDetector({
    'check_toxicity': True,
    'check_bias': True,
    'check_privacy': True,
    'toxicity_threshold': 0.7
})

safety_result = detector.analyze_content(document)
print(safety_result.is_safe)           # True/False
print(safety_result.toxicity_score)    # 0.0-1.0
print(safety_result.detected_issues)   # ["potential_bias", "pii_detected"]
```

**Safety Checks:**
- **Toxicity detection**: Hate speech, harassment, threats
- **Bias detection**: Gender, racial, cultural bias
- **Privacy protection**: PII, sensitive information
- **Content appropriateness**: Age-appropriate content filtering

## Configuration

### Taxonomy Configuration

```yaml
# configs/taxonomy.yaml
taxonomy:
  categories:
    technology:
      keywords: ["AI", "machine learning", "software", "programming"]
      patterns: ["\\b(artificial intelligence|AI)\\b"]
      subcategories:
        - software_development
        - data_science
        - cybersecurity
    
    business:
      keywords: ["finance", "marketing", "strategy", "management"]
      patterns: ["\\b(business|corporate|enterprise)\\b"]
      subcategories:
        - finance
        - marketing
        - operations
  
  classification:
    method: "hybrid"  # rule_based, ml_based, hybrid
    confidence_threshold: 0.7
    max_categories: 3
    fallback_category: "general"
```

### Metadata Extraction Configuration

```yaml
# configs/metadata.yaml
metadata:
  extraction:
    authors:
      patterns:
        - "By\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)"
        - "Author:\\s*([^\\n]+)"
      sources: ["byline", "meta_tags", "structured_data"]
    
    dates:
      formats:
        - "%Y-%m-%d"
        - "%B %d, %Y"
        - "%d/%m/%Y"
      sources: ["meta_tags", "content", "filename"]
    
    sources:
      extract_urls: true
      extract_citations: true
      validate_urls: true
```

### Entity Recognition Configuration

```python
from qudata.annotate import EntityRecognizer

# Custom entity configuration
config = {
    'model': 'en_core_web_sm',
    'entity_types': ['PERSON', 'ORG', 'GPE'],
    'custom_patterns': [
        {'label': 'PRODUCT', 'pattern': [{'LOWER': {'IN': ['iphone', 'android']}}]},
        {'label': 'TECH', 'pattern': [{'LOWER': 'ai'}, {'LOWER': 'model'}]}
    ],
    'entity_linking': {
        'enabled': True,
        'knowledge_base': 'wikidata'
    }
}

recognizer = EntityRecognizer(config)
```

## Advanced Features

### Cross-Document Linking

```python
from qudata.annotate import CrossReferencer

referencer = CrossReferencer({
    'similarity_threshold': 0.8,
    'max_references': 5,
    'reference_types': ['similar_content', 'citations', 'topics']
})

# Find related documents
references = referencer.find_references(document, document_collection)
for ref in references:
    print(f"Related: {ref.document_id} (similarity: {ref.similarity})")
```

### Batch Processing

```python
from qudata.annotate import TaxonomyClassifier

classifier = TaxonomyClassifier()

# Process multiple documents
documents = [doc1, doc2, doc3, doc4]
results = classifier.classify_batch(documents, batch_size=10)

for doc_id, result in results.items():
    print(f"{doc_id}: {result.primary_category}")
```

### Custom Classification Rules

```python
from qudata.annotate import TaxonomyClassifier

# Define custom classification logic
def custom_classifier(document):
    content = document.content.lower()
    if 'python' in content and 'programming' in content:
        return 'python_programming'
    elif 'machine learning' in content:
        return 'ml_ai'
    return 'general'

classifier = TaxonomyClassifier({
    'custom_classifier': custom_classifier,
    'fallback_to_rules': True
})
```

## Performance Optimization

### Caching

```python
from qudata.annotate import TaxonomyClassifier

classifier = TaxonomyClassifier({
    'cache_classifications': True,
    'cache_size': 10000,
    'cache_ttl': 3600  # 1 hour
})
```

### Parallel Processing

```python
from qudata.annotate import EntityRecognizer
import concurrent.futures

recognizer = EntityRecognizer()
documents = [doc1, doc2, doc3, doc4]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(recognizer.extract_entities, documents))
```

### Model Optimization

```python
# Use lightweight models for faster processing
from qudata.annotate import EntityRecognizer

recognizer = EntityRecognizer({
    'model': 'en_core_web_sm',  # Smaller, faster model
    'disable_pipes': ['parser', 'tagger'],  # Disable unused components
    'batch_size': 100  # Process in batches
})
```

## Error Handling

```python
from qudata.annotate import TaxonomyClassifier
from qudata.models import ProcessingError

classifier = TaxonomyClassifier()

try:
    result = classifier.classify_document(document)
    if result.confidence < 0.5:
        print("Low confidence classification")
except ProcessingError as e:
    print(f"Classification failed: {e.message}")
```

## Examples

### Document Classification Pipeline

```python
from qudata.annotate import TaxonomyClassifier, MetadataExtractor

# Initialize components
classifier = TaxonomyClassifier({'taxonomy_file': 'configs/taxonomy.yaml'})
metadata_extractor = MetadataExtractor()

def annotate_document(document):
    # Extract metadata
    metadata = metadata_extractor.extract_metadata(document)
    document.metadata.author = metadata.author
    document.metadata.creation_date = metadata.creation_date
    
    # Classify content
    classification = classifier.classify_document(document)
    document.metadata.domain = classification.primary_category
    document.metadata.topics = classification.categories
    
    return document

# Process document
annotated_doc = annotate_document(raw_document)
print(f"Category: {annotated_doc.metadata.domain}")
print(f"Author: {annotated_doc.metadata.author}")
```

### Entity Extraction and Linking

```python
from qudata.annotate import EntityRecognizer

recognizer = EntityRecognizer({
    'model': 'en_core_web_sm',
    'entity_linking': True
})

def extract_and_link_entities(document):
    entities = recognizer.extract_entities(document)
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        if entity.label_ not in entity_groups:
            entity_groups[entity.label_] = []
        entity_groups[entity.label_].append({
            'text': entity.text,
            'confidence': entity.confidence,
            'linked_id': entity.linked_id if hasattr(entity, 'linked_id') else None
        })
    
    return entity_groups

# Extract entities
entities = extract_and_link_entities(document)
print(f"People: {entities.get('PERSON', [])}")
print(f"Organizations: {entities.get('ORG', [])}")
```

### Topic Modeling Workflow

```python
from qudata.annotate import TopicModeler

# Initialize topic modeler
modeler = TopicModeler({
    'algorithm': 'bertopic',
    'num_topics': 15,
    'min_topic_size': 10
})

# Train on document collection
documents = [doc.content for doc in document_collection]
topic_model = modeler.fit(documents)

# Get topic information
topics = modeler.get_topic_info()
for topic_id, topic_info in topics.items():
    print(f"Topic {topic_id}: {topic_info['keywords']}")

# Assign topics to new documents
for document in new_documents:
    topic_result = modeler.assign_topics(document.content)
    document.metadata.topics = [topic[0] for topic in topic_result.topics]
```

## Testing

```bash
# Run annotation module tests
pytest tests/unit/test_taxonomy_classifier.py
pytest tests/unit/test_metadata_extractor.py
pytest tests/unit/test_entity_recognition.py

# Run integration tests
pytest tests/integration/test_annotation_pipeline.py -v
```

## Dependencies

**Core Dependencies:**
- `spacy`: Named entity recognition and NLP
- `nltk`: Natural language processing utilities
- `scikit-learn`: Machine learning algorithms
- `pyyaml`: Configuration file parsing

**Optional Dependencies:**
- `transformers`: Advanced NLP models (install with `pip install qudata[ml]`)
- `bertopic`: Advanced topic modeling (install with `pip install qudata[ml]`)
- `sentence-transformers`: Semantic similarity

## Troubleshooting

### Common Issues

**spaCy Model Not Found:**
```bash
# Download required spaCy model
python -m spacy download en_core_web_sm

# Or use a different model
python -m spacy download en_core_web_lg  # Larger, more accurate
```

**Low Classification Accuracy:**
```python
# Adjust confidence threshold
classifier = TaxonomyClassifier({
    'confidence_threshold': 0.5,  # Lower threshold
    'fallback_category': 'general'
})

# Add more training data or rules
classifier.add_training_examples([
    ('This is about AI and machine learning', 'technology'),
    ('Financial markets and trading', 'finance')
])
```

**Memory Issues with Large Models:**
```python
# Use smaller models
recognizer = EntityRecognizer({
    'model': 'en_core_web_sm',  # Instead of en_core_web_lg
    'disable_pipes': ['parser', 'tagger'],
    'batch_size': 50  # Smaller batches
})
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `taxonomy.py`: Document classification and categorization
- `metadata.py`: Metadata extraction and parsing
- `ner.py`: Named entity recognition and linking
- `topics.py`: Topic modeling and assignment
- `safety.py`: Content safety and appropriateness detection