# Text Cleaning and Preprocessing Module

The cleaning module provides comprehensive text cleaning and preprocessing capabilities to transform raw extracted text into high-quality, normalized content suitable for LLM training.

## Overview

This module handles:
- **Text Normalization**: Unicode normalization, whitespace cleanup, encoding detection
- **OCR Error Correction**: Common OCR error pattern correction
- **Deduplication**: Exact and near-duplicate detection and removal
- **Boilerplate Removal**: Headers, footers, watermarks, and navigation removal
- **HTML Cleaning**: Tag removal, emoji handling, and content extraction
- **Text Segmentation**: Sentence segmentation and paragraph normalization
- **Stopword Removal**: Configurable stopword filtering
- **Tokenization Preview**: LLM compatibility checking and token counting
- **Integrated Pipelines**: End-to-end cleaning workflows

## Core Components

### Text Normalization

```python
from qudata.clean import TextNormalizer, normalize_text_pipeline

normalizer = TextNormalizer({
    'normalize_unicode': True,
    'fix_encoding': True,
    'normalize_whitespace': True
})

result = normalizer.normalize("Messy   text\u00A0with\tnon-standard\r\nspacing")
print(result.normalized_text)  # Clean, normalized text
```

**Features:**
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Encoding detection and UTF-8 conversion
- Whitespace normalization and cleanup
- Line ending standardization
- Control character removal

### OCR Error Correction

```python
from qudata.clean import OCRCorrector

corrector = OCRCorrector({
    'fix_common_errors': True,
    'confidence_threshold': 0.8
})

corrected = corrector.correct("Th1s 1s s0me 0CR err0rs")
print(corrected.corrected_text)  # "This is some OCR errors"
```

**Common Corrections:**
- Number-letter substitutions (0→O, 1→I, 5→S)
- Character recognition errors (rn→m, cl→d)
- Word boundary issues
- Punctuation corrections

### Deduplication Engine

```python
from qudata.clean import DeduplicationEngine

deduper = DeduplicationEngine({
    'similarity_threshold': 0.85,
    'algorithm': 'jaccard',  # or 'cosine', 'levenshtein'
    'chunk_size': 1000
})

documents = ["Text 1", "Text 2", "Similar to Text 1", "Text 3"]
result = deduper.find_duplicates(documents)

for group in result.duplicate_groups:
    print(f"Duplicates: {group.document_ids}")
```

**Algorithms:**
- **Jaccard similarity**: Token-based similarity
- **Cosine similarity**: Vector-based similarity
- **Levenshtein distance**: Character-based similarity
- **MinHash**: Efficient approximate similarity

### Boilerplate Removal

```python
from qudata.clean import BoilerplateRemover

remover = BoilerplateRemover({
    'remove_headers': True,
    'remove_footers': True,
    'remove_navigation': True,
    'custom_patterns': [
        r'Copyright \d{4}.*',
        r'All rights reserved.*'
    ]
})

cleaned = remover.remove_boilerplate(text_with_boilerplate)
```

**Removal Patterns:**
- Headers and footers
- Navigation menus
- Copyright notices
- Watermarks and stamps
- Repeated disclaimers
- Social media widgets
- Advertisement blocks

### HTML Cleaning

```python
from qudata.clean import HTMLCleaner, clean_html_content

cleaner = HTMLCleaner({
    'remove_tags': True,
    'remove_emojis': True,
    'preserve_links': False,
    'extract_text_only': True
})

result = cleaner.clean("<p>Hello <b>world</b>! 😊</p>")
print(result.cleaned_text)  # "Hello world!"
```

**Features:**
- HTML tag removal with content preservation
- Emoji and special character handling
- Link extraction and preservation options
- Table structure preservation
- Script and style tag removal
- Attribute cleaning

### Text Segmentation

```python
from qudata.clean import SentenceSegmenter, segment_text_simple

segmenter = SentenceSegmenter({
    'language': 'en',
    'preserve_paragraphs': True,
    'min_sentence_length': 10
})

result = segmenter.segment(long_text)
for sentence in result.sentences:
    print(sentence)
```

**Features:**
- Language-aware sentence boundary detection
- Paragraph preservation
- Abbreviation handling
- Quote and parentheses handling
- Minimum length filtering

### Stopword Removal

```python
from qudata.clean import StopwordRemover, load_stopwords_from_file

# Built-in stopwords
remover = StopwordRemover({
    'language': 'en',
    'preserve_case': False
})

# Custom stopwords
custom_stopwords = load_stopwords_from_file("custom_stopwords.txt")
remover = StopwordRemover({
    'custom_stopwords': custom_stopwords
})

result = remover.remove_stopwords("This is a sample text with stopwords")
```

**Supported Languages:**
- English, Spanish, French, German, Italian
- Portuguese, Dutch, Russian, Chinese, Japanese
- Custom stopword lists supported

### Tokenization Preview

```python
from qudata.clean import TokenizationPreview, quick_token_count

preview = TokenizationPreview({
    'tokenizer': 'gpt-4',  # or 'claude', 'llama'
    'max_context_length': 4096
})

result = preview.analyze_text(long_document)
print(f"Token count: {result.token_count}")
print(f"Fits in context: {result.fits_in_context}")
print(f"Estimated cost: ${result.estimated_cost}")
```

**Features:**
- Multiple tokenizer support (GPT, Claude, LLaMA)
- Context length validation
- Cost estimation for API usage
- Batch processing statistics
- Token distribution analysis

### Integrated Cleaning Pipeline

```python
from qudata.clean import ComprehensiveCleaningPipeline

pipeline = ComprehensiveCleaningPipeline({
    'normalize_text': True,
    'correct_ocr': True,
    'remove_duplicates': True,
    'remove_boilerplate': True,
    'clean_html': True,
    'segment_text': False,
    'remove_stopwords': False,
    'preview_tokens': True
})

# Single document
result = pipeline.clean_document("document_id", raw_text)
print(result.cleaned_text)

# Batch processing
documents = {"doc1": "text1", "doc2": "text2"}
batch_result = pipeline.clean_documents(documents)
```

## Configuration

### Basic Configuration

```yaml
# configs/cleaning.yaml
cleaning:
  normalization:
    unicode_form: "NFKC"
    fix_encoding: true
    normalize_whitespace: true
  
  ocr_correction:
    enabled: true
    confidence_threshold: 0.8
  
  deduplication:
    enabled: true
    similarity_threshold: 0.85
    algorithm: "jaccard"
  
  boilerplate:
    remove_headers: true
    remove_footers: true
    custom_patterns:
      - "Copyright \\d{4}.*"
      - "All rights reserved.*"
  
  html_cleaning:
    remove_tags: true
    remove_emojis: true
    preserve_links: false
  
  tokenization:
    tokenizer: "gpt-4"
    max_context_length: 4096
```

### Advanced Configuration

```python
from qudata.clean import ComprehensiveCleaningPipeline

# Custom pipeline configuration
config = {
    'stages': [
        'normalize_text',
        'correct_ocr',
        'clean_html',
        'remove_boilerplate',
        'remove_duplicates'
    ],
    'parallel_processing': True,
    'batch_size': 100,
    'error_handling': 'skip',  # or 'fail', 'warn'
    'preserve_original': True
}

pipeline = ComprehensiveCleaningPipeline(config)
```

## Performance Optimization

### Batch Processing

```python
from qudata.clean import ComprehensiveCleaningPipeline

pipeline = ComprehensiveCleaningPipeline({
    'batch_size': 1000,
    'parallel_processing': True,
    'max_workers': 4
})

# Process large document collection
large_documents = {f"doc_{i}": f"content_{i}" for i in range(10000)}
result = pipeline.clean_documents(large_documents)
```

### Memory Management

```python
# Streaming processing for very large texts
from qudata.clean import TextNormalizer

normalizer = TextNormalizer({'streaming_mode': True})

def process_large_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for chunk in normalizer.normalize_stream(f, chunk_size=1024*1024):
            yield chunk.normalized_text
```

### Caching

```python
# Enable caching for repeated operations
from qudata.clean import DeduplicationEngine

deduper = DeduplicationEngine({
    'cache_similarities': True,
    'cache_size': 10000,
    'cache_ttl': 3600  # 1 hour
})
```

## Error Handling

```python
from qudata.clean import ComprehensiveCleaningPipeline
from qudata.models import ProcessingError

pipeline = ComprehensiveCleaningPipeline()

try:
    result = pipeline.clean_document("doc_id", problematic_text)
    if result.errors:
        for error in result.errors:
            print(f"Warning: {error.message}")
except ProcessingError as e:
    print(f"Critical error: {e.message}")
```

**Error Types:**
- `EncodingError`: Text encoding issues
- `NormalizationError`: Unicode normalization failures
- `DeduplicationError`: Similarity calculation errors
- `SegmentationError`: Text segmentation failures
- `TokenizationError`: Tokenizer-related errors

## Examples

### Basic Text Cleaning

```python
from qudata.clean import normalize_text_pipeline, clean_html_content

# Quick text normalization
clean_text = normalize_text_pipeline(
    "Messy\u00A0text\twith\r\nweird\u200Bspacing"
)

# HTML content cleaning
html_content = "<div><p>Hello <b>world</b>!</p></div>"
clean_content = clean_html_content(html_content)
```

### Document Deduplication

```python
from qudata.clean import DeduplicationEngine

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown fox jumps over a lazy dog.",  # Similar
    "The weather is nice today.",
    "Today the weather is nice.",  # Similar
    "Completely different content here."
]

deduper = DeduplicationEngine({'similarity_threshold': 0.8})
result = deduper.find_duplicates(documents)

# Remove duplicates
unique_docs = deduper.remove_duplicates(documents)
print(f"Original: {len(documents)}, Unique: {len(unique_docs)}")
```

### OCR Error Correction

```python
from qudata.clean import OCRCorrector

# Common OCR errors
ocr_text = """
Th1s d0cument c0nta1ns many 0CR err0rs.
The w0rds are n0t rec0gn1zed c0rrectly.
S0me characters l1ke 'rn' bec0me 'm'.
"""

corrector = OCRCorrector({
    'fix_common_errors': True,
    'custom_corrections': {
        'c0nta1ns': 'contains',
        'rec0gn1zed': 'recognized'
    }
})

corrected = corrector.correct(ocr_text)
print(corrected.corrected_text)
```

### Pipeline Customization

```python
from qudata.clean import ComprehensiveCleaningPipeline

# Create custom cleaning pipeline
pipeline = ComprehensiveCleaningPipeline({
    'stages': [
        'normalize_text',
        'clean_html',
        'correct_ocr',
        'remove_boilerplate',
        'segment_text'
    ],
    'stage_configs': {
        'normalize_text': {
            'unicode_form': 'NFKC',
            'normalize_whitespace': True
        },
        'clean_html': {
            'remove_emojis': True,
            'preserve_links': True
        },
        'segment_text': {
            'min_sentence_length': 15,
            'preserve_paragraphs': True
        }
    }
})

# Process documents
result = pipeline.clean_document("doc1", raw_document)
```

## Testing

```bash
# Run cleaning module tests
pytest tests/unit/test_cleaning_pipeline.py
pytest tests/unit/test_text_normalization.py
pytest tests/unit/test_deduplication.py

# Run specific component tests
pytest tests/unit/test_boilerplate_removal.py -v
pytest tests/unit/test_language_detection.py -v
```

## Dependencies

**Core Dependencies:**
- `nltk`: Natural language processing
- `spacy`: Advanced NLP features
- `langdetect`: Language detection
- `chardet`: Character encoding detection
- `beautifulsoup4`: HTML parsing
- `regex`: Advanced regular expressions

**Optional Dependencies:**
- `transformers`: Advanced tokenization (install with `pip install qudata[ml]`)
- `scikit-learn`: Similarity calculations
- `numpy`: Numerical operations

## Troubleshooting

### Common Issues

**Unicode Errors:**
```python
# Handle encoding issues
from qudata.clean import TextNormalizer

normalizer = TextNormalizer({
    'encoding_detection': True,
    'fallback_encoding': 'latin-1',
    'error_handling': 'replace'
})
```

**Memory Issues with Large Documents:**
```python
# Use streaming mode
from qudata.clean import ComprehensiveCleaningPipeline

pipeline = ComprehensiveCleaningPipeline({
    'streaming_mode': True,
    'chunk_size': 1024 * 1024,  # 1MB chunks
    'max_memory_usage': '2GB'
})
```

**Slow Deduplication:**
```python
# Optimize for speed
from qudata.clean import DeduplicationEngine

deduper = DeduplicationEngine({
    'algorithm': 'minhash',  # Faster for large datasets
    'num_perm': 128,  # Reduce for speed, increase for accuracy
    'parallel_processing': True
})
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `normalize.py`: Text normalization and OCR correction
- `dedupe.py`: Deduplication algorithms
- `boilerplate.py`: Boilerplate pattern removal
- `html_cleaner.py`: HTML content cleaning
- `segment.py`: Text segmentation
- `stopwords.py`: Stopword removal
- `tokenization.py`: Tokenization preview and analysis
- `pipeline.py`: Integrated cleaning pipelines