# Data Packaging and Format Generation

The pack module handles the final packaging of processed data into various formats suitable for LLM training and analysis.

## Components

### ChatML Formatter (`chatml.py`)
- Converts processed data to ChatML format
- Supports conversational training data structure
- Handles system messages, user inputs, and assistant responses
- Maintains conversation context and metadata

### JSONL Formatter (`jsonl.py`)
- Generates JSONL (JSON Lines) format for general LLM training
- Efficient streaming format for large datasets
- Preserves all metadata and annotations
- Supports custom field mapping and filtering

### Plain Text Formatter (`plain.py`)
- Simple text format for debugging and analysis
- Human-readable output with optional formatting
- Configurable delimiters and structure
- Useful for quality inspection and manual review

## Usage Examples

### ChatML Format Generation
```python
from qudata.pack import ChatMLFormatter

formatter = ChatMLFormatter()
chatml_data = formatter.format_documents(documents, config={
    "system_message": "You are a helpful assistant.",
    "include_metadata": True
})
```

### JSONL Export
```python
from qudata.pack import JSONLFormatter

formatter = JSONLFormatter()
formatter.export_to_file(
    documents=processed_docs,
    output_path="training_data.jsonl",
    fields=["text", "labels", "metadata"]
)
```

### Plain Text Output
```python
from qudata.pack import PlainTextFormatter

formatter = PlainTextFormatter()
text_output = formatter.format_documents(
    documents,
    separator="\n---\n",
    include_headers=True
)
```

## Format Specifications

### ChatML Format
```json
{
  "messages": [
    {"role": "system", "content": "System message"},
    {"role": "user", "content": "User input"},
    {"role": "assistant", "content": "Assistant response"}
  ],
  "metadata": {
    "source": "document.pdf",
    "quality_score": 0.85,
    "topics": ["technology", "AI"]
  }
}
```

### JSONL Format
```json
{"text": "Document content", "labels": ["category1"], "metadata": {"source": "file.pdf"}}
{"text": "Another document", "labels": ["category2"], "metadata": {"source": "file2.pdf"}}
```

### Plain Text Format
```
=== Document 1 ===
Source: document.pdf
Quality: 0.85
Topics: technology, AI

Content goes here...

---

=== Document 2 ===
...
```

## Configuration

Packaging formats can be configured through the pipeline configuration:

```yaml
pack:
  chatml:
    system_message: "You are a helpful assistant."
    include_metadata: true
    max_tokens_per_message: 4096
  jsonl:
    fields: ["text", "labels", "metadata", "quality_score"]
    filter_low_quality: true
    min_quality_score: 0.7
  plain:
    separator: "\n---\n"
    include_headers: true
    max_line_length: 120
```

## Quality Control

- Automatic validation of output formats
- Schema compliance checking
- Token count estimation for LLM compatibility
- Quality filtering based on configurable thresholds
- Duplicate detection and removal

## Integration

The pack module integrates seamlessly with:
- Export pipeline for final data output
- Quality scoring for filtering decisions
- Metadata extraction for rich annotations
- LLMBuilder integration for training pipeline