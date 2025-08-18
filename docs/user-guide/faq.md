# Frequently Asked Questions (FAQ)

## General Questions

### What is QuData?

QuData is a comprehensive data processing pipeline designed to transform raw multi-format data into high-quality datasets optimized for LLM training. It handles everything from data ingestion and cleaning to annotation and export in various training formats.

### What file formats does QuData support?

QuData supports a wide range of formats:
- **Documents**: PDF, DOCX, ODT, RTF, TXT, MD
- **Web**: HTML, XML
- **Structured**: CSV, JSON, JSONL, YAML
- **Images**: PNG, JPG, JPEG, TIFF (for OCR)
- **Archives**: ZIP, TAR, GZ
- **Code**: Jupyter notebooks, source code files

### How does QuData ensure data quality?

QuData uses multi-dimensional quality scoring that evaluates:
- **Content quality**: Informativeness, coherence, completeness
- **Language quality**: Grammar, spelling, fluency
- **Structure quality**: Organization, formatting
- **Metadata completeness**: Author, date, source information

### Can I customize the processing pipeline?

Yes! QuData is highly configurable through YAML configuration files. You can:
- Enable/disable processing stages
- Adjust quality thresholds
- Customize cleaning rules
- Define custom taxonomies
- Set export formats and options

## Installation and Setup

### What are the system requirements?

**Minimum Requirements:**
- Python 3.8 or higher
- 4GB RAM
- 10GB free disk space

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- 50GB+ free disk space
- Multi-core CPU for parallel processing

### How do I install QuData?

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[ml,web,dev]"

# For development
pip install -e ".[dev]"
```

### Do I need special dependencies for OCR?

Yes, for OCR functionality you need:

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### How do I set up a database?

QuData supports multiple databases:

**PostgreSQL (Recommended):**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb qudata
sudo -u postgres createuser qudata_user
```

**SQLite (Simple setup):**
No additional setup required - QuData will create the database file automatically.

## Processing Questions

### How long does processing take?

Processing time depends on:
- **File size and count**: Larger datasets take longer
- **File types**: PDFs with images take longer than plain text
- **Quality settings**: Higher quality thresholds require more processing
- **Hardware**: More CPU cores and RAM speed up processing

**Typical rates:**
- Plain text: 100-500 documents/minute
- PDFs: 10-50 documents/minute
- Web scraping: 30-100 pages/minute

### Can I process files in parallel?

Yes! Enable parallel processing in your configuration:

```yaml
pipeline:
  parallel_processing: true
  max_workers: 8  # Adjust based on CPU cores
  batch_size: 100
```

### What happens if processing fails?

QuData has robust error handling:
- **Continue on error**: Processing continues with other files
- **Error logging**: Detailed logs of what went wrong
- **Retry logic**: Automatic retries for transient failures
- **Checkpointing**: Resume from where processing stopped

### How do I handle large datasets?

For large datasets:

1. **Enable streaming mode**:
```yaml
pipeline:
  streaming_mode: true
  max_memory_usage: "4GB"
```

2. **Process in chunks**:
```python
def process_large_dataset(files, chunk_size=1000):
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        pipeline.process_files(chunk)
```

3. **Use incremental processing**:
```python
from qudata.database import IncrementalProcessor

processor = IncrementalProcessor(connection)
new_docs = processor.get_new_documents(since=last_run)
```

## Configuration Questions

### How do I create a custom configuration?

1. **Start with a template**:
```bash
cp configs/templates/academic-papers.yaml configs/my-config.yaml
```

2. **Modify settings**:
```yaml
pipeline:
  name: "my_custom_pipeline"

clean:
  min_quality_score: 0.8  # Higher threshold

export:
  formats: ["jsonl", "parquet"]
```

3. **Validate configuration**:
```python
from qudata.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_pipeline_config("configs/my-config.yaml")
```

### What's the difference between the templates?

- **academic-papers.yaml**: Optimized for research papers, higher quality thresholds, citation extraction
- **web-content.yaml**: Handles web articles, aggressive boilerplate removal, lower quality thresholds
- **code-documentation.yaml**: Preserves code blocks, technical entity recognition, programming language detection

### How do I add custom cleaning rules?

```yaml
clean:
  boilerplate:
    custom_patterns:
      - "Your custom pattern here"
      - "Copyright \\d{4}.*"
      - "All rights reserved.*"
  
  html:
    remove_elements:
      - "custom-ad-class"
      - "social-media-widget"
```

### Can I use environment variables in configuration?

Yes! Use `${VARIABLE_NAME}` syntax:

```yaml
database:
  host: "${DB_HOST}"
  username: "${DB_USER}"
  password: "${DB_PASSWORD}"
```

Set environment variables:
```bash
export DB_HOST="localhost"
export DB_USER="qudata_user"
export DB_PASSWORD="secure_password"
```

## Quality and Output Questions

### Why are my quality scores low?

Common reasons for low quality scores:

1. **Short documents**: Increase minimum length or lower thresholds
2. **Poor language quality**: Enable OCR correction or language filtering
3. **Boilerplate content**: Improve boilerplate removal patterns
4. **Mixed languages**: Set target languages or improve detection

**Debug quality issues**:
```python
from qudata.score import QualityScorer

scorer = QualityScorer({'detailed_analysis': True})
result = scorer.score_document(document)
print(result.detailed_scores)
print(result.issues)
```

### How do I improve extraction quality?

**For PDFs:**
```yaml
ingest:
  pdf:
    preserve_layout: true
    extract_tables: true
    ocr_fallback: true
    ocr_confidence_threshold: 0.7
```

**For web content:**
```yaml
ingest:
  web:
    extract_main_content: true
    remove_navigation: true
    use_readability: true
```

**For documents:**
```yaml
ingest:
  document:
    preserve_formatting: true
    extract_tables: true
    extract_properties: true
```

### What export formats are available?

QuData supports multiple export formats:

- **JSONL**: Standard format for LLM training
- **ChatML**: Conversational format for chat models
- **Alpaca**: Instruction-following format
- **Parquet**: Efficient columnar format for analytics
- **CSV**: Simple tabular format

### How do I create train/validation/test splits?

```yaml
export:
  splitting:
    enabled: true
    train_ratio: 0.8
    validation_ratio: 0.1
    test_ratio: 0.1
    stratify_by: "domain"  # Ensure balanced splits
    random_seed: 42
```

## Performance Questions

### How can I speed up processing?

1. **Increase parallel workers**:
```yaml
pipeline:
  max_workers: 16  # More workers
```

2. **Use faster algorithms**:
```yaml
clean:
  deduplication:
    algorithm: "minhash"  # Faster than jaccard
```

3. **Reduce quality checks**:
```yaml
quality:
  enabled: false  # Skip quality scoring
```

4. **Enable caching**:
```yaml
pipeline:
  enable_caching: true
  cache_directory: ".cache"
```

### Why is my processing slow?

Common bottlenecks:

1. **Large files**: PDFs with many images
2. **Complex cleaning**: Extensive deduplication
3. **Quality scoring**: Detailed quality analysis
4. **Single-threaded**: Not using parallel processing
5. **Memory constraints**: Frequent garbage collection

**Profile performance**:
```python
import cProfile
cProfile.run('pipeline.process_directory("data/raw", "data/processed")')
```

### How much memory does QuData use?

Memory usage depends on:
- **Batch size**: Larger batches use more memory
- **File sizes**: Large documents require more memory
- **Processing stages**: Some stages are memory-intensive

**Monitor memory usage**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

**Reduce memory usage**:
```yaml
pipeline:
  batch_size: 50  # Smaller batches
  streaming_mode: true
  max_memory_usage: "2GB"
```

## Integration Questions

### How do I integrate with LLMBuilder?

```yaml
export:
  llmbuilder:
    enabled: true
    llmbuilder_path: "/path/to/llmbuilder"
    auto_trigger: true
    model_config:
      model_type: "llama"
      size: "7b"
```

### Can I use QuData with other training frameworks?

Yes! QuData exports standard formats:
- **Hugging Face**: Use JSONL or Parquet exports
- **OpenAI**: Use JSONL format
- **Custom frameworks**: Use CSV or JSON exports

### How do I integrate with my existing pipeline?

**Python API**:
```python
from qudata import QuDataPipeline

pipeline = QuDataPipeline(config_path="my-config.yaml")
dataset = pipeline.process_files(file_list)
export_path = pipeline.export_dataset(dataset, "jsonl")
```

**REST API**:
```bash
curl -X POST "https://api.qudata.com/v1/datasets" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"name": "My Dataset", "files": ["file1.pdf"]}'
```

**Command Line**:
```bash
qudata process --input data/raw --output data/processed --config my-config.yaml
```

## Troubleshooting Questions

### Processing fails with "Out of Memory" error

**Solutions:**
1. Reduce batch size
2. Enable streaming mode
3. Increase system memory
4. Process in smaller chunks

```yaml
pipeline:
  batch_size: 25
  streaming_mode: true
  max_memory_usage: "2GB"
```

### PDF extraction produces garbled text

**Solutions:**
1. Enable OCR fallback
2. Improve OCR preprocessing
3. Check PDF integrity

```yaml
ingest:
  pdf:
    ocr_fallback: true
    ocr_confidence_threshold: 0.6
    preprocessing:
      enhance_contrast: true
      denoise: true
```

### Web scraping gets blocked

**Solutions:**
1. Reduce request rate
2. Use proper user agent
3. Add delays between requests

```yaml
ingest:
  web:
    requests_per_minute: 30
    delay_between_requests: 2
    user_agent: "Mozilla/5.0 (compatible; QuData/1.0)"
```

### Database connection fails

**Check connection settings**:
```python
from qudata.database import DatabaseConnector

config = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'qudata',
    'username': 'user',
    'password': 'password'
}

try:
    connector = DatabaseConnector(config)
    connection = connector.connect()
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Configuration validation fails

**Common issues:**
1. Invalid YAML syntax
2. Missing required fields
3. Invalid parameter values

**Validate configuration**:
```python
from qudata.config import ConfigManager
from pydantic import ValidationError

try:
    config_manager = ConfigManager()
    config = config_manager.load_pipeline_config("my-config.yaml")
except ValidationError as e:
    for error in e.errors():
        print(f"{error['loc']}: {error['msg']}")
```

## Advanced Questions

### Can I create custom extractors?

Yes! Extend the base extractor class:

```python
from qudata.models import BaseExtractor, ExtractedContent

class CustomExtractor(BaseExtractor):
    def extract(self, file_path: str) -> ExtractedContent:
        # Your custom extraction logic
        content = self.extract_custom_format(file_path)
        
        return ExtractedContent(
            content=content,
            metadata=self.extract_metadata(file_path)
        )
    
    def supports_format(self, file_type: str) -> bool:
        return file_type == "custom_format"
```

### How do I add custom quality metrics?

```python
from qudata.score import QualityScorer

class CustomQualityScorer(QualityScorer):
    def calculate_custom_score(self, document):
        # Your custom quality logic
        return score
    
    def score_document(self, document):
        base_result = super().score_document(document)
        custom_score = self.calculate_custom_score(document)
        
        # Combine scores
        base_result.custom_score = custom_score
        return base_result
```

### Can I use custom ML models?

Yes! For classification and NER:

```python
from qudata.annotate import TaxonomyClassifier

class CustomClassifier(TaxonomyClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.load_custom_model()
    
    def classify_document(self, document):
        # Use your custom model
        predictions = self.model.predict(document.content)
        return self.format_results(predictions)
```

### How do I deploy QuData at scale?

**Docker deployment**:
```bash
docker-compose up -d --scale qudata-worker=5
```

**Kubernetes deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qudata-workers
spec:
  replicas: 10
  selector:
    matchLabels:
      app: qudata-worker
  template:
    spec:
      containers:
      - name: qudata-worker
        image: qudata:latest
        command: ["qudata", "worker"]
```

**Distributed processing**:
```python
from qudata.orchestrate import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator({
    'backend': 'celery',
    'broker_url': 'redis://localhost:6379/0',
    'workers': 10
})

# Distribute processing across workers
orchestrator.process_dataset_distributed(large_dataset)
```

## Getting Help

### Where can I find more documentation?

- **Module documentation**: Check README files in `src/forge/*/`
- **API documentation**: See `docs/api/`
- **Examples**: Look at `examples/` directory
- **Configuration**: Review `configs/templates/`

### How do I report bugs or request features?

1. **Check existing issues**: Search the issue tracker
2. **Provide details**: Include configuration, error messages, sample data
3. **Minimal reproduction**: Create a simple example that reproduces the issue

### How do I contribute to QuData?

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Follow code style** guidelines
5. **Submit a pull request**

### Where can I get community support?

- **GitHub Discussions**: For questions and community help
- **Issue Tracker**: For bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference