# Configuration Guide

QuData uses YAML configuration files to control all aspects of data processing. This guide covers all configuration options and provides templates for common use cases.

## Configuration Structure

QuData configurations are organized into logical sections:

```yaml
# Main pipeline settings
pipeline:
  name: "my_pipeline"
  input_directory: "data/raw"
  output_directory: "data/processed"

# Data ingestion settings
ingest:
  file_types: ["pdf", "docx", "txt"]
  max_file_size: "100MB"

# Text cleaning settings
clean:
  normalize_text: true
  remove_duplicates: true

# Content annotation settings
annotate:
  classify_domains: true
  extract_entities: true

# Quality control settings
quality:
  min_score: 0.6
  auto_filter: true

# Export settings
export:
  formats: ["jsonl", "parquet"]
  split_data: true
```

## Project Initialization

Initialize a new QuData project with the CLI to create a ready-to-use folder layout and starter config:

```bash
qudata init               # in current directory
qudata init --path myproj # in a specific directory
qudata init --force       # overwrite existing scaffold files
```

This scaffolds:

- `data/raw/` for input files
- `data/processed/` for pipeline outputs
- `exports/jsonl/`, `exports/chatml/`, `exports/llmbuilder/`, `exports/plain/` for exports by format
- `configs/` with a starter `pipeline.yaml`
- `QUICKSTART.md` with basic usage instructions

You can customize `configs/pipeline.yaml` as described below.

## Pipeline Configuration

### Basic Pipeline Settings

```yaml
pipeline:
  # Pipeline identification
  name: "my_data_pipeline"
  version: "1.0.0"
  description: "Processing academic papers for LLM training"
  
  # Input/Output directories
  input_directory: "data/raw"
  output_directory: "data/processed"
  export_directory: "data/exports"
  temp_directory: "data/temp"
  
  # Processing options
  parallel_processing: true
  max_workers: 4
  batch_size: 100
  
  # Memory management
  max_memory_usage: "4GB"
  streaming_mode: false
  
  # Error handling
  continue_on_error: true
  max_errors: 10
  error_log_file: "logs/pipeline_errors.log"
```

### Advanced Pipeline Settings

```yaml
pipeline:
  # Checkpointing for recovery
  enable_checkpoints: true
  checkpoint_interval: 100  # documents
  checkpoint_directory: "checkpoints/"
  
  # Progress monitoring
  progress_reporting: true
  progress_interval: 50  # documents
  
  # Resource limits
  timeout_per_document: 300  # seconds
  max_file_size: "500MB"
  
  # Caching
  enable_caching: true
  cache_directory: ".cache"
  cache_ttl: 3600  # seconds
```

## Ingestion Configuration

### File Type Detection

```yaml
ingest:
  # Supported file types
  file_types:
    - "pdf"
    - "docx"
    - "txt"
    - "html"
    - "csv"
    - "json"
    - "md"
  
  # File size limits
  max_file_size: "100MB"
  min_file_size: "1KB"
  
  # File detection
  use_magic_numbers: true
  fallback_to_extension: true
  
  # Error handling
  skip_corrupted: true
  skip_empty: true
  log_skipped_files: true
```

### PDF Processing

```yaml
ingest:
  pdf:
    # Text extraction
    extract_text: true
    preserve_layout: true
    
    # Table extraction
    extract_tables: true
    table_detection_method: "lattice"  # or "stream"
    
    # Image extraction
    extract_images: false
    image_dpi: 300
    
    # Password handling
    password_attempts: 3
    common_passwords: ["", "password", "123456"]
    
    # OCR fallback
    ocr_fallback: true
    ocr_confidence_threshold: 0.6
```

### Document Processing (DOCX/ODT)

```yaml
ingest:
  document:
    # Content extraction
    extract_text: true
    extract_tables: true
    extract_images: false
    
    # Formatting
    preserve_formatting: true
    extract_styles: false
    
    # Metadata
    extract_properties: true
    extract_comments: false
    extract_revisions: false
```

### Web Content Processing

```yaml
ingest:
  web:
    # Content extraction
    extract_main_content: true
    remove_navigation: true
    remove_ads: true
    
    # Link handling
    extract_links: true
    follow_links: false
    max_link_depth: 2
    
    # Rate limiting
    requests_per_minute: 60
    delay_between_requests: 1
    
    # User agent
    user_agent: "QuData/1.0 (+https://github.com/qubasehq/qudata)"
    
    # Timeout settings
    request_timeout: 30
    max_redirects: 5
```

### Structured Data Processing

```yaml
ingest:
  structured:
    # CSV settings
    csv:
      delimiter: ","
      quote_char: '"'
      encoding: "utf-8"
      skip_blank_lines: true
      
    # JSON settings
    json:
      extract_nested: true
      max_depth: 10
      flatten_arrays: false
      
    # Large file handling
    streaming_threshold: "50MB"
    chunk_size: 10000
```

## Cleaning Configuration

### Text Normalization

```yaml
clean:
  normalization:
    # Unicode normalization
    unicode_form: "NFKC"  # NFC, NFD, NFKC, NFKD
    
    # Encoding
    fix_encoding: true
    target_encoding: "utf-8"
    encoding_detection: true
    
    # Whitespace
    normalize_whitespace: true
    remove_extra_spaces: true
    normalize_line_endings: true
    
    # Character cleanup
    remove_control_chars: true
    remove_zero_width_chars: true
```

### OCR Error Correction

```yaml
clean:
  ocr_correction:
    enabled: true
    confidence_threshold: 0.8
    
    # Common corrections
    fix_number_letter_substitution: true
    fix_character_recognition: true
    fix_word_boundaries: true
    
    # Custom corrections
    custom_corrections:
      "c0mputer": "computer"
      "rec0gniti0n": "recognition"
      "artific1al": "artificial"
```

### Deduplication

```yaml
clean:
  deduplication:
    enabled: true
    
    # Similarity settings
    similarity_threshold: 0.85
    algorithm: "jaccard"  # jaccard, cosine, levenshtein, minhash
    
    # Processing options
    chunk_size: 1000
    parallel_processing: true
    
    # MinHash settings (for large datasets)
    minhash:
      num_perm: 128
      threshold: 0.8
```

### Boilerplate Removal

```yaml
clean:
  boilerplate:
    enabled: true
    
    # Standard patterns
    remove_headers: true
    remove_footers: true
    remove_navigation: true
    remove_sidebars: true
    
    # Custom patterns
    custom_patterns:
      - "Copyright \\d{4}.*"
      - "All rights reserved.*"
      - "Terms of Service.*"
      - "Privacy Policy.*"
    
    # Content thresholds
    min_content_ratio: 0.3  # Minimum content vs boilerplate ratio
```

### HTML Cleaning

```yaml
clean:
  html:
    # Tag removal
    remove_tags: true
    preserve_structure: false
    
    # Content cleanup
    remove_emojis: true
    remove_special_chars: false
    
    # Link handling
    preserve_links: false
    extract_link_text: true
    
    # Table handling
    preserve_tables: true
    table_to_text: false
```

### Language Detection and Filtering

```yaml
clean:
  language:
    # Detection
    detect_language: true
    confidence_threshold: 0.8
    
    # Filtering
    target_languages: ["en", "es", "fr"]  # Empty for all languages
    remove_mixed_language: false
    
    # Handling
    fallback_language: "en"
    handle_code_switching: true
```

## Annotation Configuration

### Taxonomy Classification

```yaml
annotate:
  taxonomy:
    enabled: true
    taxonomy_file: "configs/taxonomy.yaml"
    
    # Classification method
    method: "hybrid"  # rule_based, ml_based, hybrid
    confidence_threshold: 0.7
    max_categories: 3
    
    # Fallback
    fallback_category: "general"
    
    # Custom rules
    custom_rules:
      technology:
        keywords: ["AI", "machine learning", "software"]
        patterns: ["\\b(artificial intelligence|AI)\\b"]
      business:
        keywords: ["finance", "marketing", "strategy"]
        patterns: ["\\b(business|corporate)\\b"]
```

### Metadata Extraction

```yaml
annotate:
  metadata:
    # Author extraction
    extract_authors: true
    author_patterns:
      - "By\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)"
      - "Author:\\s*([^\\n]+)"
    
    # Date extraction
    extract_dates: true
    date_formats:
      - "%Y-%m-%d"
      - "%B %d, %Y"
      - "%d/%m/%Y"
    
    # Source extraction
    extract_sources: true
    extract_urls: true
    validate_urls: true
    
    # Custom metadata
    custom_extractors:
      doi: "DOI:\\s*([^\\s]+)"
      isbn: "ISBN[:\\s]*([0-9-X]+)"
```

### Named Entity Recognition

```yaml
annotate:
  ner:
    enabled: true
    
    # Model settings
    model: "en_core_web_sm"  # spaCy model
    confidence_threshold: 0.8
    
    # Entity types
    entity_types:
      - "PERSON"
      - "ORG"
      - "GPE"
      - "DATE"
      - "MONEY"
    
    # Custom entities
    custom_entities:
      PRODUCT: ["iPhone", "Android", "Windows"]
      TECHNOLOGY: ["AI", "ML", "blockchain"]
    
    # Entity linking
    entity_linking:
      enabled: false
      knowledge_base: "wikidata"
```

## Quality Control Configuration

### Quality Scoring

```yaml
quality:
  # Overall settings
  enabled: true
  min_score: 0.6
  auto_filter: true
  
  # Scoring dimensions
  dimensions:
    content: 0.4      # Content quality weight
    language: 0.3     # Language quality weight
    structure: 0.2    # Structure quality weight
    metadata: 0.1     # Metadata completeness weight
  
  # Content quality
  content_quality:
    min_length: 50
    max_length: 10000
    check_informativeness: true
    check_coherence: true
  
  # Language quality
  language_quality:
    check_grammar: true
    check_spelling: true
    check_fluency: true
    min_language_confidence: 0.8
  
  # Structure quality
  structure_quality:
    check_formatting: true
    check_organization: true
    penalize_lists: false
```

### Quality Thresholds

```yaml
quality:
  thresholds:
    # Score ranges
    high_quality: 0.8
    medium_quality: 0.6
    low_quality: 0.4
    
    # Length thresholds
    min_words: 20
    max_words: 5000
    optimal_length: 500
    
    # Language thresholds
    min_language_confidence: 0.7
    max_mixed_language_ratio: 0.1
    
    # Content thresholds
    min_unique_words: 10
    max_repetition_ratio: 0.3
```

## Export Configuration

### Format Settings

```yaml
export:
  # Enabled formats
  formats:
    - "jsonl"
    - "parquet"
    - "csv"
  
  # General settings
  include_metadata: true
  include_quality_scores: true
  
  # JSONL format
  jsonl:
    pretty_print: false
    ensure_ascii: false
    
  # Parquet format
  parquet:
    compression: "snappy"  # snappy, gzip, lz4
    row_group_size: 50000
    
  # CSV format
  csv:
    delimiter: ","
    quote_char: '"'
    include_headers: true
    escape_newlines: true
```

### Data Splitting

```yaml
export:
  splitting:
    enabled: true
    
    # Split ratios
    train_ratio: 0.8
    validation_ratio: 0.1
    test_ratio: 0.1
    
    # Stratification
    stratify_by: "domain"  # Ensure balanced splits
    
    # Randomization
    random_seed: 42
    shuffle: true
    
    # Minimum sizes
    min_split_size: 10
```

### LLMBuilder Integration

```yaml
export:
  llmbuilder:
    enabled: true
    
    # Paths
    llmbuilder_path: "/path/to/llmbuilder"
    data_directory: "data/clean"
    
    # Auto-trigger training
    auto_trigger: false
    
    # Model configuration
    model_config:
      model_type: "llama"
      size: "7b"
      learning_rate: 2e-5
      batch_size: 4
      num_epochs: 3
```

## Configuration Templates

### Academic Papers Template

```yaml
# configs/templates/academic_papers.yaml
pipeline:
  name: "academic_papers"
  description: "Processing academic papers for research LLM"

ingest:
  file_types: ["pdf"]
  pdf:
    extract_text: true
    extract_tables: true
    preserve_layout: true

clean:
  normalization:
    unicode_form: "NFKC"
    normalize_whitespace: true
  deduplication:
    enabled: true
    similarity_threshold: 0.9  # Higher threshold for academic content
  boilerplate:
    custom_patterns:
      - "References\\s*$"
      - "Bibliography\\s*$"
      - "Acknowledgments?\\s*$"

annotate:
  taxonomy:
    enabled: true
    method: "hybrid"
  metadata:
    extract_authors: true
    extract_dates: true
    custom_extractors:
      doi: "DOI:\\s*([^\\s]+)"
      abstract: "Abstract[:\\s]*([^\\n]+(?:\\n[^\\n]+)*?)(?=\\n\\s*\\n|Keywords|Introduction)"

quality:
  min_score: 0.7  # Higher quality threshold
  dimensions:
    content: 0.5
    language: 0.3
    structure: 0.2
  content_quality:
    min_length: 1000  # Longer minimum for papers
    check_coherence: true

export:
  formats: ["jsonl", "parquet"]
  splitting:
    enabled: true
    stratify_by: "domain"
```

### Web Content Template

```yaml
# configs/templates/web_content.yaml
pipeline:
  name: "web_content"
  description: "Processing web articles and blog posts"

ingest:
  file_types: ["html"]
  web:
    extract_main_content: true
    remove_navigation: true
    remove_ads: true
    requests_per_minute: 30

clean:
  html:
    remove_tags: true
    remove_emojis: true
    preserve_links: false
  boilerplate:
    remove_headers: true
    remove_footers: true
    custom_patterns:
      - "Share this article"
      - "Subscribe to our newsletter"
      - "Related articles"
  deduplication:
    enabled: true
    similarity_threshold: 0.8

annotate:
  taxonomy:
    enabled: true
    confidence_threshold: 0.6
  metadata:
    extract_authors: true
    extract_dates: true
    extract_sources: true

quality:
  min_score: 0.5  # Lower threshold for web content
  content_quality:
    min_length: 200
    max_length: 5000
    check_readability: true

export:
  formats: ["jsonl"]
  include_metadata: true
```

### Code Documentation Template

```yaml
# configs/templates/code_documentation.yaml
pipeline:
  name: "code_documentation"
  description: "Processing code documentation and tutorials"

ingest:
  file_types: ["md", "rst", "txt"]
  preserve_code_blocks: true

clean:
  normalization:
    normalize_whitespace: false  # Preserve code formatting
  html:
    preserve_structure: true
  boilerplate:
    enabled: false  # Don't remove standard doc patterns

annotate:
  taxonomy:
    custom_rules:
      programming:
        keywords: ["function", "class", "method", "API"]
        patterns: ["```\\w+", "`[^`]+`"]
  ner:
    custom_entities:
      CODE_LANGUAGE: ["Python", "JavaScript", "Java", "C++"]
      FRAMEWORK: ["React", "Django", "Flask", "Spring"]

quality:
  min_score: 0.4  # Lower threshold for code docs
  content_quality:
    min_length: 100
    check_code_completeness: true

export:
  formats: ["jsonl"]
  preserve_code_structure: true
```

## Environment Variables

QuData supports environment variable substitution in configuration files:

```yaml
database:
  connection_string: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}/${DB_NAME}"
  
api:
  openai_api_key: "${OPENAI_API_KEY}"
  
storage:
  s3_bucket: "${S3_BUCKET_NAME}"
  aws_access_key: "${AWS_ACCESS_KEY_ID}"
```

Set environment variables:

```bash
export DB_USER="qudata_user"
export DB_PASSWORD="secure_password"
export DB_HOST="localhost"
export DB_NAME="qudata"
export OPENAI_API_KEY="sk-..."
```

## Configuration Validation

QuData validates configurations on startup:

```python
from qudata.config import ConfigManager

# Load and validate configuration
config_manager = ConfigManager()
try:
    config = config_manager.load_pipeline_config("configs/my_config.yaml")
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Use Templates
Start with provided templates and customize for your use case.

### 2. Environment-Specific Configs
Create separate configs for development, testing, and production:

```
configs/
├── dev.yaml
├── test.yaml
├── prod.yaml
└── templates/
    ├── academic_papers.yaml
    ├── web_content.yaml
    └── code_docs.yaml
```

### 3. Version Control
Keep configuration files in version control and document changes.

### 4. Validation
Always validate configurations before running large processing jobs.

### 5. Monitoring
Enable logging and monitoring to track configuration effectiveness:

```yaml
pipeline:
  logging:
    level: "INFO"
    file: "logs/pipeline.log"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

For more specific configuration details, see the individual module documentation.