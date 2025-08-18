# Troubleshooting Guide

This guide covers common issues you might encounter when using QuData and provides solutions and workarounds.

## Installation Issues

### Python Version Compatibility

**Problem**: Installation fails with Python version errors.

**Solution**:
```bash
# Check Python version
python --version

# QuData requires Python 3.8+
# If using older version, upgrade Python or use pyenv
pyenv install 3.9.16
pyenv local 3.9.16
```

### Dependency Conflicts

**Problem**: Package dependency conflicts during installation.

**Solution**:
```bash
# Create clean virtual environment
python -m venv qudata_env
source qudata_env/bin/activate  # On Windows: qudata_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install QuData
pip install -e .
```

### Missing System Dependencies

**Problem**: Installation fails due to missing system libraries.

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install python3-dev libpq-dev build-essential
sudo apt-get install tesseract-ocr libtesseract-dev  # For OCR
sudo apt-get install libxml2-dev libxslt1-dev  # For web scraping
```

**macOS**:
```bash
brew install postgresql tesseract
xcode-select --install  # For build tools
```

**Windows**:
```bash
# Install Visual Studio Build Tools
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Processing Issues

### Out of Memory Errors

**Problem**: Processing fails with memory errors on large datasets.

**Solutions**:

1. **Enable Streaming Mode**:
```yaml
# configs/memory_efficient.yaml
pipeline:
  streaming_mode: true
  batch_size: 50  # Reduce batch size
  max_memory_usage: "2GB"

clean:
  deduplication:
    chunk_size: 500  # Smaller chunks
```

2. **Use Memory-Efficient Processing**:
```python
from qudata import QuDataPipeline

pipeline = QuDataPipeline({
    'streaming_mode': True,
    'batch_size': 100,
    'parallel_processing': False  # Disable if memory constrained
})
```

3. **Process in Chunks**:
```python
import os
from pathlib import Path

def process_large_directory(input_dir, output_dir, chunk_size=100):
    files = list(Path(input_dir).rglob('*'))
    
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        chunk_output = f"{output_dir}/chunk_{i//chunk_size}"
        
        pipeline = QuDataPipeline()
        pipeline.process_files([str(f) for f in chunk])
```

### Slow Processing Performance

**Problem**: Processing is taking too long.

**Solutions**:

1. **Enable Parallel Processing**:
```yaml
pipeline:
  parallel_processing: true
  max_workers: 8  # Adjust based on CPU cores
```

2. **Optimize Configuration**:
```yaml
clean:
  deduplication:
    algorithm: "minhash"  # Faster for large datasets
    parallel_processing: true
  
quality:
  enabled: false  # Disable if not needed for speed
```

3. **Use Faster Algorithms**:
```python
from qudata.clean import DeduplicationEngine

# Use MinHash for large datasets
deduper = DeduplicationEngine({
    'algorithm': 'minhash',
    'num_perm': 64,  # Reduce for speed
    'threshold': 0.8
})
```

4. **Profile Performance**:
```python
import cProfile
import pstats

def profile_processing():
    pipeline = QuDataPipeline()
    pipeline.process_directory("data/raw", "data/processed")

# Run profiler
cProfile.run('profile_processing()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

### File Processing Errors

**Problem**: Specific files fail to process.

**Solutions**:

1. **Check File Integrity**:
```python
from qudata.ingest import FileTypeDetector

detector = FileTypeDetector()

def check_file_integrity(file_path):
    try:
        file_type = detector.detect_file_type(file_path)
        print(f"File type: {file_type}")
        
        # Try to read file
        with open(file_path, 'rb') as f:
            f.read(1024)  # Read first 1KB
        print("File appears readable")
        
    except Exception as e:
        print(f"File issue: {e}")

check_file_integrity("problematic_file.pdf")
```

2. **Enable Error Logging**:
```yaml
pipeline:
  continue_on_error: true
  error_log_file: "logs/processing_errors.log"
  log_level: "DEBUG"
```

3. **Skip Corrupted Files**:
```yaml
ingest:
  skip_corrupted: true
  skip_empty: true
  max_file_size: "100MB"  # Skip very large files
```

## Format-Specific Issues

### PDF Processing Problems

**Problem**: PDF extraction fails or produces poor results.

**Solutions**:

1. **Check PDF Type**:
```python
from qudata.ingest import PDFExtractor
import pdfplumber

def diagnose_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Pages: {len(pdf.pages)}")
            print(f"Encrypted: {pdf.is_encrypted}")
            
            # Check first page
            first_page = pdf.pages[0]
            print(f"Text objects: {len(first_page.chars)}")
            print(f"Images: {len(first_page.images)}")
            
    except Exception as e:
        print(f"PDF diagnosis failed: {e}")

diagnose_pdf("problematic.pdf")
```

2. **Handle Password-Protected PDFs**:
```yaml
ingest:
  pdf:
    password_attempts: 5
    common_passwords: ["", "password", "123456", "admin"]
    skip_encrypted: true  # Skip if can't decrypt
```

3. **Enable OCR Fallback**:
```yaml
ingest:
  pdf:
    ocr_fallback: true
    ocr_confidence_threshold: 0.5
    ocr_language: "eng"
```

### OCR Issues

**Problem**: OCR produces poor quality text.

**Solutions**:

1. **Improve Image Preprocessing**:
```python
from qudata.ingest import OCRProcessor, ImagePreprocessor

preprocessor = ImagePreprocessor({
    'enhance_contrast': True,
    'denoise': True,
    'deskew': True,
    'resize_factor': 2.0  # Upscale for better OCR
})

ocr = OCRProcessor({
    'preprocessor': preprocessor,
    'confidence_threshold': 0.8,
    'language': 'eng+fra'  # Multiple languages
})
```

2. **Check Tesseract Installation**:
```bash
# Test Tesseract
tesseract --version
tesseract --list-langs

# Install additional language packs
sudo apt-get install tesseract-ocr-fra tesseract-ocr-deu
```

3. **Adjust OCR Settings**:
```yaml
ingest:
  ocr:
    language: "eng"
    psm: 6  # Page segmentation mode
    oem: 3  # OCR Engine Mode
    confidence_threshold: 0.6
    preprocessing:
      enhance_contrast: true
      denoise: true
      deskew: true
```

### Web Scraping Issues

**Problem**: Web scraping fails or gets blocked.

**Solutions**:

1. **Respect Rate Limits**:
```yaml
ingest:
  web:
    requests_per_minute: 30  # Slower rate
    delay_between_requests: 2
    user_agent: "Mozilla/5.0 (compatible; QuData/1.0)"
```

2. **Handle Different Content Types**:
```python
from qudata.ingest import WebScraper

scraper = WebScraper({
    'timeout': 30,
    'max_redirects': 5,
    'verify_ssl': False,  # If SSL issues
    'headers': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate'
    }
})
```

3. **Debug Scraping Issues**:
```python
import requests

def debug_url(url):
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Content length: {len(response.content)}")
        
    except Exception as e:
        print(f"Request failed: {e}")

debug_url("https://example.com")
```

## Quality and Output Issues

### Poor Quality Scores

**Problem**: Documents receive unexpectedly low quality scores.

**Solutions**:

1. **Analyze Quality Components**:
```python
from qudata.score import QualityScorer

scorer = QualityScorer()
result = scorer.score_document(document)

print(f"Overall: {result.overall_score}")
print(f"Content: {result.content_score}")
print(f"Language: {result.language_score}")
print(f"Structure: {result.structure_score}")
print(f"Issues: {result.issues}")
```

2. **Adjust Quality Thresholds**:
```yaml
quality:
  min_score: 0.4  # Lower threshold
  dimensions:
    content: 0.5
    language: 0.3
    structure: 0.2
  
  content_quality:
    min_length: 20  # Shorter minimum
    check_coherence: false  # Disable strict checks
```

3. **Debug Quality Issues**:
```python
def debug_quality(document):
    from qudata.score import QualityScorer
    
    scorer = QualityScorer({'detailed_analysis': True})
    result = scorer.score_document(document)
    
    print(f"Document length: {len(document.content)}")
    print(f"Word count: {len(document.content.split())}")
    print(f"Language: {document.metadata.language}")
    print(f"Quality breakdown: {result.detailed_scores}")

debug_quality(problematic_document)
```

### Export Format Issues

**Problem**: Exported data has formatting problems.

**Solutions**:

1. **Validate Export Format**:
```python
from qudata.export import ExportValidator

validator = ExportValidator()
result = validator.validate_export("output.jsonl", "jsonl")

if not result.is_valid:
    for error in result.errors:
        print(f"Validation error: {error}")
```

2. **Check Encoding Issues**:
```python
import json

def check_jsonl_encoding(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {i+1}: {e}")
                    print(f"Content: {line[:100]}...")
                    
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")

check_jsonl_encoding("output.jsonl")
```

3. **Fix Export Configuration**:
```yaml
export:
  jsonl:
    ensure_ascii: false
    indent: null  # Compact format
    separators: [',', ':']
  
  encoding: "utf-8"
  validate_output: true
```

## Database Issues

### Connection Problems

**Problem**: Database connection fails.

**Solutions**:

1. **Test Connection**:
```python
from qudata.database import DatabaseConnector

def test_connection(config):
    try:
        connector = DatabaseConnector(config)
        connection = connector.connect()
        print("Connection successful!")
        
        # Test query
        result = connection.execute("SELECT 1")
        print(f"Test query result: {result}")
        
    except Exception as e:
        print(f"Connection failed: {e}")

config = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'qudata',
    'username': 'user',
    'password': 'password'
}

test_connection(config)
```

2. **Check Database Status**:
```bash
# PostgreSQL
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# MongoDB
sudo systemctl status mongod
mongo --eval "db.version()"
```

3. **Fix Connection Pool Issues**:
```yaml
database:
  pool_size: 5  # Reduce pool size
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
```

### Query Performance Issues

**Problem**: Database queries are slow.

**Solutions**:

1. **Add Indexes**:
```python
from qudata.database import IndexManager

index_manager = IndexManager(connection)

# Create performance indexes
index_manager.create_index(
    table='documents',
    columns=['quality_score'],
    type='btree'
)

index_manager.create_index(
    table='documents',
    columns=['created_at', 'domain'],
    type='btree'
)
```

2. **Optimize Queries**:
```python
from qudata.database import QueryOptimizer

optimizer = QueryOptimizer(connection)

# Find slow queries
slow_queries = optimizer.find_slow_queries(min_duration=1000)

for query in slow_queries:
    suggestions = optimizer.suggest_optimizations(query)
    print(f"Query: {query.sql}")
    print(f"Suggestions: {suggestions}")
```

3. **Use Query Caching**:
```python
from qudata.database import CacheManager

cache = CacheManager({
    'backend': 'redis',
    'connection_string': 'redis://localhost:6379/0',
    'default_ttl': 3600
})

# Cache expensive queries
def get_cached_documents(domain):
    cache_key = f"documents:domain:{domain}"
    
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    # Execute query and cache result
    result = execute_query(f"SELECT * FROM documents WHERE domain = '{domain}'")
    cache.set(cache_key, result, ttl=1800)
    return result
```

## Configuration Issues

### Invalid Configuration

**Problem**: Configuration validation fails.

**Solutions**:

1. **Validate Configuration**:
```python
from qudata.config import ConfigManager
from pydantic import ValidationError

try:
    config_manager = ConfigManager()
    config = config_manager.load_pipeline_config("configs/my_config.yaml")
    print("Configuration is valid!")
    
except ValidationError as e:
    print("Configuration errors:")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
        
except FileNotFoundError:
    print("Configuration file not found")
```

2. **Check YAML Syntax**:
```python
import yaml

def validate_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        print("YAML syntax is valid")
        
    except yaml.YAMLError as e:
        print(f"YAML syntax error: {e}")

validate_yaml("configs/my_config.yaml")
```

3. **Use Configuration Templates**:
```bash
# Copy working template
cp configs/templates/academic_papers.yaml configs/my_config.yaml

# Modify as needed
```

### Environment Variable Issues

**Problem**: Environment variables not being substituted.

**Solutions**:

1. **Check Environment Variables**:
```bash
# List all environment variables
env | grep -i qudata

# Check specific variable
echo $DB_PASSWORD
```

2. **Debug Variable Substitution**:
```python
import os
from qudata.config import ConfigManager

# Check if variables are set
required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST']
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"{var}: {'*' * len(value)}")  # Hide sensitive values
    else:
        print(f"{var}: NOT SET")

# Load config with debug
config_manager = ConfigManager(debug=True)
config = config_manager.load_pipeline_config("configs/my_config.yaml")
```

## Logging and Debugging

### Enable Debug Logging

```yaml
pipeline:
  logging:
    level: "DEBUG"
    file: "logs/debug.log"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
  debug_mode: true
  verbose: true
```

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable QuData debug logging
logging.getLogger('forge').setLevel(logging.DEBUG)
```

### Performance Profiling

```python
import cProfile
import pstats
from qudata import QuDataPipeline

def profile_pipeline():
    pipeline = QuDataPipeline()
    pipeline.process_directory("data/raw", "data/processed")

# Profile execution
cProfile.run('profile_pipeline()', 'profile.stats')

# Analyze results
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Getting Help

### Check System Requirements

```python
import sys
import platform

def system_info():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    
    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.1f} GB")
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")

system_info()
```

### Collect Debug Information

```python
from qudata import __version__
from qudata.config import ConfigManager

def collect_debug_info():
    print(f"QuData version: {__version__}")
    
    # Check configuration
    try:
        config_manager = ConfigManager()
        print("Configuration manager: OK")
    except Exception as e:
        print(f"Configuration manager: ERROR - {e}")
    
    # Check dependencies
    try:
        import pdfplumber
        print(f"pdfplumber: {pdfplumber.__version__}")
    except ImportError:
        print("pdfplumber: NOT INSTALLED")
    
    try:
        import spacy
        print(f"spaCy: {spacy.__version__}")
    except ImportError:
        print("spaCy: NOT INSTALLED")

collect_debug_info()
```

### Report Issues

When reporting issues, include:

1. **System Information**: OS, Python version, QuData version
2. **Configuration**: Sanitized configuration file
3. **Error Messages**: Complete error traceback
4. **Sample Data**: Minimal example that reproduces the issue
5. **Expected vs Actual**: What you expected vs what happened

### Community Resources

- **Documentation**: Check module-specific README files
- **Examples**: Review example scripts in `examples/` directory
- **Tests**: Look at test files for usage patterns
- **Configuration**: Use templates in `configs/templates/`

For persistent issues, consider:
- Reducing dataset size for testing
- Using simpler configurations
- Processing files individually to isolate problems
- Checking system resources (memory, disk space)
- Updating dependencies to latest versions