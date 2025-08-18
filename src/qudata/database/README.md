# Database Integration and Warehousing Module

The database module provides comprehensive data storage, retrieval, and warehousing capabilities with support for multiple database backends and advanced features like versioning, partitioning, and incremental updates.

## Overview

This module handles:
- **Multi-Database Support**: PostgreSQL, MySQL, MongoDB, SQLite, DuckDB
- **Data Warehousing**: Structured storage with schema management
- **Version Control**: Dataset versioning and change tracking
- **Incremental Processing**: Efficient updates for new data
- **Query Building**: Dynamic SQL and NoSQL query generation
- **Backup and Recovery**: Automated backup and restoration
- **Performance Optimization**: Indexing, partitioning, and caching

## Core Components

### Database Connectivity

```python
from qudata.database import DatabaseConnector

# SQL Database Connection
sql_connector = DatabaseConnector({
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'qudata',
    'username': 'user',
    'password': 'password',
    'pool_size': 10
})

connection = sql_connector.connect()
```

**Supported Databases:**
- **SQL**: PostgreSQL, MySQL, SQLite, DuckDB
- **NoSQL**: MongoDB, Elasticsearch
- **Cloud**: AWS RDS, Google Cloud SQL, Azure SQL
- **In-Memory**: Redis (for caching)

### SQL Data Extraction

```python
from qudata.database import SQLExtractor

extractor = SQLExtractor(connection_config={
    'type': 'postgresql',
    'connection_string': 'postgresql://user:pass@localhost/db'
})

# Extract data with custom query
data = extractor.extract_data("""
    SELECT title, content, created_at, author
    FROM articles 
    WHERE created_at > %s
    ORDER BY created_at DESC
""", params=['2024-01-01'])

# Convert to documents
documents = extractor.to_documents(data, content_field='content')
```

### NoSQL Data Extraction

```python
from qudata.database import NoSQLExtractor

# MongoDB extraction
mongo_extractor = NoSQLExtractor({
    'type': 'mongodb',
    'connection_string': 'mongodb://localhost:27017/mydb'
})

# Extract documents with query
documents = mongo_extractor.extract_documents(
    collection='articles',
    query={'status': 'published'},
    projection={'title': 1, 'content': 1, 'metadata': 1}
)
```

### Query Builder

```python
from qudata.database import QueryBuilder

builder = QueryBuilder('postgresql')

# Build complex queries dynamically
query = builder.select(['title', 'content', 'author']) \
              .from_table('articles') \
              .where('published_date', '>', '2024-01-01') \
              .where('status', '=', 'published') \
              .order_by('published_date', 'DESC') \
              .limit(100) \
              .build()

print(query.sql)    # Generated SQL
print(query.params) # Query parameters
```

### Schema Management

```python
from qudata.database import SchemaManager

manager = SchemaManager(connection)

# Create schema for QuData
schema = {
    'documents': {
        'id': 'VARCHAR(255) PRIMARY KEY',
        'content': 'TEXT NOT NULL',
        'metadata': 'JSONB',
        'quality_score': 'FLOAT',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
    },
    'processing_results': {
        'id': 'VARCHAR(255) PRIMARY KEY',
        'document_id': 'VARCHAR(255) REFERENCES documents(id)',
        'stage': 'VARCHAR(100)',
        'result': 'JSONB',
        'processing_time': 'FLOAT',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
    }
}

manager.create_schema(schema)
```

### Data Versioning

```python
from qudata.database import VersionManager

version_manager = VersionManager(connection)

# Create new dataset version
version = version_manager.create_version(
    dataset_id='my_dataset',
    version_name='v1.2.0',
    description='Added new documents and improved quality',
    metadata={'documents_added': 150, 'quality_threshold': 0.8}
)

# Track changes
changes = version_manager.track_changes(
    from_version='v1.1.0',
    to_version='v1.2.0'
)

print(f"Documents added: {changes.documents_added}")
print(f"Documents modified: {changes.documents_modified}")
print(f"Documents removed: {changes.documents_removed}")
```

### Incremental Processing

```python
from qudata.database import IncrementalProcessor

processor = IncrementalProcessor(connection)

# Process only new or changed documents
last_processed = processor.get_last_processed_timestamp('my_dataset')
new_documents = processor.get_new_documents(since=last_processed)

print(f"Processing {len(new_documents)} new documents")

# Update processing timestamp
processor.update_last_processed_timestamp('my_dataset')
```

### Backup and Recovery

```python
from qudata.database import BackupManager

backup_manager = BackupManager(connection)

# Create backup
backup_result = backup_manager.create_backup(
    backup_name='daily_backup_20240115',
    include_tables=['documents', 'processing_results'],
    compression=True
)

# Restore from backup
restore_result = backup_manager.restore_backup(
    backup_name='daily_backup_20240115',
    target_database='qudata_restored'
)
```

## Configuration

### Database Configuration

```yaml
# configs/database.yaml
database:
  primary:
    type: "postgresql"
    host: "localhost"
    port: 5432
    database: "qudata"
    username: "qudata_user"
    password: "${DB_PASSWORD}"
    pool_size: 20
    timeout: 30
  
  cache:
    type: "redis"
    host: "localhost"
    port: 6379
    database: 0
    ttl: 3600
  
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention_days: 30
    compression: true
    storage_path: "/backups/qudata"
  
  versioning:
    enabled: true
    auto_version: true
    version_format: "v{major}.{minor}.{patch}"
```

### Advanced Configuration

```python
from qudata.database import DatabaseConnector

# Connection pooling and optimization
config = {
    'type': 'postgresql',
    'connection_string': 'postgresql://user:pass@localhost/db',
    'pool_config': {
        'pool_size': 20,
        'max_overflow': 30,
        'pool_timeout': 30,
        'pool_recycle': 3600
    },
    'optimization': {
        'enable_query_cache': True,
        'cache_size': 1000,
        'enable_connection_pooling': True,
        'auto_commit': False
    }
}

connector = DatabaseConnector(config)
```

## Data Warehousing

### Warehouse Schema Design

```python
from qudata.database import DataWarehouse

warehouse = DataWarehouse(connection)

# Define warehouse schema
warehouse_schema = {
    'fact_documents': {
        'document_id': 'VARCHAR(255) PRIMARY KEY',
        'content_hash': 'VARCHAR(64)',
        'word_count': 'INTEGER',
        'quality_score': 'FLOAT',
        'language_id': 'INTEGER',
        'domain_id': 'INTEGER',
        'created_date': 'DATE',
        'processing_date': 'DATE'
    },
    'dim_languages': {
        'language_id': 'SERIAL PRIMARY KEY',
        'language_code': 'VARCHAR(10)',
        'language_name': 'VARCHAR(100)'
    },
    'dim_domains': {
        'domain_id': 'SERIAL PRIMARY KEY',
        'domain_name': 'VARCHAR(100)',
        'parent_domain_id': 'INTEGER'
    }
}

warehouse.create_warehouse_schema(warehouse_schema)
```

### ETL Operations

```python
from qudata.database import ETLProcessor

etl = ETLProcessor(source_connection, target_connection)

# Extract, Transform, Load pipeline
def etl_pipeline():
    # Extract
    raw_data = etl.extract_from_source("""
        SELECT * FROM raw_documents 
        WHERE processed = false
    """)
    
    # Transform
    transformed_data = etl.transform_data(raw_data, [
        etl.clean_text,
        etl.extract_metadata,
        etl.calculate_quality_score,
        etl.normalize_language_codes
    ])
    
    # Load
    etl.load_to_warehouse(transformed_data, 'fact_documents')
    
    # Update source
    etl.mark_as_processed(raw_data)

# Run ETL pipeline
etl_pipeline()
```

### Data Partitioning

```python
from qudata.database import PartitionManager

partition_manager = PartitionManager(connection)

# Create time-based partitions
partition_manager.create_time_partitions(
    table='documents',
    partition_column='created_at',
    partition_type='monthly',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Create hash-based partitions
partition_manager.create_hash_partitions(
    table='documents',
    partition_column='document_id',
    num_partitions=16
)
```

## Performance Optimization

### Indexing Strategy

```python
from qudata.database import IndexManager

index_manager = IndexManager(connection)

# Create performance indexes
indexes = [
    {
        'table': 'documents',
        'columns': ['quality_score'],
        'type': 'btree'
    },
    {
        'table': 'documents',
        'columns': ['content'],
        'type': 'gin',  # For full-text search
        'options': 'USING gin(to_tsvector(\'english\', content))'
    },
    {
        'table': 'documents',
        'columns': ['created_at', 'domain'],
        'type': 'btree'
    }
]

for index in indexes:
    index_manager.create_index(**index)
```

### Query Optimization

```python
from qudata.database import QueryOptimizer

optimizer = QueryOptimizer(connection)

# Analyze query performance
slow_queries = optimizer.find_slow_queries(min_duration=1000)  # > 1 second

for query in slow_queries:
    print(f"Query: {query.sql}")
    print(f"Average time: {query.avg_duration}ms")
    
    # Get optimization suggestions
    suggestions = optimizer.suggest_optimizations(query)
    for suggestion in suggestions:
        print(f"Suggestion: {suggestion}")
```

### Caching Layer

```python
from qudata.database import CacheManager

cache = CacheManager({
    'backend': 'redis',
    'connection_string': 'redis://localhost:6379/0',
    'default_ttl': 3600,
    'key_prefix': 'qudata:'
})

# Cache query results
def get_documents_by_domain(domain):
    cache_key = f"documents:domain:{domain}"
    
    # Try cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Query database
    result = connection.execute("""
        SELECT * FROM documents WHERE domain = %s
    """, [domain])
    
    # Cache result
    cache.set(cache_key, result, ttl=1800)  # 30 minutes
    return result
```

## Examples

### Basic Database Operations

```python
from qudata.database import DatabaseConnector, SQLExtractor

# Connect to database
connector = DatabaseConnector({
    'type': 'postgresql',
    'connection_string': 'postgresql://user:pass@localhost/qudata'
})

connection = connector.connect()

# Extract documents
extractor = SQLExtractor(connection)
documents = extractor.extract_documents("""
    SELECT id, title, content, metadata, quality_score
    FROM documents
    WHERE quality_score > 0.7
    ORDER BY created_at DESC
    LIMIT 1000
""")

print(f"Extracted {len(documents)} high-quality documents")
```

### Data Warehousing Pipeline

```python
from qudata.database import DataWarehouse, ETLProcessor

# Initialize warehouse
warehouse = DataWarehouse(connection)

# Create ETL processor
etl = ETLProcessor(source_conn, warehouse_conn)

# Define transformation pipeline
def transform_document(doc):
    return {
        'document_id': doc['id'],
        'content_hash': hashlib.md5(doc['content'].encode()).hexdigest(),
        'word_count': len(doc['content'].split()),
        'quality_score': doc['quality_score'],
        'language_id': get_language_id(doc['language']),
        'domain_id': get_domain_id(doc['domain']),
        'created_date': doc['created_at'].date(),
        'processing_date': datetime.now().date()
    }

# Run ETL pipeline
raw_documents = etl.extract_from_source("SELECT * FROM raw_documents")
transformed_docs = [transform_document(doc) for doc in raw_documents]
etl.load_to_warehouse(transformed_docs, 'fact_documents')
```

### Incremental Data Processing

```python
from qudata.database import IncrementalProcessor

processor = IncrementalProcessor(connection)

# Get last processing checkpoint
last_checkpoint = processor.get_checkpoint('document_processing')
print(f"Last processed: {last_checkpoint}")

# Process new documents
new_docs = processor.get_new_documents(
    table='documents',
    timestamp_column='created_at',
    since=last_checkpoint
)

print(f"Processing {len(new_docs)} new documents")

# Process documents (your processing logic here)
for doc in new_docs:
    # Process document
    processed_doc = process_document(doc)
    
    # Store result
    processor.store_processed_document(processed_doc)

# Update checkpoint
processor.update_checkpoint('document_processing', datetime.now())
```

### Version Management

```python
from qudata.database import VersionManager

version_manager = VersionManager(connection)

# Create new version
version = version_manager.create_version(
    dataset_id='training_data',
    version_name='v2.1.0',
    description='Added 500 new documents, improved quality filtering',
    metadata={
        'documents_added': 500,
        'quality_threshold_updated': 0.75,
        'new_domains': ['healthcare', 'finance']
    }
)

# Compare versions
comparison = version_manager.compare_versions(
    dataset_id='training_data',
    version_a='v2.0.0',
    version_b='v2.1.0'
)

print(f"Documents added: {comparison.documents_added}")
print(f"Quality improvement: {comparison.avg_quality_change}")
print(f"New domains: {comparison.new_domains}")
```

## Testing

```bash
# Run database module tests
pytest tests/unit/test_database_connector.py
pytest tests/unit/test_sql_extractor.py
pytest tests/unit/test_query_builder.py

# Run integration tests
pytest tests/integration/test_database_warehousing_integration.py -v

# Run performance tests
pytest tests/benchmarks/test_database_performance.py
```

## Dependencies

**Core Dependencies:**
- `sqlalchemy`: SQL database ORM and connection management
- `psycopg2`: PostgreSQL adapter
- `pymongo`: MongoDB driver
- `redis`: Redis client for caching

**Optional Dependencies:**
- `mysql-connector-python`: MySQL connectivity
- `duckdb`: DuckDB embedded database
- `elasticsearch`: Elasticsearch client

## Troubleshooting

### Common Issues

**Connection Pool Exhaustion:**
```python
# Increase pool size and add overflow
connector = DatabaseConnector({
    'pool_size': 30,
    'max_overflow': 50,
    'pool_timeout': 60
})
```

**Slow Query Performance:**
```python
# Enable query optimization
from qudata.database import QueryOptimizer

optimizer = QueryOptimizer(connection)
optimizer.analyze_table('documents')
optimizer.create_recommended_indexes()
```

**Memory Issues with Large Datasets:**
```python
# Use streaming extraction
extractor = SQLExtractor(connection)
for batch in extractor.extract_streaming(query, batch_size=1000):
    process_batch(batch)
```

## API Reference

For detailed API documentation, see the individual module docstrings:
- `connector.py`: Database connection management
- `sql_extractor.py`: SQL database data extraction
- `nosql_extractor.py`: NoSQL database integration
- `query_builder.py`: Dynamic query construction
- `schema_manager.py`: Database schema management
- `versioning.py`: Dataset version control
- `incremental.py`: Incremental processing utilities
- `backup.py`: Backup and recovery operations
- `partitioning.py`: Table partitioning management