#!/usr/bin/env python3
"""
Test script for database extractors with mocked dependencies.
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sql_extractor():
    """Test SQLExtractor functionality with mocked dependencies."""
    print("Testing SQLExtractor...")
    
    # Import directly from module file bypassing __init__.py
    import importlib.util
    
    # Import connector first
    spec = importlib.util.spec_from_file_location("connector", "src/forge/database/connector.py")
    connector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector_module)
    
    # Import query builder
    spec = importlib.util.spec_from_file_location("query_builder", "src/forge/database/query_builder.py")
    query_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query_builder_module)
    
    # Import SQL extractor
    spec = importlib.util.spec_from_file_location("sql_extractor", "src/forge/database/sql_extractor.py")
    sql_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sql_extractor_module)
    
    SQLExtractor = sql_extractor_module.SQLExtractor
    ExtractionConfig = sql_extractor_module.ExtractionConfig
    DatabaseConnector = connector_module.DatabaseConnector
    DatabaseConfig = connector_module.DatabaseConfig
    
    # Create mock connector
    config = DatabaseConfig(
        connection_type='postgresql',
        host='localhost',
        port=5432,
        database='test_db'
    )
    
    mock_connector = Mock(spec=DatabaseConnector)
    mock_connector.config = config
    mock_connector.is_connected.return_value = True
    
    # Test extractor initialization
    extractor = SQLExtractor(mock_connector)
    assert extractor.connector == mock_connector
    
    # Test invalid database type
    invalid_config = DatabaseConfig(
        connection_type='mongodb',
        host='localhost',
        port=27017,
        database='test_db'
    )
    
    invalid_connector = Mock(spec=DatabaseConnector)
    invalid_connector.config = invalid_config
    
    try:
        SQLExtractor(invalid_connector)
        assert False, "Should have raised ValueError for invalid database type"
    except ValueError as e:
        assert "SQLExtractor only supports PostgreSQL and MySQL" in str(e)
    
    print("✓ SQLExtractor tests passed")

def test_nosql_extractor():
    """Test NoSQLExtractor functionality with mocked dependencies."""
    print("Testing NoSQLExtractor...")
    
    # Import directly from module file bypassing __init__.py
    import importlib.util
    
    # Import connector first
    spec = importlib.util.spec_from_file_location("connector", "src/forge/database/connector.py")
    connector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector_module)
    
    # Import NoSQL extractor
    spec = importlib.util.spec_from_file_location("nosql_extractor", "src/forge/database/nosql_extractor.py")
    nosql_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nosql_extractor_module)
    
    NoSQLExtractor = nosql_extractor_module.NoSQLExtractor
    NoSQLExtractionConfig = nosql_extractor_module.NoSQLExtractionConfig
    DatabaseConnector = connector_module.DatabaseConnector
    DatabaseConfig = connector_module.DatabaseConfig
    
    # Test MongoDB connector
    mongodb_config = DatabaseConfig(
        connection_type='mongodb',
        host='localhost',
        port=27017,
        database='test_db'
    )
    
    mock_mongodb_connector = Mock(spec=DatabaseConnector)
    mock_mongodb_connector.config = mongodb_config
    mock_mongodb_connector.is_connected.return_value = True
    
    # Test extractor initialization
    extractor = NoSQLExtractor(mock_mongodb_connector)
    assert extractor.connector == mock_mongodb_connector
    assert extractor.db_type == 'mongodb'
    
    # Test Elasticsearch connector
    elasticsearch_config = DatabaseConfig(
        connection_type='elasticsearch',
        host='localhost',
        port=9200,
        database='test_cluster'
    )
    
    mock_elasticsearch_connector = Mock(spec=DatabaseConnector)
    mock_elasticsearch_connector.config = elasticsearch_config
    mock_elasticsearch_connector.is_connected.return_value = True
    
    extractor = NoSQLExtractor(mock_elasticsearch_connector)
    assert extractor.connector == mock_elasticsearch_connector
    assert extractor.db_type == 'elasticsearch'
    
    # Test invalid database type
    invalid_config = DatabaseConfig(
        connection_type='postgresql',
        host='localhost',
        port=5432,
        database='test_db'
    )
    
    invalid_connector = Mock(spec=DatabaseConnector)
    invalid_connector.config = invalid_config
    
    try:
        NoSQLExtractor(invalid_connector)
        assert False, "Should have raised ValueError for invalid database type"
    except ValueError as e:
        assert "NoSQLExtractor only supports MongoDB and Elasticsearch" in str(e)
    
    print("✓ NoSQLExtractor tests passed")

def test_extraction_configs():
    """Test extraction configuration classes."""
    print("Testing extraction configurations...")
    
    # Import directly from module files
    import importlib.util
    
    # Import SQL extractor config
    spec = importlib.util.spec_from_file_location("sql_extractor", "src/forge/database/sql_extractor.py")
    sql_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sql_extractor_module)
    
    # Import NoSQL extractor config
    spec = importlib.util.spec_from_file_location("nosql_extractor", "src/forge/database/nosql_extractor.py")
    nosql_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nosql_extractor_module)
    
    ExtractionConfig = sql_extractor_module.ExtractionConfig
    NoSQLExtractionConfig = nosql_extractor_module.NoSQLExtractionConfig
    
    # Test SQL extraction config
    sql_config = ExtractionConfig(
        table_name='users',
        columns=['id', 'name', 'email'],
        where_clause='active = true',
        order_by='name ASC',
        limit=100,
        batch_size=50
    )
    
    assert sql_config.table_name == 'users'
    assert sql_config.columns == ['id', 'name', 'email']
    assert sql_config.limit == 100
    assert sql_config.batch_size == 50
    
    # Test NoSQL extraction config
    nosql_config = NoSQLExtractionConfig(
        collection_name='users',
        query={'active': True},
        projection={'name': 1, 'email': 1},
        limit=100,
        batch_size=50
    )
    
    assert nosql_config.collection_name == 'users'
    assert nosql_config.query == {'active': True}
    assert nosql_config.projection == {'name': 1, 'email': 1}
    assert nosql_config.limit == 100
    assert nosql_config.batch_size == 50
    
    print("✓ Extraction configuration tests passed")

def main():
    """Run all tests."""
    print("Running database extractor tests...")
    
    try:
        test_sql_extractor()
        test_nosql_extractor()
        test_extraction_configs()
        print("\n✅ All extractor tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()