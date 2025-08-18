#!/usr/bin/env python3
"""
Basic test script for database connectivity layer without pandas dependency.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_query_builder():
    """Test QueryBuilder functionality."""
    print("Testing QueryBuilder...")
    
    # Import directly from module file bypassing __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("query_builder", "src/forge/database/query_builder.py")
    query_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query_builder_module)
    QueryBuilder = query_builder_module.QueryBuilder
    
    # Test PostgreSQL query builder
    builder = QueryBuilder('postgresql')
    
    # Test basic SELECT query
    query = builder.build_select_query(
        table='users',
        columns=['id', 'name', 'email'],
        where_clause='active = true',
        order_by='name ASC',
        limit=10
    )
    
    expected = 'SELECT "id", "name", "email" FROM "users" WHERE active = true ORDER BY name ASC LIMIT 10'
    assert query == expected, f"Expected: {expected}, Got: {query}"
    
    # Test COUNT query
    count_query = builder.build_count_query(
        table='users',
        where_clause='age > 18'
    )
    
    expected_count = 'SELECT COUNT(*) FROM "users" WHERE age > 18'
    assert count_query == expected_count, f"Expected: {expected_count}, Got: {count_query}"
    
    print("✓ QueryBuilder tests passed")

def test_database_connector():
    """Test DatabaseConnector functionality."""
    print("Testing DatabaseConnector...")
    
    # Import directly from module file bypassing __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("connector", "src/forge/database/connector.py")
    connector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector_module)
    DatabaseConnector = connector_module.DatabaseConnector
    DatabaseConfig = connector_module.DatabaseConfig
    
    # Test configuration
    config = DatabaseConfig(
        connection_type='postgresql',
        host='localhost',
        port=5432,
        database='test_db',
        username='user',
        password='pass'
    )
    
    # Test connector initialization
    connector = DatabaseConnector(config)
    assert connector.config == config
    assert not connector.is_connected()
    
    # Test invalid database type
    try:
        invalid_config = DatabaseConfig(
            connection_type='invalid_db',
            host='localhost',
            port=5432,
            database='test_db'
        )
        DatabaseConnector(invalid_config)
        assert False, "Should have raised ValueError for invalid database type"
    except ValueError as e:
        assert "Unsupported database type" in str(e)
    
    print("✓ DatabaseConnector tests passed")

def main():
    """Run all tests."""
    print("Running database connectivity layer tests...")
    
    try:
        test_query_builder()
        test_database_connector()
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()