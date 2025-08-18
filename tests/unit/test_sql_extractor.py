"""
Unit tests for SQL extractor functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.qudata.database.connector import DatabaseConnector, DatabaseConfig
from src.qudata.database.sql_extractor import SQLExtractor, ExtractionConfig, ExtractionResult


class TestSQLExtractor:
    """Test SQL extractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        self.mock_connector = Mock(spec=DatabaseConnector)
        self.mock_connector.config = self.config
        self.mock_connector.is_connected.return_value = True
    
    def test_init_with_valid_connector(self):
        """Test extractor initialization with valid connector."""
        extractor = SQLExtractor(self.mock_connector)
        assert extractor.connector == self.mock_connector
        assert extractor.query_builder is not None
    
    def test_init_with_invalid_database_type(self):
        """Test extractor initialization with invalid database type."""
        invalid_config = DatabaseConfig(
            connection_type='mongodb',
            host='localhost',
            port=27017,
            database='test_db'
        )
        
        invalid_connector = Mock(spec=DatabaseConnector)
        invalid_connector.config = invalid_config
        
        with pytest.raises(ValueError, match="SQLExtractor only supports PostgreSQL and MySQL"):
            SQLExtractor(invalid_connector)
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_table_basic(self, mock_read_sql):
        """Test basic table extraction."""
        # Mock DataFrame result
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        mock_read_sql.return_value = mock_df
        
        # Mock count query
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.return_value = 3
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(
            table_name='users',
            columns=['id', 'name', 'age'],
            include_metadata=True
        )
        
        result = extractor.extract_table(config)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.data) == 3
        assert result.total_rows == 3
        assert result.extracted_rows == 3
        assert result.metadata['table_name'] == 'users'
        assert 'extraction_timestamp' in result.metadata
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_table_with_where_clause(self, mock_read_sql):
        """Test table extraction with WHERE clause."""
        mock_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })
        mock_read_sql.return_value = mock_df
        
        # Mock count query
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.return_value = 2
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(
            table_name='users',
            where_clause='age < 35',
            include_metadata=False
        )
        
        result = extractor.extract_table(config)
        
        assert len(result.data) == 2
        assert result.total_rows == 2
        assert 'WHERE age < 35' in result.query
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_table_with_limit_offset(self, mock_read_sql):
        """Test table extraction with LIMIT and OFFSET."""
        mock_df = pd.DataFrame({
            'id': [11, 12],
            'name': ['User11', 'User12']
        })
        mock_read_sql.return_value = mock_df
        
        # Mock count query
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.return_value = 100
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(
            table_name='users',
            limit=2,
            offset=10,
            include_metadata=False
        )
        
        result = extractor.extract_table(config)
        
        assert len(result.data) == 2
        assert result.total_rows == 100
        assert 'LIMIT 2' in result.query
        assert 'OFFSET 10' in result.query
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_with_custom_query(self, mock_read_sql):
        """Test extraction with custom SQL query."""
        mock_df = pd.DataFrame({
            'user_count': [5],
            'avg_age': [28.5]
        })
        mock_read_sql.return_value = mock_df
        
        extractor = SQLExtractor(self.mock_connector)
        
        custom_query = "SELECT COUNT(*) as user_count, AVG(age) as avg_age FROM users"
        result = extractor.extract_with_query(custom_query)
        
        assert len(result.data) == 1
        assert result.metadata['custom_query'] is True
        assert result.query == custom_query
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_with_query_parameters(self, mock_read_sql):
        """Test extraction with parameterized query."""
        mock_df = pd.DataFrame({
            'id': [1],
            'name': ['Alice']
        })
        mock_read_sql.return_value = mock_df
        
        extractor = SQLExtractor(self.mock_connector)
        
        query = "SELECT * FROM users WHERE age > :min_age"
        params = {'min_age': 25}
        
        result = extractor.extract_with_query(query, params)
        
        assert len(result.data) == 1
        mock_read_sql.assert_called_once()
        # Verify parameters were passed
        call_args = mock_read_sql.call_args
        assert call_args[1]['params'] == params
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_batched(self, mock_read_sql):
        """Test batched extraction for large tables."""
        # Mock different batches
        batch1_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        batch2_df = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'David']})
        batch3_df = pd.DataFrame({'id': [5], 'name': ['Eve']})
        
        mock_read_sql.side_effect = [batch1_df, batch2_df, batch3_df]
        
        # Mock count query
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.return_value = 5
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(
            table_name='users',
            batch_size=2,
            include_metadata=False
        )
        
        batches = list(extractor.extract_batched(config))
        
        assert len(batches) == 3
        assert len(batches[0].data) == 2
        assert len(batches[1].data) == 2
        assert len(batches[2].data) == 1
        
        # Check batch metadata
        assert batches[0].metadata['batch_number'] == 0
        assert batches[1].metadata['batch_number'] == 1
        assert batches[2].metadata['batch_number'] == 2
        assert batches[0].metadata['total_batches'] == 3
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_multiple_tables(self, mock_read_sql):
        """Test extraction from multiple tables."""
        users_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        posts_df = pd.DataFrame({'id': [1, 2], 'title': ['Post1', 'Post2']})
        
        mock_read_sql.side_effect = [users_df, posts_df]
        
        # Mock count queries
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.side_effect = [2, 2]
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        
        configs = [
            ExtractionConfig(table_name='users', include_metadata=False),
            ExtractionConfig(table_name='posts', include_metadata=False)
        ]
        
        results = extractor.extract_multiple_tables(configs)
        
        assert 'users' in results
        assert 'posts' in results
        assert len(results['users'].data) == 2
        assert len(results['posts'].data) == 2
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_with_join(self, mock_read_sql):
        """Test extraction with table joins."""
        joined_df = pd.DataFrame({
            'user_id': [1, 2],
            'user_name': ['Alice', 'Bob'],
            'post_title': ['Post1', 'Post2']
        })
        mock_read_sql.return_value = joined_df
        
        extractor = SQLExtractor(self.mock_connector)
        
        join_specs = [
            {
                'table': 'posts',
                'join_type': 'INNER',
                'on_condition': 'users.id = posts.user_id'
            }
        ]
        
        result = extractor.extract_with_join(
            main_table='users',
            join_specs=join_specs,
            columns=['users.id as user_id', 'users.name as user_name', 'posts.title as post_title']
        )
        
        assert len(result.data) == 2
        assert result.metadata['main_table'] == 'users'
        assert 'posts' in result.metadata['joined_tables']
        assert 'JOIN' in result.query
    
    def test_extract_table_not_connected(self):
        """Test extraction when database is not connected."""
        self.mock_connector.is_connected.return_value = False
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(table_name='users')
        
        with pytest.raises(ConnectionError, match="Database not connected"):
            extractor.extract_table(config)
    
    @patch('src.qudata.database.sql_extractor.pd.read_sql')
    def test_extract_table_query_failure(self, mock_read_sql):
        """Test handling of query execution failure."""
        mock_read_sql.side_effect = Exception("Query failed")
        
        extractor = SQLExtractor(self.mock_connector)
        
        config = ExtractionConfig(table_name='users', include_metadata=False)
        
        with pytest.raises(Exception, match="Query failed"):
            extractor.extract_table(config)
    
    def test_list_tables_postgresql(self):
        """Test listing tables for PostgreSQL."""
        mock_result = [
            {'table_name': 'users', 'table_type': 'BASE TABLE', 'table_schema': 'public'},
            {'table_name': 'posts', 'table_type': 'BASE TABLE', 'table_schema': 'public'}
        ]
        
        mock_connection = Mock()
        mock_connection.execute.return_value = [
            Mock(_mapping=row) for row in mock_result
        ]
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        tables = extractor.list_tables()
        
        assert len(tables) == 2
        assert tables[0]['table_name'] == 'users'
        assert tables[1]['table_name'] == 'posts'
    
    def test_list_tables_mysql(self):
        """Test listing tables for MySQL."""
        mysql_config = DatabaseConfig(
            connection_type='mysql',
            host='localhost',
            port=3306,
            database='test_db'
        )
        
        mysql_connector = Mock(spec=DatabaseConnector)
        mysql_connector.config = mysql_config
        mysql_connector.is_connected.return_value = True
        
        mock_result = [
            {'table_name': 'users', 'table_type': 'BASE TABLE', 'table_schema': 'test_db'},
            {'table_name': 'posts', 'table_type': 'BASE TABLE', 'table_schema': 'test_db'}
        ]
        
        mock_connection = Mock()
        mock_connection.execute.return_value = [
            Mock(_mapping=row) for row in mock_result
        ]
        mysql_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(mysql_connector)
        tables = extractor.list_tables()
        
        assert len(tables) == 2
        # Verify MySQL-specific query was used
        call_args = mock_connection.execute.call_args[0]
        query = str(call_args[0])
        assert 'test_db' in query
    
    def test_get_table_info(self):
        """Test getting table information."""
        # Mock schema query result
        schema_result = [
            {
                'column_name': 'id',
                'data_type': 'integer',
                'is_nullable': 'NO',
                'column_default': 'nextval(...)'
            },
            {
                'column_name': 'name',
                'data_type': 'character varying',
                'is_nullable': 'YES',
                'column_default': None
            }
        ]
        
        mock_connection = Mock()
        mock_connection.execute.return_value = [
            Mock(_mapping=row) for row in schema_result
        ]
        mock_connection.execute.return_value.scalar.return_value = 100
        
        self.mock_connector.get_connection.return_value.__enter__.return_value = mock_connection
        
        extractor = SQLExtractor(self.mock_connector)
        info = extractor.get_table_info('users')
        
        assert info['table_name'] == 'users'
        assert len(info['schema']) == 2
        assert info['row_count'] == 100
        assert 'size_info' in info