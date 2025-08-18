"""
Unit tests for database connector functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.qudata.database.connector import DatabaseConnector, DatabaseConfig, ConnectionInfo


class TestDatabaseConnector:
    """Test database connector functionality."""
    
    def test_init_with_valid_config(self):
        """Test connector initialization with valid configuration."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db',
            username='user',
            password='pass'
        )
        
        connector = DatabaseConnector(config)
        assert connector.config == config
        assert connector._connection is None
        assert connector._connection_pool is None
    
    def test_init_with_invalid_database_type(self):
        """Test connector initialization with invalid database type."""
        config = DatabaseConfig(
            connection_type='invalid_db',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseConnector(config)
    
    def test_supported_databases(self):
        """Test that all supported databases are recognized."""
        supported_types = ['postgresql', 'mysql', 'mongodb', 'elasticsearch']
        
        for db_type in supported_types:
            config = DatabaseConfig(
                connection_type=db_type,
                host='localhost',
                port=5432,
                database='test_db'
            )
            connector = DatabaseConnector(config)
            assert connector.config.connection_type == db_type
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_connect_postgresql_success(self, mock_sqlalchemy, mock_create_engine):
        """Test successful PostgreSQL connection."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db',
            username='user',
            password='pass'
        )
        
        # Mock engine and connection
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        connector = DatabaseConnector(config)
        result = connector.connect()
        
        assert result is True
        assert connector.is_connected() is True
        assert connector._connection_pool == mock_engine
        
        # Verify connection string construction
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0]
        assert 'postgresql+psycopg2://user:pass@localhost:5432/test_db' in call_args[0]
    
    @patch('src.qudata.database.connector.MongoClient')
    def test_connect_mongodb_success(self, mock_mongo_client):
        """Test successful MongoDB connection."""
        config = DatabaseConfig(
            connection_type='mongodb',
            host='localhost',
            port=27017,
            database='test_db',
            username='user',
            password='pass'
        )
        
        # Mock MongoDB client
        mock_client = Mock()
        mock_mongo_client.return_value = mock_client
        
        connector = DatabaseConnector(config)
        result = connector.connect()
        
        assert result is True
        assert connector.is_connected() is True
        assert connector._connection == mock_client
        
        # Verify ping was called
        mock_client.admin.command.assert_called_once_with('ping')
    
    @patch('src.qudata.database.connector.Elasticsearch')
    def test_connect_elasticsearch_success(self, mock_elasticsearch):
        """Test successful Elasticsearch connection."""
        config = DatabaseConfig(
            connection_type='elasticsearch',
            host='localhost',
            port=9200,
            database='test_cluster'
        )
        
        # Mock Elasticsearch client
        mock_es = Mock()
        mock_es.ping.return_value = True
        mock_elasticsearch.return_value = mock_es
        
        connector = DatabaseConnector(config)
        result = connector.connect()
        
        assert result is True
        assert connector.is_connected() is True
        assert connector._connection == mock_es
        
        # Verify ping was called
        mock_es.ping.assert_called_once()
    
    def test_connect_failure(self):
        """Test connection failure handling."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='invalid_host',
            port=5432,
            database='test_db'
        )
        
        with patch('src.qudata.database.connector.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection failed")
            
            connector = DatabaseConnector(config)
            result = connector.connect()
            
            assert result is False
            assert connector.is_connected() is False
            assert connector._connection_info.is_connected is False
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_get_connection_context_manager(self, mock_sqlalchemy, mock_create_engine):
        """Test connection context manager for SQL databases."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        # Mock engine and connection
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        connector = DatabaseConnector(config)
        connector.connect()
        
        with connector.get_connection() as conn:
            assert conn == mock_connection
    
    def test_get_connection_not_connected(self):
        """Test get_connection when not connected."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        connector = DatabaseConnector(config)
        
        with pytest.raises(ConnectionError, match="Database not connected"):
            with connector.get_connection():
                pass
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_test_connection_success(self, mock_sqlalchemy, mock_create_engine):
        """Test connection testing functionality."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        # Mock engine and connection
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        connector = DatabaseConnector(config)
        connector.connect()
        
        result = connector.test_connection()
        assert result is True
    
    def test_test_connection_failure(self):
        """Test connection testing with failure."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        connector = DatabaseConnector(config)
        # Don't connect
        
        result = connector.test_connection()
        assert result is False
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_get_postgresql_schema_info(self, mock_sqlalchemy, mock_create_engine):
        """Test PostgreSQL schema information retrieval."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        # Mock engine and connection
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        # Mock query results
        mock_tables_result = [
            Mock(_mapping={'table_name': 'users', 'table_type': 'BASE TABLE'}),
            Mock(_mapping={'table_name': 'posts', 'table_type': 'BASE TABLE'})
        ]
        
        mock_columns_result = [
            Mock(_mapping={
                'column_name': 'id',
                'data_type': 'integer',
                'is_nullable': 'NO',
                'column_default': 'nextval(...)'
            }),
            Mock(_mapping={
                'column_name': 'name',
                'data_type': 'character varying',
                'is_nullable': 'YES',
                'column_default': None
            })
        ]
        
        mock_connection.execute.side_effect = [
            mock_tables_result,  # First call for tables
            mock_columns_result,  # Second call for columns of first table
            mock_columns_result   # Third call for columns of second table
        ]
        
        connector = DatabaseConnector(config)
        connector.connect()
        
        schema_info = connector.get_schema_info()
        
        assert schema_info['database_type'] == 'postgresql'
        assert schema_info['database_name'] == 'test_db'
        assert len(schema_info['tables']) == 2
        assert 'users' in schema_info['table_columns']
        assert 'posts' in schema_info['table_columns']
    
    @patch('src.qudata.database.connector.MongoClient')
    def test_get_mongodb_schema_info(self, mock_mongo_client):
        """Test MongoDB schema information retrieval."""
        config = DatabaseConfig(
            connection_type='mongodb',
            host='localhost',
            port=27017,
            database='test_db'
        )
        
        # Mock MongoDB client and database
        mock_client = Mock()
        mock_db = Mock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client
        
        # Mock collections and stats
        mock_db.list_collection_names.return_value = ['users', 'posts']
        mock_db.command.return_value = {
            'count': 100,
            'size': 1024
        }
        
        # Mock collection find
        mock_collection = Mock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.find.return_value.limit.return_value = [
            {'_id': '123', 'name': 'John', 'age': 30},
            {'_id': '456', 'name': 'Jane', 'age': 25}
        ]
        
        connector = DatabaseConnector(config)
        connector.connect()
        
        schema_info = connector.get_schema_info()
        
        assert schema_info['database_type'] == 'mongodb'
        assert schema_info['database_name'] == 'test_db'
        assert 'users' in schema_info['collections']
        assert 'posts' in schema_info['collections']
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_close_connection(self, mock_sqlalchemy, mock_create_engine):
        """Test connection closing."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        # Mock engine
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        connector = DatabaseConnector(config)
        connector.connect()
        
        assert connector.is_connected() is True
        
        connector.close()
        
        assert connector.is_connected() is False
        mock_engine.dispose.assert_called_once()
    
    @patch('src.qudata.database.connector.create_engine')
    @patch('src.qudata.database.connector.sqlalchemy')
    def test_context_manager(self, mock_sqlalchemy, mock_create_engine):
        """Test connector as context manager."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        # Mock engine
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_sqlalchemy.text.return_value = "SELECT 1"
        
        with DatabaseConnector(config) as connector:
            assert connector.is_connected() is True
        
        # Connection should be closed after exiting context
        mock_engine.dispose.assert_called_once()
    
    def test_connection_string_with_ssl(self):
        """Test connection string construction with SSL parameters."""
        config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db',
            username='user',
            password='pass',
            ssl_mode='require',
            additional_params={'connect_timeout': '10'}
        )
        
        with patch('src.qudata.database.connector.create_engine') as mock_create_engine:
            with patch('src.qudata.database.connector.sqlalchemy'):
                mock_engine = Mock()
                mock_connection = Mock()
                mock_engine.connect.return_value.__enter__.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                connector = DatabaseConnector(config)
                connector.connect()
                
                # Verify SSL and additional parameters in connection string
                call_args = mock_create_engine.call_args[0]
                connection_string = call_args[0]
                assert 'sslmode=require' in connection_string
                assert 'connect_timeout=10' in connection_string