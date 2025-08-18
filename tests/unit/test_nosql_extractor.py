"""
Unit tests for NoSQL extractor functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.qudata.database.connector import DatabaseConnector, DatabaseConfig
from src.qudata.database.nosql_extractor import NoSQLExtractor, NoSQLExtractionConfig, NoSQLExtractionResult


class TestNoSQLExtractor:
    """Test NoSQL extractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mongodb_config = DatabaseConfig(
            connection_type='mongodb',
            host='localhost',
            port=27017,
            database='test_db'
        )
        
        self.elasticsearch_config = DatabaseConfig(
            connection_type='elasticsearch',
            host='localhost',
            port=9200,
            database='test_cluster'
        )
        
        self.mock_mongodb_connector = Mock(spec=DatabaseConnector)
        self.mock_mongodb_connector.config = self.mongodb_config
        self.mock_mongodb_connector.is_connected.return_value = True
        
        self.mock_elasticsearch_connector = Mock(spec=DatabaseConnector)
        self.mock_elasticsearch_connector.config = self.elasticsearch_config
        self.mock_elasticsearch_connector.is_connected.return_value = True
    
    def test_init_with_mongodb_connector(self):
        """Test extractor initialization with MongoDB connector."""
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        assert extractor.connector == self.mock_mongodb_connector
        assert extractor.db_type == 'mongodb'
    
    def test_init_with_elasticsearch_connector(self):
        """Test extractor initialization with Elasticsearch connector."""
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        assert extractor.connector == self.mock_elasticsearch_connector
        assert extractor.db_type == 'elasticsearch'
    
    def test_init_with_invalid_database_type(self):
        """Test extractor initialization with invalid database type."""
        invalid_config = DatabaseConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='test_db'
        )
        
        invalid_connector = Mock(spec=DatabaseConnector)
        invalid_connector.config = invalid_config
        
        with pytest.raises(ValueError, match="NoSQLExtractor only supports MongoDB and Elasticsearch"):
            NoSQLExtractor(invalid_connector)
    
    def test_extract_mongodb_collection_basic(self):
        """Test basic MongoDB collection extraction."""
        # Mock MongoDB database and collection
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock collection methods
        mock_collection.count_documents.return_value = 3
        mock_collection.find.return_value = [
            {'_id': '507f1f77bcf86cd799439011', 'name': 'Alice', 'age': 25},
            {'_id': '507f1f77bcf86cd799439012', 'name': 'Bob', 'age': 30},
            {'_id': '507f1f77bcf86cd799439013', 'name': 'Charlie', 'age': 35}
        ]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(
            collection_name='users',
            query={'age': {'$gte': 25}},
            include_metadata=True
        )
        
        result = extractor.extract_collection(config)
        
        assert isinstance(result, NoSQLExtractionResult)
        assert len(result.data) == 3
        assert result.total_documents == 3
        assert result.extracted_documents == 3
        assert result.metadata['collection_name'] == 'users'
        assert result.metadata['database_type'] == 'mongodb'
        
        # Verify ObjectId was converted to string
        assert isinstance(result.data[0]['_id'], str)
    
    def test_extract_mongodb_collection_with_projection(self):
        """Test MongoDB collection extraction with projection."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock cursor with projection
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.__iter__.return_value = iter([
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30}
        ])
        
        mock_collection.count_documents.return_value = 2
        mock_collection.find.return_value = mock_cursor
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(
            collection_name='users',
            projection={'name': 1, 'age': 1, '_id': 0},
            sort={'age': 1},
            limit=10,
            skip=5,
            include_metadata=False
        )
        
        result = extractor.extract_collection(config)
        
        assert len(result.data) == 2
        assert '_id' not in result.data[0]  # Excluded by projection
        
        # Verify cursor methods were called
        mock_collection.find.assert_called_once_with({}, {'name': 1, 'age': 1, '_id': 0})
        mock_cursor.sort.assert_called_once()
        mock_cursor.skip.assert_called_once_with(5)
        mock_cursor.limit.assert_called_once_with(10)
    
    def test_extract_elasticsearch_index_basic(self):
        """Test basic Elasticsearch index extraction."""
        # Mock Elasticsearch client
        mock_es = Mock()
        mock_response = {
            'hits': {
                'total': {'value': 2},
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 1.0,
                        '_source': {'name': 'Alice', 'age': 25}
                    },
                    {
                        '_id': 'doc2',
                        '_score': 0.8,
                        '_source': {'name': 'Bob', 'age': 30}
                    }
                ]
            },
            'took': 5,
            'timed_out': False,
            '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}
        }
        
        mock_es.search.return_value = mock_response
        self.mock_elasticsearch_connector.get_raw_connection.return_value = mock_es
        
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        
        config = NoSQLExtractionConfig(
            collection_name='users',
            query={'match': {'name': 'Alice'}},
            include_metadata=True
        )
        
        result = extractor.extract_collection(config)
        
        assert len(result.data) == 2
        assert result.total_documents == 2
        assert result.data[0]['_id'] == 'doc1'
        assert result.data[0]['_score'] == 1.0
        assert result.data[0]['name'] == 'Alice'
        assert result.metadata['took'] == 5
    
    def test_extract_elasticsearch_index_with_source_filter(self):
        """Test Elasticsearch extraction with _source filtering."""
        mock_es = Mock()
        mock_response = {
            'hits': {
                'total': {'value': 1},
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 1.0,
                        '_source': {'name': 'Alice'}  # Only name field returned
                    }
                ]
            }
        }
        
        mock_es.search.return_value = mock_response
        self.mock_elasticsearch_connector.get_raw_connection.return_value = mock_es
        
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        
        config = NoSQLExtractionConfig(
            collection_name='users',
            projection=['name'],  # _source filter
            sort={'age': {'order': 'asc'}},
            limit=100,
            skip=10,
            include_metadata=False
        )
        
        result = extractor.extract_collection(config)
        
        assert len(result.data) == 1
        assert 'age' not in result.data[0]  # Filtered out by _source
        
        # Verify search was called with correct parameters
        call_args = mock_es.search.call_args[1]
        assert call_args['body']['_source'] == ['name']
        assert call_args['body']['sort'] == {'age': {'order': 'asc'}}
        assert call_args['body']['size'] == 100
        assert call_args['body']['from'] == 10
    
    def test_extract_with_aggregation_mongodb(self):
        """Test MongoDB aggregation extraction."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock aggregation result
        aggregation_result = [
            {'_id': 'Engineering', 'count': 10, 'avg_age': 30.5},
            {'_id': 'Marketing', 'count': 5, 'avg_age': 28.2}
        ]
        
        mock_collection.aggregate.return_value = aggregation_result
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        pipeline = [
            {'$group': {
                '_id': '$department',
                'count': {'$sum': 1},
                'avg_age': {'$avg': '$age'}
            }}
        ]
        
        result = extractor.extract_with_aggregation('users', pipeline)
        
        assert len(result.data) == 2
        assert result.data[0]['_id'] == 'Engineering'
        assert result.data[0]['count'] == 10
        assert result.metadata['aggregation_pipeline'] == pipeline
    
    def test_extract_with_aggregation_elasticsearch_error(self):
        """Test that aggregation raises error for Elasticsearch."""
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        
        pipeline = [{'$group': {'_id': '$department'}}]
        
        with pytest.raises(ValueError, match="Aggregation is only supported for MongoDB"):
            extractor.extract_with_aggregation('users', pipeline)
    
    def test_extract_batched_mongodb(self):
        """Test batched extraction for MongoDB."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock total count
        mock_collection.count_documents.return_value = 5
        
        # Mock batched results
        batch1_data = [
            {'_id': '1', 'name': 'Alice'},
            {'_id': '2', 'name': 'Bob'}
        ]
        batch2_data = [
            {'_id': '3', 'name': 'Charlie'},
            {'_id': '4', 'name': 'David'}
        ]
        batch3_data = [
            {'_id': '5', 'name': 'Eve'}
        ]
        
        mock_collection.find.side_effect = [batch1_data, batch2_data, batch3_data]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(
            collection_name='users',
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
    
    def test_extract_multiple_collections(self):
        """Test extraction from multiple collections."""
        mock_db = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        
        # Mock collections
        mock_users_collection = Mock()
        mock_posts_collection = Mock()
        
        def get_collection(name):
            if name == 'users':
                return mock_users_collection
            elif name == 'posts':
                return mock_posts_collection
        
        mock_db.__getitem__.side_effect = get_collection
        
        # Mock collection data
        mock_users_collection.count_documents.return_value = 2
        mock_users_collection.find.return_value = [
            {'_id': '1', 'name': 'Alice'},
            {'_id': '2', 'name': 'Bob'}
        ]
        
        mock_posts_collection.count_documents.return_value = 3
        mock_posts_collection.find.return_value = [
            {'_id': '1', 'title': 'Post 1'},
            {'_id': '2', 'title': 'Post 2'},
            {'_id': '3', 'title': 'Post 3'}
        ]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        configs = [
            NoSQLExtractionConfig(collection_name='users', include_metadata=False),
            NoSQLExtractionConfig(collection_name='posts', include_metadata=False)
        ]
        
        results = extractor.extract_multiple_collections(configs)
        
        assert 'users' in results
        assert 'posts' in results
        assert len(results['users'].data) == 2
        assert len(results['posts'].data) == 3
    
    def test_list_collections_mongodb(self):
        """Test listing MongoDB collections."""
        mock_db = Mock()
        mock_db.list_collection_names.return_value = ['users', 'posts', 'comments']
        mock_client = {self.mongodb_config.database: mock_db}
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        collections = extractor.list_collections()
        
        assert collections == ['users', 'posts', 'comments']
    
    def test_list_collections_elasticsearch(self):
        """Test listing Elasticsearch indices."""
        mock_es = Mock()
        mock_es.indices.get_alias.return_value = {
            'users': {},
            'posts': {},
            'logs-2023': {}
        }
        
        self.mock_elasticsearch_connector.get_raw_connection.return_value = mock_es
        
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        collections = extractor.list_collections()
        
        assert set(collections) == {'users', 'posts', 'logs-2023'}
    
    def test_get_mongodb_collection_info(self):
        """Test getting MongoDB collection information."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock collection stats
        mock_db.command.return_value = {
            'count': 100,
            'size': 1024,
            'storageSize': 2048,
            'avgObjSize': 10.24,
            'capped': False
        }
        
        # Mock sample documents
        mock_collection.find.return_value.limit.return_value = [
            {'_id': '1', 'name': 'Alice', 'age': 25, 'active': True},
            {'_id': '2', 'name': 'Bob', 'age': 30, 'score': 95.5}
        ]
        
        # Mock indices
        mock_collection.list_indexes.return_value = [
            {'name': '_id_', 'key': {'_id': 1}},
            {'name': 'name_1', 'key': {'name': 1}}
        ]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        info = extractor.get_collection_info('users')
        
        assert info['collection_name'] == 'users'
        assert info['document_count'] == 100
        assert info['size_bytes'] == 1024
        assert 'name' in info['fields']
        assert 'age' in info['fields']
        assert 'str' in info['fields']['name']
        assert 'int' in info['fields']['age']
        assert len(info['indices']) == 2
    
    def test_get_elasticsearch_index_info(self):
        """Test getting Elasticsearch index information."""
        mock_es = Mock()
        
        # Mock index stats
        mock_es.indices.stats.return_value = {
            'indices': {
                'users': {
                    'total': {
                        'docs': {'count': 1000},
                        'store': {'size_in_bytes': 5242880}
                    },
                    'primaries': {
                        'docs': {'count': 1000},
                        'store': {'size_in_bytes': 5242880}
                    }
                }
            }
        }
        
        # Mock mapping
        mock_es.indices.get_mapping.return_value = {
            'users': {
                'mappings': {
                    'properties': {
                        'name': {'type': 'text'},
                        'age': {'type': 'integer'},
                        'created_at': {'type': 'date'}
                    }
                }
            }
        }
        
        # Mock settings
        mock_es.indices.get_settings.return_value = {
            'users': {
                'settings': {
                    'index': {
                        'number_of_shards': '1',
                        'number_of_replicas': '0'
                    }
                }
            }
        }
        
        self.mock_elasticsearch_connector.get_raw_connection.return_value = mock_es
        
        extractor = NoSQLExtractor(self.mock_elasticsearch_connector)
        info = extractor.get_collection_info('users')
        
        assert info['index_name'] == 'users'
        assert info['document_count'] == 1000
        assert info['size_bytes'] == 5242880
        assert 'properties' in info['mapping']
        assert 'index' in info['settings']
    
    def test_extract_collection_not_connected(self):
        """Test extraction when database is not connected."""
        self.mock_mongodb_connector.is_connected.return_value = False
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(collection_name='users')
        
        with pytest.raises(ConnectionError, match="Database not connected"):
            extractor.extract_collection(config)
    
    def test_dataframe_conversion_success(self):
        """Test successful DataFrame conversion."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        mock_collection.count_documents.return_value = 2
        mock_collection.find.return_value = [
            {'_id': '1', 'name': 'Alice', 'age': 25},
            {'_id': '2', 'name': 'Bob', 'age': 30}
        ]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(collection_name='users', include_metadata=False)
        result = extractor.extract_collection(config)
        
        assert result.dataframe is not None
        assert isinstance(result.dataframe, pd.DataFrame)
        assert len(result.dataframe) == 2
        assert 'name' in result.dataframe.columns
        assert 'age' in result.dataframe.columns
    
    @patch('src.qudata.database.nosql_extractor.pd.json_normalize')
    def test_dataframe_conversion_failure(self, mock_json_normalize):
        """Test DataFrame conversion failure handling."""
        mock_json_normalize.side_effect = Exception("Normalization failed")
        
        mock_db = Mock()
        mock_collection = Mock()
        mock_client = {self.mongodb_config.database: mock_db}
        mock_db.__getitem__.return_value = mock_collection
        
        mock_collection.count_documents.return_value = 1
        mock_collection.find.return_value = [
            {'_id': '1', 'complex_data': {'nested': {'deep': 'value'}}}
        ]
        
        self.mock_mongodb_connector.get_raw_connection.return_value = mock_client
        
        extractor = NoSQLExtractor(self.mock_mongodb_connector)
        
        config = NoSQLExtractionConfig(collection_name='users', include_metadata=False)
        result = extractor.extract_collection(config)
        
        # Should handle the error gracefully
        assert result.dataframe is None
        assert len(result.data) == 1