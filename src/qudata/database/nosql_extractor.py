"""
NoSQL database extractor for MongoDB and Elasticsearch.

Provides functionality to extract data from MongoDB and Elasticsearch
with support for queries, aggregations, and data transformation.
"""

import logging
from typing import Dict, Any, List, Optional, Iterator, Union
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .connector import DatabaseConnector, DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class NoSQLExtractionConfig:
    """Configuration for NoSQL data extraction."""
    collection_name: str  # MongoDB collection or Elasticsearch index
    query: Optional[Dict[str, Any]] = None  # MongoDB query or Elasticsearch query DSL
    projection: Optional[Dict[str, Any]] = None  # MongoDB projection or Elasticsearch _source
    sort: Optional[Dict[str, Any]] = None  # Sort specification
    limit: Optional[int] = None
    skip: Optional[int] = 0
    batch_size: int = 1000
    include_metadata: bool = True
    aggregation_pipeline: Optional[List[Dict[str, Any]]] = None  # MongoDB aggregation


@dataclass
class NoSQLExtractionResult:
    """Result of NoSQL data extraction."""
    data: List[Dict[str, Any]]
    dataframe: Optional[Any]  # pd.DataFrame when pandas is available
    total_documents: int
    extracted_documents: int
    execution_time: float
    query: Dict[str, Any]
    metadata: Dict[str, Any]


class NoSQLExtractor:
    """NoSQL database data extractor for MongoDB and Elasticsearch."""
    
    def __init__(self, connector: DatabaseConnector):
        """Initialize NoSQL extractor with database connector."""
        if connector.config.connection_type not in ['mongodb', 'elasticsearch']:
            raise ValueError(
                f"NoSQLExtractor only supports MongoDB and Elasticsearch, "
                f"got: {connector.config.connection_type}"
            )
        
        self.connector = connector
        self.db_type = connector.config.connection_type
    
    def extract_collection(self, config: NoSQLExtractionConfig) -> NoSQLExtractionResult:
        """Extract data from a collection/index."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        start_time = datetime.now()
        
        if self.db_type == 'mongodb':
            result = self._extract_mongodb_collection(config)
        else:  # elasticsearch
            result = self._extract_elasticsearch_index(config)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to DataFrame if requested
        dataframe = None
        if result['data'] and HAS_PANDAS:
            try:
                dataframe = pd.json_normalize(result['data'])
            except Exception as e:
                logger.warning(f"Could not convert to DataFrame: {e}")
        
        # Collect metadata
        metadata = {
            'collection_name': config.collection_name,
            'extraction_timestamp': start_time.isoformat(),
            'database_type': self.db_type,
            'database_name': self.connector.config.database
        }
        
        if config.include_metadata:
            metadata.update(result.get('metadata', {}))
        
        return NoSQLExtractionResult(
            data=result['data'],
            dataframe=dataframe,
            total_documents=result['total_documents'],
            extracted_documents=len(result['data']),
            execution_time=execution_time,
            query=result['query'],
            metadata=metadata
        )
    
    def extract_with_aggregation(self, 
                                collection_name: str,
                                pipeline: List[Dict[str, Any]]) -> NoSQLExtractionResult:
        """Extract data using aggregation pipeline (MongoDB only)."""
        if self.db_type != 'mongodb':
            raise ValueError("Aggregation is only supported for MongoDB")
        
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        start_time = datetime.now()
        
        try:
            db = self.connector.get_raw_connection()[self.connector.config.database]
            collection = db[collection_name]
            
            # Execute aggregation
            cursor = collection.aggregate(pipeline)
            data = list(cursor)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to DataFrame
            dataframe = None
            if data and HAS_PANDAS:
                try:
                    dataframe = pd.json_normalize(data)
                except Exception as e:
                    logger.warning(f"Could not convert aggregation result to DataFrame: {e}")
            
            metadata = {
                'collection_name': collection_name,
                'extraction_timestamp': start_time.isoformat(),
                'database_type': 'mongodb',
                'database_name': self.connector.config.database,
                'aggregation_pipeline': pipeline
            }
            
            return NoSQLExtractionResult(
                data=data,
                dataframe=dataframe,
                total_documents=len(data),
                extracted_documents=len(data),
                execution_time=execution_time,
                query={'aggregation_pipeline': pipeline},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Aggregation extraction failed: {e}")
            raise
    
    def extract_batched(self, config: NoSQLExtractionConfig) -> Iterator[NoSQLExtractionResult]:
        """Extract data in batches for large collections."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        # Get total count
        total_documents = self._get_document_count(config.collection_name, config.query)
        
        # Extract in batches
        skip = config.skip
        batch_num = 0
        
        while skip < total_documents:
            batch_config = NoSQLExtractionConfig(
                collection_name=config.collection_name,
                query=config.query,
                projection=config.projection,
                sort=config.sort,
                limit=config.batch_size,
                skip=skip,
                batch_size=config.batch_size,
                include_metadata=config.include_metadata
            )
            
            result = self.extract_collection(batch_config)
            result.metadata['batch_number'] = batch_num
            result.metadata['total_batches'] = (total_documents + config.batch_size - 1) // config.batch_size
            result.metadata['batch_skip'] = skip
            
            yield result
            
            if len(result.data) < config.batch_size:
                break  # Last batch
            
            skip += config.batch_size
            batch_num += 1
    
    def extract_multiple_collections(self, 
                                   collection_configs: List[NoSQLExtractionConfig]) -> Dict[str, NoSQLExtractionResult]:
        """Extract data from multiple collections/indices."""
        results = {}
        
        for config in collection_configs:
            try:
                result = self.extract_collection(config)
                results[config.collection_name] = result
                logger.info(f"Extracted {result.extracted_documents} documents from {config.collection_name}")
            except Exception as e:
                logger.error(f"Failed to extract from collection {config.collection_name}: {e}")
                results[config.collection_name] = None
        
        return results
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection/index."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        if self.db_type == 'mongodb':
            return self._get_mongodb_collection_info(collection_name)
        else:  # elasticsearch
            return self._get_elasticsearch_index_info(collection_name)
    
    def list_collections(self) -> List[str]:
        """List all collections/indices in the database."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        if self.db_type == 'mongodb':
            db = self.connector.get_raw_connection()[self.connector.config.database]
            return db.list_collection_names()
        else:  # elasticsearch
            es = self.connector.get_raw_connection()
            indices = es.indices.get_alias(index="*")
            return list(indices.keys())
    
    def _extract_mongodb_collection(self, config: NoSQLExtractionConfig) -> Dict[str, Any]:
        """Extract data from MongoDB collection."""
        try:
            db = self.connector.get_raw_connection()[self.connector.config.database]
            collection = db[config.collection_name]
            
            # Build query
            query = config.query or {}
            
            # Get total count
            total_documents = collection.count_documents(query)
            
            # Build cursor
            cursor = collection.find(query, config.projection)
            
            if config.sort:
                cursor = cursor.sort(list(config.sort.items()))
            
            if config.skip:
                cursor = cursor.skip(config.skip)
            
            if config.limit:
                cursor = cursor.limit(config.limit)
            
            # Execute query
            data = []
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                data.append(doc)
            
            return {
                'data': data,
                'total_documents': total_documents,
                'query': query,
                'metadata': {
                    'projection': config.projection,
                    'sort': config.sort,
                    'skip': config.skip,
                    'limit': config.limit
                }
            }
            
        except Exception as e:
            logger.error(f"MongoDB extraction failed: {e}")
            raise
    
    def _extract_elasticsearch_index(self, config: NoSQLExtractionConfig) -> Dict[str, Any]:
        """Extract data from Elasticsearch index."""
        try:
            es = self.connector.get_raw_connection()
            
            # Build query
            query_body = {
                'query': config.query or {'match_all': {}},
                'size': config.limit or 10000,
                'from': config.skip or 0
            }
            
            if config.projection:
                query_body['_source'] = config.projection
            
            if config.sort:
                query_body['sort'] = config.sort
            
            # Execute query
            response = es.search(
                index=config.collection_name,
                body=query_body
            )
            
            # Extract documents
            data = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['_id'] = hit['_id']
                doc['_score'] = hit.get('_score')
                data.append(doc)
            
            total_documents = response['hits']['total']['value']
            
            return {
                'data': data,
                'total_documents': total_documents,
                'query': query_body,
                'metadata': {
                    'took': response.get('took'),
                    'timed_out': response.get('timed_out'),
                    'shards': response.get('_shards')
                }
            }
            
        except Exception as e:
            logger.error(f"Elasticsearch extraction failed: {e}")
            raise
    
    def _get_document_count(self, collection_name: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Get document count for a collection/index."""
        try:
            if self.db_type == 'mongodb':
                db = self.connector.get_raw_connection()[self.connector.config.database]
                collection = db[collection_name]
                return collection.count_documents(query or {})
            else:  # elasticsearch
                es = self.connector.get_raw_connection()
                query_body = {'query': query or {'match_all': {}}}
                response = es.count(index=collection_name, body=query_body)
                return response['count']
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def _get_mongodb_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get MongoDB collection information."""
        try:
            db = self.connector.get_raw_connection()[self.connector.config.database]
            collection = db[collection_name]
            
            # Get collection stats
            stats = db.command("collStats", collection_name)
            
            # Sample documents to infer schema
            sample_docs = list(collection.find().limit(10))
            
            # Extract field types from sample documents
            fields = {}
            for doc in sample_docs:
                for field, value in doc.items():
                    field_type = type(value).__name__
                    if field not in fields:
                        fields[field] = set()
                    fields[field].add(field_type)
            
            # Convert sets to lists for JSON serialization
            fields = {field: list(types) for field, types in fields.items()}
            
            # Get indices
            indices = list(collection.list_indexes())
            
            return {
                'collection_name': collection_name,
                'document_count': stats.get('count', 0),
                'size_bytes': stats.get('size', 0),
                'storage_size': stats.get('storageSize', 0),
                'average_object_size': stats.get('avgObjSize', 0),
                'fields': fields,
                'indices': indices,
                'capped': stats.get('capped', False)
            }
            
        except Exception as e:
            logger.error(f"Failed to get MongoDB collection info: {e}")
            return {'error': str(e)}
    
    def _get_elasticsearch_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get Elasticsearch index information."""
        try:
            es = self.connector.get_raw_connection()
            
            # Get index stats
            stats = es.indices.stats(index=index_name)
            
            # Get mapping
            mapping = es.indices.get_mapping(index=index_name)
            
            # Get settings
            settings = es.indices.get_settings(index=index_name)
            
            index_stats = stats['indices'][index_name]
            
            return {
                'index_name': index_name,
                'document_count': index_stats['total']['docs']['count'],
                'size_bytes': index_stats['total']['store']['size_in_bytes'],
                'mapping': mapping[index_name]['mappings'],
                'settings': settings[index_name]['settings'],
                'shards': {
                    'primary': index_stats['primaries'],
                    'total': index_stats['total']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Elasticsearch index info: {e}")
            return {'error': str(e)}