"""
Database connector for multiple database types.

Provides unified interface for connecting to PostgreSQL, MySQL, MongoDB,
and Elasticsearch databases with connection pooling and error handling.
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    connection_type: str  # postgresql, mysql, mongodb, elasticsearch
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    ssl_mode: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class ConnectionInfo:
    """Information about database connection."""
    connection_type: str
    host: str
    port: int
    database: str
    is_connected: bool
    connection_time: Optional[float] = None
    schema_info: Optional[Dict[str, Any]] = None


class DatabaseConnector:
    """Universal database connector supporting multiple database types."""
    
    SUPPORTED_DATABASES = {
        'postgresql': 'PostgreSQL',
        'mysql': 'MySQL', 
        'mongodb': 'MongoDB',
        'elasticsearch': 'Elasticsearch',
        'sqlite': 'SQLite'
    }
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database connector with configuration."""
        self.config = config
        self._connection = None
        self._connection_pool = None
        self._connection_info = None
        
        if config.connection_type not in self.SUPPORTED_DATABASES:
            raise ValueError(
                f"Unsupported database type: {config.connection_type}. "
                f"Supported types: {list(self.SUPPORTED_DATABASES.keys())}"
            )
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            start_time = time.time()
            
            if self.config.connection_type in ['postgresql', 'mysql', 'sqlite']:
                self._connect_sql()
            elif self.config.connection_type == 'mongodb':
                self._connect_mongodb()
            elif self.config.connection_type == 'elasticsearch':
                self._connect_elasticsearch()
            
            connection_time = time.time() - start_time
            
            self._connection_info = ConnectionInfo(
                connection_type=self.config.connection_type,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                is_connected=True,
                connection_time=connection_time
            )
            
            logger.info(
                f"Connected to {self.config.connection_type} database "
                f"at {self.config.host}:{self.config.port} "
                f"in {connection_time:.2f}s"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connection_info = ConnectionInfo(
                connection_type=self.config.connection_type,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                is_connected=False
            )
            return False
    
    def _connect_sql(self):
        """Connect to SQL databases (PostgreSQL, MySQL)."""
        try:
            import sqlalchemy
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for SQL database connections. "
                "Install with: pip install sqlalchemy"
            )
        
        # Build connection string
        if self.config.connection_string:
            connection_string = self.config.connection_string
        else:
            if self.config.connection_type == 'postgresql':
                try:
                    import psycopg2
                except ImportError:
                    raise ImportError(
                        "psycopg2 is required for PostgreSQL connections. "
                        "Install with: pip install psycopg2-binary"
                    )
                driver = 'postgresql+psycopg2'
            elif self.config.connection_type == 'mysql':
                try:
                    import pymysql
                except ImportError:
                    raise ImportError(
                        "PyMySQL is required for MySQL connections. "
                        "Install with: pip install pymysql"
                    )
                driver = 'mysql+pymysql'
            elif self.config.connection_type == 'sqlite':
                driver = 'sqlite'
            
            if self.config.connection_type == 'sqlite':
                connection_string = f"{driver}:///{self.config.database}"
            else:
                auth_part = ""
                if self.config.username and self.config.password:
                    auth_part = f"{self.config.username}:{self.config.password}@"
                elif self.config.username:
                    auth_part = f"{self.config.username}@"
                
                connection_string = (
                    f"{driver}://{auth_part}{self.config.host}:"
                    f"{self.config.port}/{self.config.database}"
                )
            
            # Add SSL and additional parameters
            params = []
            if self.config.ssl_mode:
                params.append(f"sslmode={self.config.ssl_mode}")
            if self.config.additional_params:
                for key, value in self.config.additional_params.items():
                    params.append(f"{key}={value}")
            
            if params:
                connection_string += "?" + "&".join(params)
        
        # Create engine with connection pooling
        self._connection_pool = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_pre_ping=True,  # Validate connections before use
            echo=False  # Set to True for SQL debugging
        )
        
        # Test connection
        with self._connection_pool.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    
    def _connect_mongodb(self):
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB connections. "
                "Install with: pip install pymongo"
            )
        
        # Build connection string
        if self.config.connection_string:
            connection_string = self.config.connection_string
        else:
            auth_part = ""
            if self.config.username and self.config.password:
                auth_part = f"{self.config.username}:{self.config.password}@"
            
            connection_string = (
                f"mongodb://{auth_part}{self.config.host}:"
                f"{self.config.port}/{self.config.database}"
            )
            
            # Add additional parameters
            if self.config.additional_params:
                params = "&".join([
                    f"{key}={value}" 
                    for key, value in self.config.additional_params.items()
                ])
                connection_string += f"?{params}"
        
        self._connection = MongoClient(
            connection_string,
            maxPoolSize=self.config.pool_size,
            serverSelectionTimeoutMS=self.config.pool_timeout * 1000
        )
        
        # Test connection
        self._connection.admin.command('ping')
    
    def _connect_elasticsearch(self):
        """Connect to Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch is required for Elasticsearch connections. "
                "Install with: pip install elasticsearch"
            )
        
        # Build connection configuration
        if self.config.connection_string:
            hosts = [self.config.connection_string]
        else:
            host_config = {
                'host': self.config.host,
                'port': self.config.port
            }
            
            if self.config.username and self.config.password:
                host_config.update({
                    'http_auth': (self.config.username, self.config.password)
                })
            
            hosts = [host_config]
        
        # Additional configuration
        es_config = {
            'hosts': hosts,
            'timeout': self.config.pool_timeout,
            'max_retries': 3,
            'retry_on_timeout': True
        }
        
        if self.config.ssl_mode:
            es_config['use_ssl'] = True
            if self.config.ssl_mode == 'disable':
                es_config['verify_certs'] = False
        
        if self.config.additional_params:
            es_config.update(self.config.additional_params)
        
        self._connection = Elasticsearch(**es_config)
        
        # Test connection
        if not self._connection.ping():
            raise ConnectionError("Failed to ping Elasticsearch cluster")
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        if not self.is_connected():
            raise ConnectionError("Database not connected. Call connect() first.")
        
        if self.config.connection_type in ['postgresql', 'mysql', 'sqlite']:
            with self._connection_pool.connect() as conn:
                yield conn
        else:
            yield self._connection
    
    def get_raw_connection(self):
        """Get raw database connection object."""
        if not self.is_connected():
            raise ConnectionError("Database not connected. Call connect() first.")
        
        if self.config.connection_type in ['postgresql', 'mysql', 'sqlite']:
            return self._connection_pool
        else:
            return self._connection
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return (
            self._connection_info is not None and 
            self._connection_info.is_connected and
            (self._connection_pool is not None or self._connection is not None)
        )
    
    def get_connection_info(self) -> Optional[ConnectionInfo]:
        """Get connection information."""
        return self._connection_info
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if self.config.connection_type in ['postgresql', 'mysql', 'sqlite']:
                with self.get_connection() as conn:
                    import sqlalchemy
                    conn.execute(sqlalchemy.text("SELECT 1"))
            elif self.config.connection_type == 'mongodb':
                self._connection.admin.command('ping')
            elif self.config.connection_type == 'elasticsearch':
                return self._connection.ping()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        if not self.is_connected():
            raise ConnectionError("Database not connected")
        
        schema_info = {}
        
        try:
            if self.config.connection_type == 'postgresql':
                schema_info = self._get_postgresql_schema()
            elif self.config.connection_type == 'mysql':
                schema_info = self._get_mysql_schema()
            elif self.config.connection_type == 'mongodb':
                schema_info = self._get_mongodb_schema()
            elif self.config.connection_type == 'elasticsearch':
                schema_info = self._get_elasticsearch_schema()
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            schema_info = {'error': str(e)}
        
        return schema_info
    
    def _get_postgresql_schema(self) -> Dict[str, Any]:
        """Get PostgreSQL schema information."""
        with self.get_connection() as conn:
            import sqlalchemy
            
            # Get tables
            tables_query = sqlalchemy.text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [dict(row._mapping) for row in conn.execute(tables_query)]
            
            # Get columns for each table
            table_columns = {}
            for table in tables:
                table_name = table['table_name']
                columns_query = sqlalchemy.text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = :table_name
                    ORDER BY ordinal_position
                """)
                columns = [
                    dict(row._mapping) 
                    for row in conn.execute(columns_query, {'table_name': table_name})
                ]
                table_columns[table_name] = columns
            
            return {
                'database_type': 'postgresql',
                'database_name': self.config.database,
                'tables': tables,
                'table_columns': table_columns
            }
    
    def _get_mysql_schema(self) -> Dict[str, Any]:
        """Get MySQL schema information."""
        with self.get_connection() as conn:
            import sqlalchemy
            
            # Get tables
            tables_query = sqlalchemy.text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = :database_name
                ORDER BY table_name
            """)
            tables = [
                dict(row._mapping) 
                for row in conn.execute(tables_query, {'database_name': self.config.database})
            ]
            
            # Get columns for each table
            table_columns = {}
            for table in tables:
                table_name = table['table_name']
                columns_query = sqlalchemy.text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_schema = :database_name AND table_name = :table_name
                    ORDER BY ordinal_position
                """)
                columns = [
                    dict(row._mapping) 
                    for row in conn.execute(columns_query, {
                        'database_name': self.config.database,
                        'table_name': table_name
                    })
                ]
                table_columns[table_name] = columns
            
            return {
                'database_type': 'mysql',
                'database_name': self.config.database,
                'tables': tables,
                'table_columns': table_columns
            }
    
    def _get_mongodb_schema(self) -> Dict[str, Any]:
        """Get MongoDB schema information."""
        db = self._connection[self.config.database]
        
        # Get collections
        collections = db.list_collection_names()
        
        # Sample documents from each collection to infer schema
        collection_schemas = {}
        for collection_name in collections:
            collection = db[collection_name]
            
            # Get collection stats
            stats = db.command("collStats", collection_name)
            
            # Sample a few documents to infer schema
            sample_docs = list(collection.find().limit(5))
            
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
            
            collection_schemas[collection_name] = {
                'document_count': stats.get('count', 0),
                'size_bytes': stats.get('size', 0),
                'fields': fields
            }
        
        return {
            'database_type': 'mongodb',
            'database_name': self.config.database,
            'collections': collections,
            'collection_schemas': collection_schemas
        }
    
    def _get_elasticsearch_schema(self) -> Dict[str, Any]:
        """Get Elasticsearch schema information."""
        # Get indices
        indices = self._connection.indices.get_alias(index="*")
        
        # Get mappings for each index
        index_mappings = {}
        for index_name in indices.keys():
            try:
                mapping = self._connection.indices.get_mapping(index=index_name)
                index_mappings[index_name] = mapping[index_name]['mappings']
            except Exception as e:
                logger.warning(f"Failed to get mapping for index {index_name}: {e}")
                index_mappings[index_name] = {'error': str(e)}
        
        return {
            'database_type': 'elasticsearch',
            'cluster_name': self.config.database,
            'indices': list(indices.keys()),
            'index_mappings': index_mappings
        }
    
    def close(self):
        """Close database connection."""
        try:
            if self.config.connection_type in ['postgresql', 'mysql']:
                if self._connection_pool:
                    self._connection_pool.dispose()
            elif self.config.connection_type in ['mongodb', 'elasticsearch']:
                if self._connection:
                    self._connection.close()
            
            self._connection = None
            self._connection_pool = None
            
            if self._connection_info:
                self._connection_info.is_connected = False
            
            logger.info(f"Closed {self.config.connection_type} database connection")
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()