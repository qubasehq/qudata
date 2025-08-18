"""
SQL database extractor for structured data extraction.

Provides functionality to extract data from PostgreSQL and MySQL databases
using SQL queries with support for pagination, filtering, and data transformation.
"""

import logging
from typing import Dict, Any, List, Optional, Iterator, Union
from dataclasses import dataclass
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .connector import DatabaseConnector, DatabaseConfig
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for SQL data extraction."""
    table_name: str
    columns: Optional[List[str]] = None  # None means all columns
    where_clause: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = 0
    batch_size: int = 1000
    include_metadata: bool = True


@dataclass
class ExtractionResult:
    """Result of SQL data extraction."""
    data: Any  # pd.DataFrame when pandas is available, List[Dict] otherwise
    total_rows: int
    extracted_rows: int
    execution_time: float
    query: str
    metadata: Dict[str, Any]


class SQLExtractor:
    """SQL database data extractor for PostgreSQL and MySQL."""
    
    def __init__(self, connector: DatabaseConnector):
        """Initialize SQL extractor with database connector."""
        if connector.config.connection_type not in ['postgresql', 'mysql']:
            raise ValueError(
                f"SQLExtractor only supports PostgreSQL and MySQL, "
                f"got: {connector.config.connection_type}"
            )
        
        self.connector = connector
        self.query_builder = QueryBuilder(connector.config.connection_type)
    
    def extract_table(self, config: ExtractionConfig) -> ExtractionResult:
        """Extract data from a single table."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        start_time = datetime.now()
        
        # Build query
        query = self.query_builder.build_select_query(
            table=config.table_name,
            columns=config.columns,
            where_clause=config.where_clause,
            order_by=config.order_by,
            limit=config.limit,
            offset=config.offset
        )
        
        # Get total count if needed
        total_rows = 0
        if config.include_metadata:
            count_query = self.query_builder.build_count_query(
                table=config.table_name,
                where_clause=config.where_clause
            )
            total_rows = self._execute_count_query(count_query)
        
        # Execute main query
        data = self._execute_query_to_dataframe(query)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Collect metadata
        metadata = {
            'table_name': config.table_name,
            'extraction_timestamp': start_time.isoformat(),
            'database_type': self.connector.config.connection_type,
            'database_name': self.connector.config.database
        }
        
        if config.include_metadata:
            metadata.update({
                'table_schema': self._get_table_schema(config.table_name),
                'column_types': dict(data.dtypes.astype(str)) if HAS_PANDAS and hasattr(data, 'dtypes') and not data.empty else {}
            })
        
        return ExtractionResult(
            data=data,
            total_rows=total_rows,
            extracted_rows=len(data),
            execution_time=execution_time,
            query=query,
            metadata=metadata
        )
    
    def extract_with_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """Extract data using custom SQL query."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        start_time = datetime.now()
        
        # Execute query
        data = self._execute_query_to_dataframe(query, params)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            'extraction_timestamp': start_time.isoformat(),
            'database_type': self.connector.config.connection_type,
            'database_name': self.connector.config.database,
            'custom_query': True,
            'column_types': dict(data.dtypes.astype(str)) if HAS_PANDAS and hasattr(data, 'dtypes') and not data.empty else {}
        }
        
        return ExtractionResult(
            data=data,
            total_rows=len(data),
            extracted_rows=len(data),
            execution_time=execution_time,
            query=query,
            metadata=metadata
        )
    
    def extract_batched(self, config: ExtractionConfig) -> Iterator[ExtractionResult]:
        """Extract data in batches for large tables."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        # Get total count
        count_query = self.query_builder.build_count_query(
            table=config.table_name,
            where_clause=config.where_clause
        )
        total_rows = self._execute_count_query(count_query)
        
        # Extract in batches
        offset = config.offset
        batch_num = 0
        
        while offset < total_rows:
            batch_config = ExtractionConfig(
                table_name=config.table_name,
                columns=config.columns,
                where_clause=config.where_clause,
                order_by=config.order_by,
                limit=config.batch_size,
                offset=offset,
                batch_size=config.batch_size,
                include_metadata=config.include_metadata
            )
            
            result = self.extract_table(batch_config)
            result.metadata['batch_number'] = batch_num
            result.metadata['total_batches'] = (total_rows + config.batch_size - 1) // config.batch_size
            result.metadata['batch_offset'] = offset
            
            yield result
            
            if len(result.data) < config.batch_size:
                break  # Last batch
            
            offset += config.batch_size
            batch_num += 1
    
    def extract_multiple_tables(self, table_configs: List[ExtractionConfig]) -> Dict[str, ExtractionResult]:
        """Extract data from multiple tables."""
        results = {}
        
        for config in table_configs:
            try:
                result = self.extract_table(config)
                results[config.table_name] = result
                logger.info(f"Extracted {result.extracted_rows} rows from {config.table_name}")
            except Exception as e:
                logger.error(f"Failed to extract from table {config.table_name}: {e}")
                results[config.table_name] = None
        
        return results
    
    def extract_with_join(self, 
                         main_table: str,
                         join_specs: List[Dict[str, str]],
                         columns: Optional[List[str]] = None,
                         where_clause: Optional[str] = None,
                         order_by: Optional[str] = None,
                         limit: Optional[int] = None) -> ExtractionResult:
        """Extract data with table joins."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        start_time = datetime.now()
        
        # Build join query
        query = self.query_builder.build_join_query(
            main_table=main_table,
            join_specs=join_specs,
            columns=columns,
            where_clause=where_clause,
            order_by=order_by,
            limit=limit
        )
        
        # Execute query
        data = self._execute_query_to_dataframe(query)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            'main_table': main_table,
            'joined_tables': [spec.get('table') for spec in join_specs],
            'extraction_timestamp': start_time.isoformat(),
            'database_type': self.connector.config.connection_type,
            'database_name': self.connector.config.database,
            'column_types': dict(data.dtypes.astype(str)) if HAS_PANDAS and hasattr(data, 'dtypes') and not data.empty else {}
        }
        
        return ExtractionResult(
            data=data,
            total_rows=len(data),
            extracted_rows=len(data),
            execution_time=execution_time,
            query=query,
            metadata=metadata
        )
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        info = {
            'table_name': table_name,
            'schema': self._get_table_schema(table_name),
            'row_count': self._get_table_row_count(table_name),
            'size_info': self._get_table_size_info(table_name)
        }
        
        return info
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """List all tables in the database."""
        if not self.connector.is_connected():
            raise ConnectionError("Database not connected")
        
        if self.connector.config.connection_type == 'postgresql':
            query = """
                SELECT table_name, table_type, table_schema
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
        else:  # mysql
            query = f"""
                SELECT table_name, table_type, table_schema
                FROM information_schema.tables 
                WHERE table_schema = '{self.connector.config.database}'
                ORDER BY table_name
            """
        
        return self._execute_query_to_dict_list(query)
    
    def _execute_query_to_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Execute query and return results as DataFrame or list of dicts."""
        try:
            if HAS_PANDAS:
                with self.connector.get_connection() as conn:
                    if params:
                        import sqlalchemy
                        query_obj = sqlalchemy.text(query)
                        data = pd.read_sql(query_obj, conn, params=params)
                    else:
                        data = pd.read_sql(query, conn)
                    return data
            else:
                # Fallback to list of dictionaries when pandas is not available
                return self._execute_query_to_dict_list(query, params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def _execute_query_to_dict_list(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries."""
        try:
            with self.connector.get_connection() as conn:
                import sqlalchemy
                if params:
                    query_obj = sqlalchemy.text(query)
                    result = conn.execute(query_obj, params)
                else:
                    query_obj = sqlalchemy.text(query)
                    result = conn.execute(query_obj)
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def _execute_count_query(self, query: str) -> int:
        """Execute count query and return the count."""
        try:
            with self.connector.get_connection() as conn:
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query))
                return result.scalar()
        except Exception as e:
            logger.error(f"Count query execution failed: {e}")
            logger.error(f"Query: {query}")
            return 0
    
    def _get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        if self.connector.config.connection_type == 'postgresql':
            query = """
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = :table_name
                ORDER BY ordinal_position
            """
        else:  # mysql
            query = f"""
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_schema = '{self.connector.config.database}' 
                AND table_name = :table_name
                ORDER BY ordinal_position
            """
        
        return self._execute_query_to_dict_list(query, {'table_name': table_name})
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get table row count."""
        query = f"SELECT COUNT(*) FROM {table_name}"
        return self._execute_count_query(query)
    
    def _get_table_size_info(self, table_name: str) -> Dict[str, Any]:
        """Get table size information."""
        size_info = {'table_name': table_name}
        
        try:
            if self.connector.config.connection_type == 'postgresql':
                query = """
                    SELECT 
                        pg_size_pretty(pg_total_relation_size(:table_name)) as total_size,
                        pg_size_pretty(pg_relation_size(:table_name)) as table_size,
                        pg_size_pretty(pg_total_relation_size(:table_name) - pg_relation_size(:table_name)) as index_size
                """
                result = self._execute_query_to_dict_list(query, {'table_name': table_name})
                if result:
                    size_info.update(result[0])
            
            elif self.connector.config.connection_type == 'mysql':
                query = f"""
                    SELECT 
                        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS total_size_mb,
                        ROUND((data_length / 1024 / 1024), 2) AS data_size_mb,
                        ROUND((index_length / 1024 / 1024), 2) AS index_size_mb,
                        table_rows
                    FROM information_schema.tables 
                    WHERE table_schema = '{self.connector.config.database}' 
                    AND table_name = :table_name
                """
                result = self._execute_query_to_dict_list(query, {'table_name': table_name})
                if result:
                    size_info.update(result[0])
        
        except Exception as e:
            logger.warning(f"Could not get size info for table {table_name}: {e}")
            size_info['error'] = str(e)
        
        return size_info