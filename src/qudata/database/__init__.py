"""
Database connectivity and extraction module for QuData.

This module provides database connectivity for PostgreSQL, MySQL, MongoDB,
and Elasticsearch, along with SQL and NoSQL extractors, query builders,
and advanced database warehousing capabilities.
"""

from .connector import DatabaseConnector
from .query_builder import QueryBuilder
from .schema_manager import SchemaManager
from .versioning import DataVersioning
from .incremental import IncrementalProcessor
from .partitioning import PartitionManager
from .backup import BackupManager

# Import extractors conditionally to handle pandas dependency
try:
    from .sql_extractor import SQLExtractor
    from .nosql_extractor import NoSQLExtractor
    _HAS_EXTRACTORS = True
except ImportError:
    SQLExtractor = None
    NoSQLExtractor = None
    _HAS_EXTRACTORS = False

__all__ = [
    'DatabaseConnector',
    'QueryBuilder',
    'SchemaManager',
    'DataVersioning',
    'IncrementalProcessor',
    'PartitionManager',
    'BackupManager'
]

if _HAS_EXTRACTORS:
    __all__.extend(['SQLExtractor', 'NoSQLExtractor'])