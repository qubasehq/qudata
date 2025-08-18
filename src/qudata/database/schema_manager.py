"""
Schema manager for database schema design and management.

This module provides comprehensive database schema management capabilities
including schema creation, migration, validation, and optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Database column types."""
    TEXT = "TEXT"
    VARCHAR = "VARCHAR"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    BOOLEAN = "BOOLEAN"
    DATETIME = "DATETIME"
    TIMESTAMP = "TIMESTAMP"
    JSON = "JSON"
    BLOB = "BLOB"


class IndexType(Enum):
    """Database index types."""
    PRIMARY = "PRIMARY"
    UNIQUE = "UNIQUE"
    INDEX = "INDEX"
    FULLTEXT = "FULLTEXT"
    SPATIAL = "SPATIAL"


@dataclass
class ColumnDefinition:
    """Definition of a database column."""
    name: str
    column_type: ColumnType
    length: Optional[int] = None
    nullable: bool = True
    default_value: Optional[Any] = None
    auto_increment: bool = False
    comment: Optional[str] = None
    
    def to_sql(self, dialect: str = "postgresql") -> str:
        """Convert column definition to SQL."""
        sql_parts = [self.name]
        
        # Handle column type based on dialect
        if dialect == "postgresql":
            type_mapping = {
                ColumnType.TEXT: "TEXT",
                ColumnType.VARCHAR: f"VARCHAR({self.length or 255})",
                ColumnType.INTEGER: "INTEGER",
                ColumnType.BIGINT: "BIGINT",
                ColumnType.FLOAT: "REAL",
                ColumnType.DOUBLE: "DOUBLE PRECISION",
                ColumnType.BOOLEAN: "BOOLEAN",
                ColumnType.DATETIME: "TIMESTAMP",
                ColumnType.TIMESTAMP: "TIMESTAMP",
                ColumnType.JSON: "JSONB",
                ColumnType.BLOB: "BYTEA"
            }
        elif dialect == "mysql":
            type_mapping = {
                ColumnType.TEXT: "TEXT",
                ColumnType.VARCHAR: f"VARCHAR({self.length or 255})",
                ColumnType.INTEGER: "INT",
                ColumnType.BIGINT: "BIGINT",
                ColumnType.FLOAT: "FLOAT",
                ColumnType.DOUBLE: "DOUBLE",
                ColumnType.BOOLEAN: "BOOLEAN",
                ColumnType.DATETIME: "DATETIME",
                ColumnType.TIMESTAMP: "TIMESTAMP",
                ColumnType.JSON: "JSON",
                ColumnType.BLOB: "BLOB"
            }
        else:  # sqlite
            type_mapping = {
                ColumnType.TEXT: "TEXT",
                ColumnType.VARCHAR: "TEXT",
                ColumnType.INTEGER: "INTEGER",
                ColumnType.BIGINT: "INTEGER",
                ColumnType.FLOAT: "REAL",
                ColumnType.DOUBLE: "REAL",
                ColumnType.BOOLEAN: "INTEGER",
                ColumnType.DATETIME: "TEXT",
                ColumnType.TIMESTAMP: "TEXT",
                ColumnType.JSON: "TEXT",
                ColumnType.BLOB: "BLOB"
            }
        
        sql_parts.append(type_mapping[self.column_type])
        
        # Add constraints
        if not self.nullable:
            sql_parts.append("NOT NULL")
        
        if self.default_value is not None:
            if isinstance(self.default_value, str):
                sql_parts.append(f"DEFAULT '{self.default_value}'")
            else:
                sql_parts.append(f"DEFAULT {self.default_value}")
        
        if self.auto_increment:
            if dialect == "postgresql":
                sql_parts.append("GENERATED ALWAYS AS IDENTITY")
            elif dialect == "mysql":
                sql_parts.append("AUTO_INCREMENT")
        
        return " ".join(sql_parts)


@dataclass
class IndexDefinition:
    """Definition of a database index."""
    name: str
    table_name: str
    columns: List[str]
    index_type: IndexType = IndexType.INDEX
    unique: bool = False
    comment: Optional[str] = None
    
    def to_sql(self, dialect: str = "postgresql") -> str:
        """Convert index definition to SQL."""
        if self.index_type == IndexType.PRIMARY:
            return f"PRIMARY KEY ({', '.join(self.columns)})"
        
        sql_parts = ["CREATE"]
        
        if self.unique or self.index_type == IndexType.UNIQUE:
            sql_parts.append("UNIQUE")
        
        sql_parts.extend(["INDEX", self.name, "ON", self.table_name])
        sql_parts.append(f"({', '.join(self.columns)})")
        
        return " ".join(sql_parts)


@dataclass
class TableDefinition:
    """Definition of a database table."""
    name: str
    columns: List[ColumnDefinition] = field(default_factory=list)
    indexes: List[IndexDefinition] = field(default_factory=list)
    primary_key: Optional[List[str]] = None
    comment: Optional[str] = None
    
    def add_column(self, column: ColumnDefinition) -> None:
        """Add a column to the table."""
        self.columns.append(column)
    
    def add_index(self, index: IndexDefinition) -> None:
        """Add an index to the table."""
        self.indexes.append(index)
    
    def to_sql(self, dialect: str = "postgresql") -> str:
        """Convert table definition to SQL."""
        sql_parts = [f"CREATE TABLE {self.name} ("]
        
        # Add columns
        column_sql = [col.to_sql(dialect) for col in self.columns]
        
        # Add primary key
        if self.primary_key:
            column_sql.append(f"PRIMARY KEY ({', '.join(self.primary_key)})")
        
        sql_parts.append("  " + ",\n  ".join(column_sql))
        sql_parts.append(")")
        
        return "\n".join(sql_parts)


@dataclass
class SchemaDefinition:
    """Complete database schema definition."""
    name: str
    version: str
    tables: List[TableDefinition] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    
    def add_table(self, table: TableDefinition) -> None:
        """Add a table to the schema."""
        self.tables.append(table)
    
    def get_table(self, name: str) -> Optional[TableDefinition]:
        """Get a table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "tables": [
                {
                    "name": table.name,
                    "comment": table.comment,
                    "columns": [
                        {
                            "name": col.name,
                            "type": col.column_type.value,
                            "length": col.length,
                            "nullable": col.nullable,
                            "default_value": col.default_value,
                            "auto_increment": col.auto_increment,
                            "comment": col.comment
                        }
                        for col in table.columns
                    ],
                    "indexes": [
                        {
                            "name": idx.name,
                            "columns": idx.columns,
                            "type": idx.index_type.value,
                            "unique": idx.unique,
                            "comment": idx.comment
                        }
                        for idx in table.indexes
                    ],
                    "primary_key": table.primary_key
                }
                for table in self.tables
            ]
        }


class SchemaManager:
    """Manages database schema design, creation, and migration."""
    
    def __init__(self, database_connector, config: Dict[str, Any] = None):
        """
        Initialize schema manager.
        
        Args:
            database_connector: DatabaseConnector instance
            config: Configuration for schema management
        """
        self.connector = database_connector
        self.config = config or {}
        self.current_schema: Optional[SchemaDefinition] = None
        
    def create_qudata_schema(self) -> SchemaDefinition:
        """Create the standard QuData schema definition."""
        schema = SchemaDefinition(
            name="qudata",
            version="1.0",
            description="QuData LLM processing system database schema"
        )
        
        # Documents table
        documents_table = TableDefinition(
            name="documents",
            comment="Core documents table storing processed content"
        )
        
        documents_table.add_column(ColumnDefinition(
            name="id", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Unique document identifier"
        ))
        documents_table.add_column(ColumnDefinition(
            name="source_path", column_type=ColumnType.TEXT,
            nullable=False, comment="Original file path"
        ))
        documents_table.add_column(ColumnDefinition(
            name="content", column_type=ColumnType.TEXT,
            nullable=False, comment="Processed document content"
        ))
        documents_table.add_column(ColumnDefinition(
            name="file_type", column_type=ColumnType.VARCHAR, length=50,
            nullable=False, comment="Document file type"
        ))
        documents_table.add_column(ColumnDefinition(
            name="size_bytes", column_type=ColumnType.BIGINT,
            nullable=False, comment="Document size in bytes"
        ))
        documents_table.add_column(ColumnDefinition(
            name="language", column_type=ColumnType.VARCHAR, length=10,
            nullable=False, comment="Detected language code"
        ))
        documents_table.add_column(ColumnDefinition(
            name="domain", column_type=ColumnType.VARCHAR, length=100,
            nullable=False, default_value="uncategorized",
            comment="Document domain/category"
        ))
        documents_table.add_column(ColumnDefinition(
            name="quality_score", column_type=ColumnType.FLOAT,
            nullable=False, default_value=0.0,
            comment="Overall quality score"
        ))
        documents_table.add_column(ColumnDefinition(
            name="processing_timestamp", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="When document was processed"
        ))
        documents_table.add_column(ColumnDefinition(
            name="version", column_type=ColumnType.VARCHAR, length=20,
            nullable=False, default_value="1.0",
            comment="Document version"
        ))
        documents_table.add_column(ColumnDefinition(
            name="metadata", column_type=ColumnType.JSON,
            nullable=True, comment="Additional metadata as JSON"
        ))
        documents_table.add_column(ColumnDefinition(
            name="created_at", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="Record creation timestamp"
        ))
        documents_table.add_column(ColumnDefinition(
            name="updated_at", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="Record update timestamp"
        ))
        
        documents_table.primary_key = ["id"]
        documents_table.add_index(IndexDefinition(
            name="idx_documents_domain", table_name="documents",
            columns=["domain"], comment="Index on document domain"
        ))
        documents_table.add_index(IndexDefinition(
            name="idx_documents_language", table_name="documents",
            columns=["language"], comment="Index on document language"
        ))
        documents_table.add_index(IndexDefinition(
            name="idx_documents_quality", table_name="documents",
            columns=["quality_score"], comment="Index on quality score"
        ))
        
        schema.add_table(documents_table)
        
        # Document entities table
        entities_table = TableDefinition(
            name="document_entities",
            comment="Named entities extracted from documents"
        )
        
        entities_table.add_column(ColumnDefinition(
            name="id", column_type=ColumnType.BIGINT,
            nullable=False, auto_increment=True,
            comment="Auto-increment primary key"
        ))
        entities_table.add_column(ColumnDefinition(
            name="document_id", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Reference to documents.id"
        ))
        entities_table.add_column(ColumnDefinition(
            name="text", column_type=ColumnType.TEXT,
            nullable=False, comment="Entity text"
        ))
        entities_table.add_column(ColumnDefinition(
            name="label", column_type=ColumnType.VARCHAR, length=50,
            nullable=False, comment="Entity label/type"
        ))
        entities_table.add_column(ColumnDefinition(
            name="start_pos", column_type=ColumnType.INTEGER,
            nullable=False, comment="Start position in text"
        ))
        entities_table.add_column(ColumnDefinition(
            name="end_pos", column_type=ColumnType.INTEGER,
            nullable=False, comment="End position in text"
        ))
        entities_table.add_column(ColumnDefinition(
            name="confidence", column_type=ColumnType.FLOAT,
            nullable=False, default_value=0.0,
            comment="Entity confidence score"
        ))
        
        entities_table.primary_key = ["id"]
        entities_table.add_index(IndexDefinition(
            name="idx_entities_document", table_name="document_entities",
            columns=["document_id"], comment="Index on document ID"
        ))
        entities_table.add_index(IndexDefinition(
            name="idx_entities_label", table_name="document_entities",
            columns=["label"], comment="Index on entity label"
        ))
        
        schema.add_table(entities_table)
        
        # Datasets table
        datasets_table = TableDefinition(
            name="datasets",
            comment="Dataset definitions and metadata"
        )
        
        datasets_table.add_column(ColumnDefinition(
            name="id", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Unique dataset identifier"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="name", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Dataset name"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="version", column_type=ColumnType.VARCHAR, length=50,
            nullable=False, comment="Dataset version"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="description", column_type=ColumnType.TEXT,
            nullable=True, comment="Dataset description"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="metadata", column_type=ColumnType.JSON,
            nullable=True, comment="Dataset metadata as JSON"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="quality_metrics", column_type=ColumnType.JSON,
            nullable=True, comment="Quality metrics as JSON"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="created_at", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="Dataset creation timestamp"
        ))
        datasets_table.add_column(ColumnDefinition(
            name="updated_at", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="Dataset update timestamp"
        ))
        
        datasets_table.primary_key = ["id"]
        datasets_table.add_index(IndexDefinition(
            name="idx_datasets_name_version", table_name="datasets",
            columns=["name", "version"], unique=True,
            comment="Unique index on name and version"
        ))
        
        schema.add_table(datasets_table)
        
        # Dataset documents table (many-to-many relationship)
        dataset_docs_table = TableDefinition(
            name="dataset_documents",
            comment="Relationship between datasets and documents"
        )
        
        dataset_docs_table.add_column(ColumnDefinition(
            name="dataset_id", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Reference to datasets.id"
        ))
        dataset_docs_table.add_column(ColumnDefinition(
            name="document_id", column_type=ColumnType.VARCHAR, length=255,
            nullable=False, comment="Reference to documents.id"
        ))
        dataset_docs_table.add_column(ColumnDefinition(
            name="split_type", column_type=ColumnType.VARCHAR, length=20,
            nullable=True, comment="train/validation/test split"
        ))
        dataset_docs_table.add_column(ColumnDefinition(
            name="added_at", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="When document was added to dataset"
        ))
        
        dataset_docs_table.primary_key = ["dataset_id", "document_id"]
        dataset_docs_table.add_index(IndexDefinition(
            name="idx_dataset_docs_split", table_name="dataset_documents",
            columns=["dataset_id", "split_type"],
            comment="Index on dataset and split type"
        ))
        
        schema.add_table(dataset_docs_table)
        
        # Processing logs table
        logs_table = TableDefinition(
            name="processing_logs",
            comment="Processing pipeline execution logs"
        )
        
        logs_table.add_column(ColumnDefinition(
            name="id", column_type=ColumnType.BIGINT,
            nullable=False, auto_increment=True,
            comment="Auto-increment primary key"
        ))
        logs_table.add_column(ColumnDefinition(
            name="document_id", column_type=ColumnType.VARCHAR, length=255,
            nullable=True, comment="Document being processed"
        ))
        logs_table.add_column(ColumnDefinition(
            name="stage", column_type=ColumnType.VARCHAR, length=50,
            nullable=False, comment="Processing stage"
        ))
        logs_table.add_column(ColumnDefinition(
            name="level", column_type=ColumnType.VARCHAR, length=20,
            nullable=False, comment="Log level (INFO, WARNING, ERROR)"
        ))
        logs_table.add_column(ColumnDefinition(
            name="message", column_type=ColumnType.TEXT,
            nullable=False, comment="Log message"
        ))
        logs_table.add_column(ColumnDefinition(
            name="metadata", column_type=ColumnType.JSON,
            nullable=True, comment="Additional log metadata"
        ))
        logs_table.add_column(ColumnDefinition(
            name="timestamp", column_type=ColumnType.TIMESTAMP,
            nullable=False, comment="Log timestamp"
        ))
        
        logs_table.primary_key = ["id"]
        logs_table.add_index(IndexDefinition(
            name="idx_logs_timestamp", table_name="processing_logs",
            columns=["timestamp"], comment="Index on timestamp"
        ))
        logs_table.add_index(IndexDefinition(
            name="idx_logs_document_stage", table_name="processing_logs",
            columns=["document_id", "stage"],
            comment="Index on document and stage"
        ))
        
        schema.add_table(logs_table)
        
        self.current_schema = schema
        return schema
    
    def create_schema(self, schema: SchemaDefinition) -> bool:
        """
        Create database schema from definition.
        
        Args:
            schema: Schema definition to create
            
        Returns:
            True if schema was created successfully
        """
        try:
            dialect = self._get_dialect()
            
            with self.connector.get_connection() as conn:
                # Create tables
                for table in schema.tables:
                    table_sql = table.to_sql(dialect)
                    logger.info(f"Creating table {table.name}")
                    
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(table_sql))
                    
                    # Create indexes
                    for index in table.indexes:
                        if index.index_type != IndexType.PRIMARY:
                            index_sql = index.to_sql(dialect)
                            logger.info(f"Creating index {index.name}")
                            
                            import sqlalchemy
                            conn.execute(sqlalchemy.text(index_sql))
                
                # Commit transaction
                if hasattr(conn, 'commit'):
                    conn.commit()
            
            logger.info(f"Successfully created schema '{schema.name}' version {schema.version}")
            self.current_schema = schema
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def validate_schema(self, schema: SchemaDefinition) -> List[str]:
        """
        Validate schema definition for consistency and best practices.
        
        Args:
            schema: Schema definition to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for duplicate table names
        table_names = [table.name for table in schema.tables]
        if len(table_names) != len(set(table_names)):
            errors.append("Duplicate table names found")
        
        # Validate each table
        for table in schema.tables:
            # Check for duplicate column names
            column_names = [col.name for col in table.columns]
            if len(column_names) != len(set(column_names)):
                errors.append(f"Table {table.name}: Duplicate column names")
            
            # Check for primary key
            if not table.primary_key and not any(
                col.auto_increment for col in table.columns
            ):
                errors.append(f"Table {table.name}: No primary key defined")
            
            # Validate column definitions
            for col in table.columns:
                if col.column_type == ColumnType.VARCHAR and not col.length:
                    errors.append(
                        f"Table {table.name}, column {col.name}: "
                        "VARCHAR columns must specify length"
                    )
        
        return errors
    
    def get_current_schema(self) -> Optional[Dict[str, Any]]:
        """Get current database schema information."""
        try:
            return self.connector.get_schema_info()
        except Exception as e:
            logger.error(f"Failed to get current schema: {e}")
            return None
    
    def compare_schemas(self, schema1: SchemaDefinition, 
                       schema2: SchemaDefinition) -> Dict[str, Any]:
        """
        Compare two schema definitions.
        
        Args:
            schema1: First schema to compare
            schema2: Second schema to compare
            
        Returns:
            Dictionary containing differences
        """
        differences = {
            "added_tables": [],
            "removed_tables": [],
            "modified_tables": []
        }
        
        schema1_tables = {table.name: table for table in schema1.tables}
        schema2_tables = {table.name: table for table in schema2.tables}
        
        # Find added and removed tables
        differences["added_tables"] = [
            name for name in schema2_tables 
            if name not in schema1_tables
        ]
        differences["removed_tables"] = [
            name for name in schema1_tables 
            if name not in schema2_tables
        ]
        
        # Find modified tables
        for table_name in schema1_tables:
            if table_name in schema2_tables:
                table1 = schema1_tables[table_name]
                table2 = schema2_tables[table_name]
                
                # Compare columns
                cols1 = {col.name: col for col in table1.columns}
                cols2 = {col.name: col for col in table2.columns}
                
                if cols1 != cols2:
                    differences["modified_tables"].append({
                        "table": table_name,
                        "added_columns": [
                            name for name in cols2 if name not in cols1
                        ],
                        "removed_columns": [
                            name for name in cols1 if name not in cols2
                        ]
                    })
        
        return differences
    
    def drop_schema(self, schema_name: str = None) -> bool:
        """
        Drop database schema.
        
        Args:
            schema_name: Name of schema to drop (optional)
            
        Returns:
            True if schema was dropped successfully
        """
        try:
            dialect = self._get_dialect()
            
            with self.connector.get_connection() as conn:
                if self.current_schema:
                    # Drop tables in reverse order to handle dependencies
                    for table in reversed(self.current_schema.tables):
                        drop_sql = f"DROP TABLE IF EXISTS {table.name}"
                        logger.info(f"Dropping table {table.name}")
                        
                        if dialect == "postgresql":
                            import sqlalchemy
                            conn.execute(sqlalchemy.text(drop_sql))
                        else:
                            conn.execute(drop_sql)
                
                # Commit transaction
                if hasattr(conn, 'commit'):
                    conn.commit()
            
            logger.info("Successfully dropped schema")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop schema: {e}")
            return False
    
    def export_schema(self, file_path: str) -> bool:
        """
        Export schema definition to JSON file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export was successful
        """
        if not self.current_schema:
            logger.error("No current schema to export")
            return False
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.current_schema.to_dict(), f, indent=2)
            
            logger.info(f"Schema exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export schema: {e}")
            return False
    
    def import_schema(self, file_path: str) -> Optional[SchemaDefinition]:
        """
        Import schema definition from JSON file.
        
        Args:
            file_path: Path to import file
            
        Returns:
            SchemaDefinition if import was successful
        """
        try:
            with open(file_path, 'r') as f:
                schema_data = json.load(f)
            
            # Reconstruct schema from dictionary
            schema = SchemaDefinition(
                name=schema_data["name"],
                version=schema_data["version"],
                description=schema_data.get("description"),
                created_at=datetime.fromisoformat(schema_data["created_at"])
            )
            
            # Reconstruct tables
            for table_data in schema_data["tables"]:
                table = TableDefinition(
                    name=table_data["name"],
                    comment=table_data.get("comment"),
                    primary_key=table_data.get("primary_key")
                )
                
                # Reconstruct columns
                for col_data in table_data["columns"]:
                    column = ColumnDefinition(
                        name=col_data["name"],
                        column_type=ColumnType(col_data["type"]),
                        length=col_data.get("length"),
                        nullable=col_data.get("nullable", True),
                        default_value=col_data.get("default_value"),
                        auto_increment=col_data.get("auto_increment", False),
                        comment=col_data.get("comment")
                    )
                    table.add_column(column)
                
                # Reconstruct indexes
                for idx_data in table_data.get("indexes", []):
                    index = IndexDefinition(
                        name=idx_data["name"],
                        table_name=table.name,
                        columns=idx_data["columns"],
                        index_type=IndexType(idx_data["type"]),
                        unique=idx_data.get("unique", False),
                        comment=idx_data.get("comment")
                    )
                    table.add_index(index)
                
                schema.add_table(table)
            
            logger.info(f"Schema imported from {file_path}")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to import schema: {e}")
            return None
    
    def _get_dialect(self) -> str:
        """Get database dialect for SQL generation."""
        connection_type = self.connector.config.connection_type
        if connection_type == "postgresql":
            return "postgresql"
        elif connection_type == "mysql":
            return "mysql"
        else:
            return "sqlite"