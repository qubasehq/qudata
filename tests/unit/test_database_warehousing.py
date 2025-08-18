"""
Unit tests for database warehousing components.

Tests for SchemaManager, DataVersioning, IncrementalProcessor,
PartitionManager, and BackupManager.
"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.qudata.database.schema_manager import (
    SchemaManager, ColumnDefinition, ColumnType, IndexDefinition, 
    IndexType, TableDefinition, SchemaDefinition
)
from src.qudata.database.versioning import (
    DataVersioning, ChangeType, VersionStatus, ChangeRecord, VersionInfo
)
from src.qudata.database.incremental import (
    IncrementalProcessor, UpdateType, ProcessingMode, FileChange
)
from src.qudata.database.partitioning import (
    PartitionManager, PartitionStrategy, PartitionStatus, PartitionInfo, PartitionConfig
)
from src.qudata.database.backup import (
    BackupManager, BackupType, BackupStatus, CompressionType, BackupInfo
)
from src.qudata.models import Dataset, Document, DocumentMetadata


class TestSchemaManager:
    """Test cases for SchemaManager."""
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    @pytest.fixture
    def schema_manager(self, mock_connector):
        """Create a SchemaManager instance."""
        return SchemaManager(mock_connector)
    
    def test_column_definition_to_sql(self):
        """Test column definition SQL generation."""
        column = ColumnDefinition(
            name="test_column",
            column_type=ColumnType.VARCHAR,
            length=255,
            nullable=False,
            default_value="default"
        )
        
        sql = column.to_sql("postgresql")
        assert "test_column" in sql
        assert "VARCHAR(255)" in sql
        assert "NOT NULL" in sql
        assert "DEFAULT 'default'" in sql
    
    def test_table_definition_to_sql(self):
        """Test table definition SQL generation."""
        table = TableDefinition(name="test_table")
        
        # Add columns
        table.add_column(ColumnDefinition(
            name="id", column_type=ColumnType.INTEGER, nullable=False
        ))
        table.add_column(ColumnDefinition(
            name="name", column_type=ColumnType.VARCHAR, length=100
        ))
        
        table.primary_key = ["id"]
        
        sql = table.to_sql("postgresql")
        assert "CREATE TABLE test_table" in sql
        assert "id INTEGER NOT NULL" in sql
        assert "name VARCHAR(100)" in sql
        assert "PRIMARY KEY (id)" in sql
    
    def test_create_qudata_schema(self, schema_manager):
        """Test QuData schema creation."""
        schema = schema_manager.create_qudata_schema()
        
        assert schema.name == "qudata"
        assert schema.version == "1.0"
        assert len(schema.tables) > 0
        
        # Check for required tables
        table_names = [table.name for table in schema.tables]
        assert "documents" in table_names
        assert "datasets" in table_names
        assert "document_entities" in table_names
    
    def test_validate_schema(self, schema_manager):
        """Test schema validation."""
        schema = SchemaDefinition(name="test", version="1.0")
        
        # Add table with duplicate columns
        table = TableDefinition(name="test_table")
        table.add_column(ColumnDefinition(name="col1", column_type=ColumnType.TEXT))
        table.add_column(ColumnDefinition(name="col1", column_type=ColumnType.TEXT))  # Duplicate
        schema.add_table(table)
        
        errors = schema_manager.validate_schema(schema)
        assert len(errors) > 0
        assert any("Duplicate column names" in error for error in errors)
    
    def test_compare_schemas(self, schema_manager):
        """Test schema comparison."""
        schema1 = SchemaDefinition(name="schema1", version="1.0")
        schema2 = SchemaDefinition(name="schema2", version="1.0")
        
        # Add table to schema1
        table1 = TableDefinition(name="table1")
        table1.add_column(ColumnDefinition(name="col1", column_type=ColumnType.TEXT))
        schema1.add_table(table1)
        
        # Add different table to schema2
        table2 = TableDefinition(name="table2")
        table2.add_column(ColumnDefinition(name="col2", column_type=ColumnType.TEXT))
        schema2.add_table(table2)
        
        differences = schema_manager.compare_schemas(schema1, schema2)
        
        assert "table1" in differences["removed_tables"]
        assert "table2" in differences["added_tables"]


class TestDataVersioning:
    """Test cases for DataVersioning."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    @pytest.fixture
    def versioning(self, mock_connector, temp_dir):
        """Create a DataVersioning instance."""
        return DataVersioning(mock_connector, temp_dir)
    
    def test_create_version(self, versioning):
        """Test version creation."""
        version_id = versioning.create_version(
            description="Test version",
            tags=["test"],
            created_by="test_user"
        )
        
        assert version_id is not None
        assert version_id.startswith("v_")
        assert versioning.current_version == version_id
    
    def test_track_change(self, versioning):
        """Test change tracking."""
        # Create a version first
        version_id = versioning.create_version("Test version")
        
        change_id = versioning.track_change(
            change_type=ChangeType.CREATE,
            entity_type="document",
            entity_id="doc123",
            new_value={"content": "test content"},
            description="Created new document"
        )
        
        assert change_id is not None
        assert change_id.startswith("chg_")
        assert len(versioning.change_log) == 1
    
    def test_create_snapshot(self, versioning):
        """Test snapshot creation."""
        # Create test dataset
        dataset = Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0"
        )
        
        snapshot_id = versioning.create_snapshot(dataset)
        
        assert snapshot_id is not None
        assert snapshot_id.startswith("snap_")
        
        # Check if snapshot file was created
        snapshot_files = list(Path(versioning.storage_path).glob("*.json"))
        assert len(snapshot_files) > 0
    
    def test_load_snapshot(self, versioning):
        """Test snapshot loading."""
        # Create and save a snapshot
        dataset = Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0"
        )
        
        snapshot_id = versioning.create_snapshot(dataset)
        
        # Mock the database query for snapshot info
        versioning._get_snapshot_info = Mock(return_value={
            "data_path": str(versioning.storage_path / f"{snapshot_id}.json")
        })
        
        # Load the snapshot
        loaded_dataset = versioning.load_snapshot(snapshot_id)
        
        assert loaded_dataset is not None
        assert loaded_dataset.id == dataset.id
        assert loaded_dataset.name == dataset.name


class TestIncrementalProcessor:
    """Test cases for IncrementalProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    @pytest.fixture
    def processor(self, mock_connector, temp_dir):
        """Create an IncrementalProcessor instance."""
        return IncrementalProcessor(mock_connector, temp_dir)
    
    def test_file_change_creation(self):
        """Test FileChange object creation."""
        change = FileChange(
            file_path="/test/file.txt",
            change_type=UpdateType.ADD,
            new_hash="abc123",
            new_size=1024
        )
        
        assert change.file_path == "/test/file.txt"
        assert change.change_type == UpdateType.ADD
        assert change.new_hash == "abc123"
        assert change.new_size == 1024
    
    def test_scan_for_changes(self, processor, temp_dir):
        """Test scanning for file changes."""
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        changes = processor.scan_for_changes([temp_dir])
        
        assert len(changes) >= 1
        assert any(change.change_type == UpdateType.ADD for change in changes)
    
    def test_create_incremental_update(self, processor):
        """Test incremental update creation."""
        changes = [
            FileChange(
                file_path="/test/file1.txt",
                change_type=UpdateType.ADD,
                new_hash="hash1",
                new_size=100
            ),
            FileChange(
                file_path="/test/file2.txt",
                change_type=UpdateType.UPDATE,
                old_hash="old_hash",
                new_hash="new_hash",
                old_size=50,
                new_size=75
            )
        ]
        
        update_id = processor.create_incremental_update(
            dataset_id="test_dataset",
            changes=changes
        )
        
        assert update_id is not None
        assert update_id.startswith("inc_")
    
    def test_processing_mode_determination(self, processor):
        """Test processing mode determination."""
        # Test with mostly new files (should be incremental)
        changes = [
            FileChange(file_path=f"/test/file{i}.txt", change_type=UpdateType.ADD)
            for i in range(10)
        ]
        
        mode = processor._determine_processing_mode(changes)
        assert mode == ProcessingMode.INCREMENTAL
        
        # Test with mostly updated files (might trigger full reprocessing)
        changes = [
            FileChange(file_path=f"/test/file{i}.txt", change_type=UpdateType.UPDATE)
            for i in range(10)
        ]
        
        mode = processor._determine_processing_mode(changes)
        # Mode depends on configuration thresholds


class TestPartitionManager:
    """Test cases for PartitionManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    @pytest.fixture
    def partition_manager(self, mock_connector, temp_dir):
        """Create a PartitionManager instance."""
        return PartitionManager(mock_connector, temp_dir)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        documents = []
        for i in range(10):
            doc = Document(
                id=f"doc_{i}",
                source_path=f"/test/doc_{i}.txt",
                content=f"Content for document {i}" * 100,  # Make content substantial
                metadata=DocumentMetadata(
                    file_type="txt",
                    size_bytes=1000 + i * 100,
                    language="en",
                    domain=f"domain_{i % 3}",  # 3 different domains
                    quality_score=0.1 + (i * 0.1)  # Varying quality scores
                )
            )
            documents.append(doc)
        
        return Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0",
            documents=documents
        )
    
    def test_partition_info_creation(self):
        """Test PartitionInfo object creation."""
        partition_info = PartitionInfo(
            partition_id="test_partition",
            dataset_id="test_dataset",
            strategy=PartitionStrategy.SIZE_BASED,
            partition_key="size_partition_1",
            document_count=100,
            size_bytes=1024000
        )
        
        assert partition_info.partition_id == "test_partition"
        assert partition_info.strategy == PartitionStrategy.SIZE_BASED
        assert partition_info.document_count == 100
        assert partition_info.size_bytes == 1024000
    
    def test_create_size_based_partitions(self, partition_manager, sample_dataset):
        """Test size-based partitioning."""
        config = PartitionConfig(
            strategy=PartitionStrategy.SIZE_BASED,
            max_partition_size_bytes=5000  # Small size to force multiple partitions
        )
        
        partitions = partition_manager._create_size_based_partitions(sample_dataset, config)
        
        assert len(partitions) > 1  # Should create multiple partitions
        
        # Check that each partition respects size limits
        for partition_info, doc_ids in partitions:
            assert partition_info.size_bytes <= config.max_partition_size_bytes or len(doc_ids) == 1
    
    def test_create_count_based_partitions(self, partition_manager, sample_dataset):
        """Test count-based partitioning."""
        config = PartitionConfig(
            strategy=PartitionStrategy.COUNT_BASED,
            max_partition_count=3  # Small count to force multiple partitions
        )
        
        partitions = partition_manager._create_count_based_partitions(sample_dataset, config)
        
        assert len(partitions) > 1  # Should create multiple partitions
        
        # Check that each partition respects count limits
        for partition_info, doc_ids in partitions:
            assert len(doc_ids) <= config.max_partition_count
    
    def test_create_domain_based_partitions(self, partition_manager, sample_dataset):
        """Test domain-based partitioning."""
        config = PartitionConfig(strategy=PartitionStrategy.DOMAIN_BASED)
        
        partitions = partition_manager._create_domain_based_partitions(sample_dataset, config)
        
        # Should create 3 partitions (one for each domain)
        assert len(partitions) == 3
        
        # Check that documents are grouped by domain
        for partition_info, doc_ids in partitions:
            assert partition_info.partition_key.startswith("domain_")
    
    def test_create_quality_based_partitions(self, partition_manager, sample_dataset):
        """Test quality-based partitioning."""
        config = PartitionConfig(
            strategy=PartitionStrategy.QUALITY_BASED,
            metadata={"quality_thresholds": [0.3, 0.6, 0.8]}
        )
        
        partitions = partition_manager._create_quality_based_partitions(sample_dataset, config)
        
        assert len(partitions) > 0
        
        # Check that partitions are created based on quality ranges
        partition_keys = [partition_info.partition_key for partition_info, _ in partitions]
        assert any("quality" in key for key in partition_keys)
    
    def test_partition_config(self):
        """Test partition configuration."""
        config = PartitionConfig(
            strategy=PartitionStrategy.SIZE_BASED,
            max_partition_size_bytes=1000000,
            max_partition_count=500,
            auto_merge=True,
            auto_split=True
        )
        
        assert config.strategy == PartitionStrategy.SIZE_BASED
        assert config.max_partition_size_bytes == 1000000
        assert config.auto_merge is True
        assert config.auto_split is True


class TestBackupManager:
    """Test cases for BackupManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    @pytest.fixture
    def backup_manager(self, mock_connector, temp_dir):
        """Create a BackupManager instance."""
        return BackupManager(mock_connector, temp_dir)
    
    def test_backup_info_creation(self):
        """Test BackupInfo object creation."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            dataset_id="test_dataset",
            compression_type=CompressionType.GZIP,
            description="Test backup"
        )
        
        assert backup_info.backup_id == "test_backup"
        assert backup_info.backup_type == BackupType.FULL
        assert backup_info.dataset_id == "test_dataset"
        assert backup_info.compression_type == CompressionType.GZIP
        assert backup_info.status == BackupStatus.PENDING
    
    def test_generate_backup_id(self, backup_manager):
        """Test backup ID generation."""
        backup_id = backup_manager._generate_backup_id()
        
        assert backup_id.startswith("backup_")
        assert len(backup_id) > 20  # Should include timestamp and hash
    
    def test_calculate_file_checksum(self, backup_manager, temp_dir):
        """Test file checksum calculation."""
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "test content for checksum"
        test_file.write_text(test_content)
        
        checksum = backup_manager._calculate_file_checksum(str(test_file))
        
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 hash length
        
        # Calculate again to ensure consistency
        checksum2 = backup_manager._calculate_file_checksum(str(test_file))
        assert checksum == checksum2
    
    @patch('src.qudata.database.backup.BackupManager._export_dataset')
    def test_create_full_backup(self, mock_export, backup_manager):
        """Test full backup creation."""
        # Mock the export function
        mock_export.return_value = {"test": "data"}
        
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            dataset_id="test_dataset",
            compression_type=CompressionType.NONE  # Use no compression for testing
        )
        
        success = backup_manager._create_full_backup(backup_info)
        
        assert success is True
        assert backup_info.file_path is not None
        assert backup_info.size_bytes > 0
        assert backup_info.checksum is not None
        
        # Check that backup file was created
        backup_file = Path(backup_info.file_path)
        assert backup_file.exists()
    
    def test_verify_backup(self, backup_manager, temp_dir):
        """Test backup verification."""
        # Create a test backup file
        backup_file = Path(temp_dir) / "test_backup.json"
        test_data = {"test": "backup data"}
        backup_file.write_text(json.dumps(test_data))
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            file_path=str(backup_file),
            compression_type=CompressionType.NONE,
            checksum=backup_manager._calculate_file_checksum(str(backup_file)),
            status=BackupStatus.COMPLETED
        )
        
        # Mock get_backup_info to return our test backup
        backup_manager.get_backup_info = Mock(return_value=backup_info)
        
        # Verify the backup
        is_valid = backup_manager.verify_backup("test_backup")
        
        assert is_valid is True
    
    def test_verify_corrupted_backup(self, backup_manager, temp_dir):
        """Test verification of corrupted backup."""
        # Create a test backup file
        backup_file = Path(temp_dir) / "test_backup.json"
        backup_file.write_text("invalid json content")
        
        # Create backup info with wrong checksum
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            file_path=str(backup_file),
            compression_type=CompressionType.NONE,
            checksum="wrong_checksum",
            status=BackupStatus.COMPLETED
        )
        
        # Mock get_backup_info to return our test backup
        backup_manager.get_backup_info = Mock(return_value=backup_info)
        
        # Verify the backup (should fail due to wrong checksum)
        is_valid = backup_manager.verify_backup("test_backup")
        
        assert is_valid is False


class TestIntegration:
    """Integration tests for database warehousing components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = Mock()
        connector.config.connection_type = "postgresql"
        
        # Mock connection context manager
        mock_conn = Mock()
        mock_conn.execute = Mock()
        mock_conn.commit = Mock()
        
        connector.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
        connector.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        return connector
    
    def test_schema_and_versioning_integration(self, mock_connector, temp_dir):
        """Test integration between SchemaManager and DataVersioning."""
        # Create schema manager and create schema
        schema_manager = SchemaManager(mock_connector)
        schema = schema_manager.create_qudata_schema()
        
        # Create versioning system
        versioning = DataVersioning(mock_connector, temp_dir)
        
        # Create a version for the schema
        version_id = versioning.create_version(
            description=f"Schema {schema.name} version {schema.version}",
            tags=["schema", "initial"]
        )
        
        assert version_id is not None
        assert versioning.current_version == version_id
    
    def test_partitioning_and_backup_integration(self, mock_connector, temp_dir):
        """Test integration between PartitionManager and BackupManager."""
        # Create sample dataset
        documents = [
            Document(
                id=f"doc_{i}",
                source_path=f"/test/doc_{i}.txt",
                content=f"Content {i}",
                metadata=DocumentMetadata(
                    file_type="txt",
                    size_bytes=100,
                    language="en"
                )
            )
            for i in range(5)
        ]
        
        dataset = Dataset(
            id="test_dataset",
            name="Test Dataset",
            version="1.0",
            documents=documents
        )
        
        # Create partitions
        partition_manager = PartitionManager(mock_connector, temp_dir)
        partition_ids = partition_manager.create_partitions(
            dataset, PartitionStrategy.COUNT_BASED
        )
        
        # Create backup of partitioned dataset
        backup_manager = BackupManager(mock_connector, temp_dir)
        
        # Mock the export function for backup
        backup_manager._export_dataset = Mock(return_value=dataset.to_dict())
        
        backup_id = backup_manager.create_backup(
            BackupType.FULL,
            dataset_id=dataset.id,
            description="Backup of partitioned dataset"
        )
        
        assert len(partition_ids) > 0
        assert backup_id is not None


if __name__ == "__main__":
    pytest.main([__file__])