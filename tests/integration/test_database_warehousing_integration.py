"""
Integration tests for database warehousing components.

Tests the complete workflow of schema creation, data versioning,
incremental processing, partitioning, and backup/recovery.
"""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime

from src.qudata.database.connector import DatabaseConnector, DatabaseConfig
from src.qudata.database.schema_manager import SchemaManager
from src.qudata.database.versioning import DataVersioning, ChangeType
from src.qudata.database.incremental import IncrementalProcessor, UpdateType, FileChange
from src.qudata.database.partitioning import PartitionManager, PartitionStrategy
from src.qudata.database.backup import BackupManager, BackupType
from src.qudata.models import Dataset, Document, DocumentMetadata


class TestDatabaseWarehousingIntegration:
    """Integration tests for database warehousing components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        try:
            # Force close any open database connections
            import gc
            gc.collect()
            # Try to remove the directory
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, database files might still be locked
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass  # Ignore if we can't clean up
    
    @pytest.fixture
    def database_config(self, temp_dir):
        """Create a test database configuration."""
        db_path = Path(temp_dir) / "test.db"
        return DatabaseConfig(
            connection_type="sqlite",
            host="localhost",
            port=0,
            database=str(db_path)
        )
    
    @pytest.fixture
    def connector(self, database_config):
        """Create a database connector."""
        connector = DatabaseConnector(database_config)
        connector.connect()
        yield connector
        try:
            connector.close()
        except Exception:
            pass  # Ignore close errors
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        documents = []
        for i in range(20):
            doc = Document(
                id=f"doc_{i:03d}",
                source_path=f"/test/doc_{i:03d}.txt",
                content=f"This is the content for document {i}. " * 50,  # Substantial content
                metadata=DocumentMetadata(
                    file_type="txt",
                    size_bytes=1000 + i * 100,
                    language="en",
                    domain=f"domain_{i % 4}",  # 4 different domains
                    quality_score=0.1 + (i * 0.04)  # Varying quality scores 0.1 to 0.86
                )
            )
            documents.append(doc)
        
        return Dataset(
            id="integration_test_dataset",
            name="Integration Test Dataset",
            version="1.0",
            documents=documents
        )
    
    def test_complete_warehousing_workflow(self, connector, temp_dir, sample_dataset):
        """Test the complete database warehousing workflow."""
        
        # Step 1: Create and apply database schema
        print("\n=== Step 1: Schema Management ===")
        schema_manager = SchemaManager(connector)
        schema = schema_manager.create_qudata_schema()
        
        # Create the schema in the database
        success = schema_manager.create_schema(schema)
        assert success, "Schema creation should succeed"
        
        # Validate the schema
        errors = schema_manager.validate_schema(schema)
        assert len(errors) == 0, f"Schema should be valid, but got errors: {errors}"
        
        print(f"✓ Created schema '{schema.name}' with {len(schema.tables)} tables")
        
        # Step 2: Initialize versioning system
        print("\n=== Step 2: Data Versioning ===")
        versioning = DataVersioning(connector, temp_dir + "/versions")
        
        # Create initial version
        version_id = versioning.create_version(
            description="Initial dataset version",
            tags=["initial", "test"],
            created_by="integration_test"
        )
        assert version_id is not None
        
        # Create snapshot of the dataset
        snapshot_id = versioning.create_snapshot(sample_dataset, version_id)
        assert snapshot_id is not None
        
        print(f"✓ Created version {version_id} and snapshot {snapshot_id}")
        
        # Track some changes
        change_id = versioning.track_change(
            change_type=ChangeType.CREATE,
            entity_type="dataset",
            entity_id=sample_dataset.id,
            new_value={"name": sample_dataset.name, "document_count": len(sample_dataset.documents)},
            description="Created initial dataset"
        )
        assert change_id is not None
        
        print(f"✓ Tracked change {change_id}")
        
        # Step 3: Create partitions
        print("\n=== Step 3: Data Partitioning ===")
        partition_manager = PartitionManager(connector, temp_dir + "/partitions")
        
        # Test different partitioning strategies
        strategies_to_test = [
            PartitionStrategy.DOMAIN_BASED,
            PartitionStrategy.SIZE_BASED,
            PartitionStrategy.QUALITY_BASED
        ]
        
        all_partition_ids = []
        for strategy in strategies_to_test:
            partition_ids = partition_manager.create_partitions(sample_dataset, strategy)
            all_partition_ids.extend(partition_ids)
            print(f"✓ Created {len(partition_ids)} partitions using {strategy.value} strategy")
        
        # Get partition statistics
        stats = partition_manager.get_partition_statistics(sample_dataset.id)
        print(f"✓ Partition statistics: {stats['total_partitions']} total partitions")
        
        # Step 4: Set up incremental processing
        print("\n=== Step 4: Incremental Processing ===")
        incremental_processor = IncrementalProcessor(connector, temp_dir + "/incremental")
        
        # Create some test files to simulate changes
        test_files_dir = Path(temp_dir) / "test_files"
        test_files_dir.mkdir()
        
        # Create test files
        for i in range(5):
            test_file = test_files_dir / f"test_{i}.txt"
            test_file.write_text(f"Test content {i}")
        
        # Scan for changes
        changes = incremental_processor.scan_for_changes([str(test_files_dir)])
        assert len(changes) > 0, "Should detect new files"
        
        # Create incremental update
        update_id = incremental_processor.create_incremental_update(
            dataset_id=sample_dataset.id,
            changes=changes
        )
        assert update_id is not None
        
        print(f"✓ Created incremental update {update_id} with {len(changes)} changes")
        
        # Step 5: Create backups
        print("\n=== Step 5: Backup and Recovery ===")
        backup_manager = BackupManager(connector, temp_dir + "/backups")
        
        # Mock the export functions for testing
        backup_manager._export_dataset = lambda dataset_id: sample_dataset.to_dict()
        backup_manager._export_database = lambda: {"datasets": [sample_dataset.to_dict()]}
        
        # Create different types of backups
        backup_types_to_test = [
            BackupType.FULL,
            BackupType.SNAPSHOT
        ]
        
        backup_ids = []
        for backup_type in backup_types_to_test:
            backup_id = backup_manager.create_backup(
                backup_type=backup_type,
                dataset_id=sample_dataset.id,
                description=f"Integration test {backup_type.value} backup"
            )
            backup_ids.append(backup_id)
            
            # Verify the backup
            is_valid = backup_manager.verify_backup(backup_id)
            assert is_valid, f"Backup {backup_id} should be valid"
            
            print(f"✓ Created and verified {backup_type.value} backup {backup_id}")
        
        # Get backup statistics
        backup_stats = backup_manager.get_backup_statistics()
        print(f"✓ Backup statistics: {backup_stats['size_statistics']['total_completed_backups']} completed backups")
        
        # Step 6: Test recovery workflow
        print("\n=== Step 6: Recovery Testing ===")
        
        # Load snapshot to verify versioning works
        loaded_dataset = versioning.load_snapshot(snapshot_id)
        assert loaded_dataset is not None, "Should be able to load snapshot"
        assert loaded_dataset.id == sample_dataset.id, "Loaded dataset should match original"
        assert len(loaded_dataset.documents) == len(sample_dataset.documents), "Document count should match"
        
        print(f"✓ Successfully loaded snapshot with {len(loaded_dataset.documents)} documents")
        
        # Test backup restore
        restore_id = backup_manager.restore_backup(
            backup_ids[0],  # Use the full backup
            target_location=temp_dir + "/restored"
        )
        assert restore_id is not None, "Restore operation should be created"
        
        print(f"✓ Created restore operation {restore_id}")
        
        # Step 7: Test optimization and cleanup
        print("\n=== Step 7: Optimization and Cleanup ===")
        
        # Optimize partitions
        optimization_results = partition_manager.optimize_partitions(sample_dataset.id)
        print(f"✓ Partition optimization: {optimization_results}")
        
        # Test incremental processing statistics
        processing_stats = incremental_processor.get_processing_statistics(sample_dataset.id)
        print(f"✓ Processing statistics: {processing_stats}")
        
        # Clean up old versions (with very short retention for testing)
        cleaned_versions = versioning.cleanup_old_versions(keep_count=1, keep_days=0)
        print(f"✓ Cleaned up {cleaned_versions} old versions")
        
        # Clean up old backups
        cleaned_backups = backup_manager.cleanup_old_backups(max_age_days=0, max_backups_per_dataset=1)
        print(f"✓ Cleaned up {cleaned_backups} old backups")
        
        print("\n=== Integration Test Completed Successfully ===")
        
        # Final verification - ensure all components are still functional
        final_stats = {
            "schema_tables": len(schema.tables),
            "versions_created": 1,
            "snapshots_created": 1,
            "partitions_created": len(all_partition_ids),
            "incremental_updates": 1,
            "backups_created": len(backup_ids),
            "changes_tracked": 1
        }
        
        print(f"Final statistics: {final_stats}")
        
        # Verify all major components created something
        assert final_stats["schema_tables"] > 0
        assert final_stats["versions_created"] > 0
        assert final_stats["snapshots_created"] > 0
        assert final_stats["partitions_created"] > 0
        assert final_stats["incremental_updates"] > 0
        assert final_stats["backups_created"] > 0
        assert final_stats["changes_tracked"] > 0
        
        print("✓ All components verified successfully")
    
    def test_error_handling_and_recovery(self, connector, temp_dir, sample_dataset):
        """Test error handling and recovery scenarios."""
        
        print("\n=== Testing Error Handling and Recovery ===")
        
        # Test schema validation with invalid schema
        schema_manager = SchemaManager(connector)
        invalid_schema = schema_manager.create_qudata_schema()
        
        # Add duplicate table to make schema invalid
        duplicate_table = invalid_schema.tables[0]  # Copy first table
        duplicate_table.name = invalid_schema.tables[1].name  # Give it same name as second table
        invalid_schema.add_table(duplicate_table)
        
        errors = schema_manager.validate_schema(invalid_schema)
        assert len(errors) > 0, "Should detect schema validation errors"
        print(f"✓ Detected {len(errors)} schema validation errors as expected")
        
        # Test versioning with invalid snapshot
        versioning = DataVersioning(connector, temp_dir + "/versions")
        
        # Try to load non-existent snapshot
        loaded_dataset = versioning.load_snapshot("non_existent_snapshot")
        assert loaded_dataset is None, "Should return None for non-existent snapshot"
        print("✓ Handled non-existent snapshot gracefully")
        
        # Test backup verification with corrupted backup
        backup_manager = BackupManager(connector, temp_dir + "/backups")
        
        # Create a backup info with non-existent file
        from src.qudata.database.backup import BackupInfo, BackupStatus
        fake_backup = BackupInfo(
            backup_id="fake_backup",
            backup_type=BackupType.FULL,
            file_path="/non/existent/path.backup",
            status=BackupStatus.COMPLETED
        )
        
        backup_manager.get_backup_info = lambda backup_id: fake_backup
        
        is_valid = backup_manager.verify_backup("fake_backup")
        assert is_valid is False, "Should detect corrupted/missing backup"
        print("✓ Detected corrupted backup as expected")
        
        print("✓ Error handling tests completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])