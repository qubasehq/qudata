"""
Backup manager for data recovery and disaster protection.

This module provides comprehensive backup and recovery capabilities
including automated backups, incremental backups, and disaster recovery.
"""

import logging
import shutil
import gzip
import json
import tarfile
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import threading
import time

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Status of a backup operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    RESTORED = "restored"


class CompressionType(Enum):
    """Compression types for backups."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class BackupInfo:
    """Information about a backup."""
    backup_id: str
    backup_type: BackupType
    dataset_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    file_path: Optional[str] = None
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_type: CompressionType = CompressionType.GZIP
    checksum: Optional[str] = None
    parent_backup_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backup info to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_type": self.compression_type.value,
            "checksum": self.checksum,
            "parent_backup_id": self.parent_backup_id,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class RestoreInfo:
    """Information about a restore operation."""
    restore_id: str
    backup_id: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"
    target_location: Optional[str] = None
    restored_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert restore info to dictionary."""
        return {
            "restore_id": self.restore_id,
            "backup_id": self.backup_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "target_location": self.target_location,
            "restored_files": self.restored_files,
            "errors": self.errors,
            "metadata": self.metadata
        }


class BackupManager:
    """Manages backup and recovery operations for data protection."""
    
    def __init__(self, database_connector, backup_path: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize backup manager.
        
        Args:
            database_connector: DatabaseConnector instance
            backup_path: Path for storing backups
            config: Configuration for backup operations
        """
        self.connector = database_connector
        self.backup_path = Path(backup_path or "data/backups")
        self.config = config or {}
        
        # Configuration options
        self.default_compression = CompressionType(
            self.config.get("default_compression", "gzip")
        )
        self.max_backup_age_days = self.config.get("max_backup_age_days", 90)
        self.max_backups_per_dataset = self.config.get("max_backups_per_dataset", 10)
        self.backup_schedule_hours = self.config.get("backup_schedule_hours", 24)
        self.verify_backups = self.config.get("verify_backups", True)
        self.parallel_compression = self.config.get("parallel_compression", True)
        
        # Ensure backup directory exists
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize backup tables
        self._initialize_backup_tables()
        
        # Background backup scheduler
        self._scheduler_running = False
        self._scheduler_thread = None
    
    def _initialize_backup_tables(self) -> None:
        """Initialize database tables for backup management."""
        try:
            with self.connector.get_connection() as conn:
                # Backups table
                backups_sql = """
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id VARCHAR(255) PRIMARY KEY,
                    backup_type VARCHAR(20) NOT NULL,
                    dataset_id VARCHAR(255),
                    created_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    file_path VARCHAR(500),
                    size_bytes BIGINT DEFAULT 0,
                    compressed_size_bytes BIGINT DEFAULT 0,
                    compression_type VARCHAR(20) DEFAULT 'gzip',
                    checksum VARCHAR(64),
                    parent_backup_id VARCHAR(255),
                    description TEXT,
                    metadata TEXT
                )
                """
                
                # Restore operations table
                restores_sql = """
                CREATE TABLE IF NOT EXISTS restore_operations (
                    restore_id VARCHAR(255) PRIMARY KEY,
                    backup_id VARCHAR(255) NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    target_location VARCHAR(500),
                    restored_files TEXT,
                    errors TEXT,
                    metadata TEXT
                )
                """
                
                # Backup schedule table
                schedule_sql = """
                CREATE TABLE IF NOT EXISTS backup_schedule (
                    schedule_id VARCHAR(255) PRIMARY KEY,
                    dataset_id VARCHAR(255),
                    backup_type VARCHAR(20) NOT NULL,
                    schedule_expression VARCHAR(100) NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    last_run TIMESTAMP,
                    next_run TIMESTAMP,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(backups_sql))
                    conn.execute(sqlalchemy.text(restores_sql))
                    conn.execute(sqlalchemy.text(schedule_sql))
                else:
                    conn.execute(backups_sql)
                    conn.execute(restores_sql)
                    conn.execute(schedule_sql)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to initialize backup tables: {e}")
    
    def create_backup(self, backup_type: BackupType, dataset_id: str = None,
                     compression: CompressionType = None, 
                     description: str = None) -> str:
        """
        Create a backup.
        
        Args:
            backup_type: Type of backup to create
            dataset_id: ID of dataset to backup (None for full system backup)
            compression: Compression type to use
            description: Description of the backup
            
        Returns:
            Backup ID
        """
        backup_id = self._generate_backup_id()
        
        if compression is None:
            compression = self.default_compression
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=backup_type,
            dataset_id=dataset_id,
            compression_type=compression,
            description=description
        )
        
        logger.info(f"Creating {backup_type.value} backup {backup_id}")
        
        try:
            # Store backup info
            self._store_backup_info(backup_info)
            
            # Update status to in progress
            self._update_backup_status(backup_id, BackupStatus.IN_PROGRESS)
            
            # Perform backup based on type
            if backup_type == BackupType.FULL:
                success = self._create_full_backup(backup_info)
            elif backup_type == BackupType.INCREMENTAL:
                success = self._create_incremental_backup(backup_info)
            elif backup_type == BackupType.DIFFERENTIAL:
                success = self._create_differential_backup(backup_info)
            elif backup_type == BackupType.SNAPSHOT:
                success = self._create_snapshot_backup(backup_info)
            else:
                raise ValueError(f"Unsupported backup type: {backup_type}")
            
            # Update final status
            final_status = BackupStatus.COMPLETED if success else BackupStatus.FAILED
            self._update_backup_status(backup_id, final_status)
            
            if success:
                logger.info(f"Successfully created backup {backup_id}")
            else:
                logger.error(f"Failed to create backup {backup_id}")
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            self._update_backup_status(backup_id, BackupStatus.FAILED)
            raise
    
    def restore_backup(self, backup_id: str, target_location: str = None,
                      selective_restore: List[str] = None) -> str:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of the backup to restore
            target_location: Target location for restore (uses original if None)
            selective_restore: List of specific files/tables to restore
            
        Returns:
            Restore operation ID
        """
        restore_id = self._generate_restore_id()
        
        logger.info(f"Starting restore operation {restore_id} from backup {backup_id}")
        
        try:
            # Get backup info
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                raise ValueError(f"Backup {backup_id} not found")
            
            if backup_info.status != BackupStatus.COMPLETED:
                raise ValueError(f"Backup {backup_id} is not in completed state")
            
            # Create restore info
            restore_info = RestoreInfo(
                restore_id=restore_id,
                backup_id=backup_id,
                target_location=target_location,
                metadata={"selective_restore": selective_restore}
            )
            
            # Store restore info
            self._store_restore_info(restore_info)
            
            # Perform restore
            success = self._perform_restore(backup_info, restore_info, selective_restore)
            
            # Update final status
            final_status = "completed" if success else "failed"
            self._update_restore_status(restore_id, final_status)
            
            if success:
                logger.info(f"Successfully completed restore operation {restore_id}")
            else:
                logger.error(f"Failed restore operation {restore_id}")
            
            return restore_id
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            self._update_restore_status(restore_id, "failed")
            raise
    
    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_id: ID of the backup to verify
            
        Returns:
            True if backup is valid
        """
        try:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            if not backup_info.file_path or not Path(backup_info.file_path).exists():
                logger.error(f"Backup file not found: {backup_info.file_path}")
                self._update_backup_status(backup_id, BackupStatus.CORRUPTED)
                return False
            
            # Verify checksum if available
            if backup_info.checksum:
                calculated_checksum = self._calculate_file_checksum(backup_info.file_path)
                if calculated_checksum != backup_info.checksum:
                    logger.error(f"Checksum mismatch for backup {backup_id}")
                    self._update_backup_status(backup_id, BackupStatus.CORRUPTED)
                    return False
            
            # Try to read backup file
            try:
                if backup_info.compression_type == CompressionType.GZIP:
                    with gzip.open(backup_info.file_path, 'rt') as f:
                        # Try to read first few lines
                        for _ in range(10):
                            line = f.readline()
                            if not line:
                                break
                elif backup_info.compression_type == CompressionType.NONE:
                    with open(backup_info.file_path, 'r') as f:
                        # Try to read first few lines
                        for _ in range(10):
                            line = f.readline()
                            if not line:
                                break
                else:
                    # For other compression types, just check if file can be opened
                    with tarfile.open(backup_info.file_path, 'r') as tar:
                        tar.getnames()[:10]  # Get first 10 file names
                        
            except Exception as e:
                logger.error(f"Failed to read backup file {backup_info.file_path}: {e}")
                self._update_backup_status(backup_id, BackupStatus.CORRUPTED)
                return False
            
            logger.info(f"Backup {backup_id} verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify backup {backup_id}: {e}")
            return False
    
    def list_backups(self, dataset_id: str = None, 
                    backup_type: BackupType = None,
                    status: BackupStatus = None) -> List[BackupInfo]:
        """
        List available backups.
        
        Args:
            dataset_id: Filter by dataset ID
            backup_type: Filter by backup type
            status: Filter by status
            
        Returns:
            List of backup information
        """
        try:
            with self.connector.get_connection() as conn:
                query_parts = ["SELECT * FROM backups WHERE 1=1"]
                params = []
                
                if dataset_id:
                    query_parts.append("AND dataset_id = ?")
                    params.append(dataset_id)
                
                if backup_type:
                    query_parts.append("AND backup_type = ?")
                    params.append(backup_type.value)
                
                if status:
                    query_parts.append("AND status = ?")
                    params.append(status.value)
                
                query_parts.append("ORDER BY created_at DESC")
                query = " ".join(query_parts)
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), params)
                
                backups = []
                for row in result:
                    backup_info = self._row_to_backup_info(row)
                    backups.append(backup_info)
                
                return backups
                
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """
        Get information about a specific backup.
        
        Args:
            backup_id: ID of the backup
            
        Returns:
            BackupInfo object if found
        """
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM backups WHERE backup_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (backup_id,))
                
                row = result.fetchone()
                if row:
                    return self._row_to_backup_info(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get backup info for {backup_id}: {e}")
            return None
    
    def cleanup_old_backups(self, max_age_days: int = None,
                           max_backups_per_dataset: int = None) -> int:
        """
        Clean up old backups.
        
        Args:
            max_age_days: Maximum age in days (uses config default if None)
            max_backups_per_dataset: Maximum backups per dataset (uses config default if None)
            
        Returns:
            Number of backups cleaned up
        """
        if max_age_days is None:
            max_age_days = self.max_backup_age_days
        
        if max_backups_per_dataset is None:
            max_backups_per_dataset = self.max_backups_per_dataset
        
        logger.info(f"Cleaning up backups older than {max_age_days} days")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0
            
            # Get old backups
            with self.connector.get_connection() as conn:
                query = """
                SELECT backup_id FROM backups 
                WHERE created_at < ? AND status = 'completed'
                ORDER BY created_at
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query), (cutoff_date.isoformat(),))
                else:
                    result = conn.execute(query, (cutoff_date.isoformat(),))
                
                old_backups = [row[0] for row in result]
            
            # Delete old backups
            for backup_id in old_backups:
                if self.delete_backup(backup_id):
                    deleted_count += 1
            
            # Clean up excess backups per dataset
            datasets = self._get_datasets_with_backups()
            for dataset_id in datasets:
                dataset_backups = self.list_backups(dataset_id=dataset_id, status=BackupStatus.COMPLETED)
                
                if len(dataset_backups) > max_backups_per_dataset:
                    # Sort by creation date and keep only the most recent
                    dataset_backups.sort(key=lambda b: b.created_at, reverse=True)
                    excess_backups = dataset_backups[max_backups_per_dataset:]
                    
                    for backup in excess_backups:
                        if self.delete_backup(backup.backup_id):
                            deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of the backup to delete
            
        Returns:
            True if backup was deleted successfully
        """
        try:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                logger.warning(f"Backup {backup_id} not found")
                return False
            
            # Delete backup file
            if backup_info.file_path:
                backup_file = Path(backup_info.file_path)
                if backup_file.exists():
                    backup_file.unlink()
                    logger.info(f"Deleted backup file: {backup_info.file_path}")
            
            # Delete from database
            with self.connector.get_connection() as conn:
                # Delete restore operations for this backup
                conn.execute("DELETE FROM restore_operations WHERE backup_id = ?", (backup_id,))
                
                # Delete backup record
                conn.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id,))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
            
            logger.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """
        Get backup statistics.
        
        Returns:
            Dictionary containing backup statistics
        """
        try:
            with self.connector.get_connection() as conn:
                # Get backup counts by status
                status_query = """
                SELECT status, COUNT(*) as count
                FROM backups
                GROUP BY status
                """
                
                # Get backup counts by type
                type_query = """
                SELECT backup_type, COUNT(*) as count
                FROM backups
                GROUP BY backup_type
                """
                
                # Get total backup size
                size_query = """
                SELECT 
                    SUM(size_bytes) as total_size,
                    SUM(compressed_size_bytes) as total_compressed_size,
                    AVG(size_bytes) as avg_size,
                    COUNT(*) as total_backups
                FROM backups
                WHERE status = 'completed'
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    status_result = conn.execute(sqlalchemy.text(status_query))
                    type_result = conn.execute(sqlalchemy.text(type_query))
                    size_result = conn.execute(sqlalchemy.text(size_query))
                else:
                    status_result = conn.execute(status_query)
                    type_result = conn.execute(type_query)
                    size_result = conn.execute(size_query)
                
                # Aggregate results
                status_counts = {row[0]: row[1] for row in status_result}
                type_counts = {row[0]: row[1] for row in type_result}
                
                size_row = size_result.fetchone()
                size_stats = {
                    "total_size_bytes": size_row[0] or 0,
                    "total_compressed_size_bytes": size_row[1] or 0,
                    "average_size_bytes": size_row[2] or 0,
                    "total_completed_backups": size_row[3] or 0
                }
                
                return {
                    "status_distribution": status_counts,
                    "type_distribution": type_counts,
                    "size_statistics": size_stats,
                    "backup_directory": str(self.backup_path),
                    "available_disk_space": shutil.disk_usage(self.backup_path).free
                }
                
        except Exception as e:
            logger.error(f"Failed to get backup statistics: {e}")
            return {}
    
    def start_backup_scheduler(self) -> None:
        """Start the automatic backup scheduler."""
        if self._scheduler_running:
            logger.warning("Backup scheduler is already running")
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._backup_scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Started backup scheduler")
    
    def stop_backup_scheduler(self) -> None:
        """Stop the automatic backup scheduler."""
        if not self._scheduler_running:
            return
        
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("Stopped backup scheduler")
    
    def _create_full_backup(self, backup_info: BackupInfo) -> bool:
        """Create a full backup."""
        try:
            backup_file = self.backup_path / f"{backup_info.backup_id}_full.backup"
            
            # Export data based on dataset_id
            if backup_info.dataset_id:
                # Backup specific dataset
                data = self._export_dataset(backup_info.dataset_id)
            else:
                # Backup entire database
                data = self._export_database()
            
            # Write backup file
            if backup_info.compression_type == CompressionType.GZIP:
                with gzip.open(backup_file, 'wt') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(backup_file, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Calculate sizes and checksum
            backup_info.file_path = str(backup_file)
            backup_info.size_bytes = len(json.dumps(data).encode('utf-8'))
            backup_info.compressed_size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_file_checksum(str(backup_file))
            backup_info.completed_at = datetime.now()
            
            # Update backup info in database
            self._update_backup_info(backup_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create full backup: {e}")
            return False
    
    def _create_incremental_backup(self, backup_info: BackupInfo) -> bool:
        """Create an incremental backup."""
        try:
            # Find the last full or incremental backup
            parent_backup = self._find_parent_backup(backup_info.dataset_id, BackupType.INCREMENTAL)
            if not parent_backup:
                logger.info("No parent backup found, creating full backup instead")
                return self._create_full_backup(backup_info)
            
            backup_info.parent_backup_id = parent_backup.backup_id
            
            # Get changes since parent backup
            changes = self._get_changes_since_backup(parent_backup)
            
            backup_file = self.backup_path / f"{backup_info.backup_id}_incremental.backup"
            
            backup_data = {
                "backup_type": "incremental",
                "parent_backup_id": parent_backup.backup_id,
                "changes": changes,
                "timestamp": datetime.now().isoformat()
            }
            
            # Write backup file
            if backup_info.compression_type == CompressionType.GZIP:
                with gzip.open(backup_file, 'wt') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            else:
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Update backup info
            backup_info.file_path = str(backup_file)
            backup_info.size_bytes = len(json.dumps(backup_data).encode('utf-8'))
            backup_info.compressed_size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_file_checksum(str(backup_file))
            backup_info.completed_at = datetime.now()
            
            self._update_backup_info(backup_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create incremental backup: {e}")
            return False
    
    def _create_differential_backup(self, backup_info: BackupInfo) -> bool:
        """Create a differential backup."""
        try:
            # Find the last full backup
            parent_backup = self._find_parent_backup(backup_info.dataset_id, BackupType.FULL)
            if not parent_backup:
                logger.info("No full backup found, creating full backup instead")
                return self._create_full_backup(backup_info)
            
            backup_info.parent_backup_id = parent_backup.backup_id
            
            # Get changes since last full backup
            changes = self._get_changes_since_backup(parent_backup)
            
            backup_file = self.backup_path / f"{backup_info.backup_id}_differential.backup"
            
            backup_data = {
                "backup_type": "differential",
                "parent_backup_id": parent_backup.backup_id,
                "changes": changes,
                "timestamp": datetime.now().isoformat()
            }
            
            # Write backup file
            if backup_info.compression_type == CompressionType.GZIP:
                with gzip.open(backup_file, 'wt') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            else:
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Update backup info
            backup_info.file_path = str(backup_file)
            backup_info.size_bytes = len(json.dumps(backup_data).encode('utf-8'))
            backup_info.compressed_size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_file_checksum(str(backup_file))
            backup_info.completed_at = datetime.now()
            
            self._update_backup_info(backup_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create differential backup: {e}")
            return False
    
    def _create_snapshot_backup(self, backup_info: BackupInfo) -> bool:
        """Create a snapshot backup."""
        try:
            backup_file = self.backup_path / f"{backup_info.backup_id}_snapshot.tar.gz"
            
            # Create tar archive with compression
            with tarfile.open(backup_file, 'w:gz') as tar:
                # Add database files
                if backup_info.dataset_id:
                    # Add specific dataset files
                    dataset_files = self._get_dataset_files(backup_info.dataset_id)
                    for file_path in dataset_files:
                        if Path(file_path).exists():
                            tar.add(file_path, arcname=Path(file_path).name)
                else:
                    # Add all data files
                    data_dir = Path("data")
                    if data_dir.exists():
                        tar.add(data_dir, arcname="data")
            
            # Update backup info
            backup_info.file_path = str(backup_file)
            backup_info.compressed_size_bytes = backup_file.stat().st_size
            backup_info.checksum = self._calculate_file_checksum(str(backup_file))
            backup_info.completed_at = datetime.now()
            
            self._update_backup_info(backup_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create snapshot backup: {e}")
            return False
    
    def _perform_restore(self, backup_info: BackupInfo, restore_info: RestoreInfo,
                        selective_restore: List[str] = None) -> bool:
        """Perform restore operation."""
        try:
            if backup_info.backup_type == BackupType.FULL:
                return self._restore_full_backup(backup_info, restore_info, selective_restore)
            elif backup_info.backup_type in [BackupType.INCREMENTAL, BackupType.DIFFERENTIAL]:
                return self._restore_incremental_backup(backup_info, restore_info, selective_restore)
            elif backup_info.backup_type == BackupType.SNAPSHOT:
                return self._restore_snapshot_backup(backup_info, restore_info, selective_restore)
            else:
                logger.error(f"Unsupported backup type for restore: {backup_info.backup_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to perform restore: {e}")
            return False
    
    def _restore_full_backup(self, backup_info: BackupInfo, restore_info: RestoreInfo,
                           selective_restore: List[str] = None) -> bool:
        """Restore from a full backup."""
        try:
            # Read backup file
            if backup_info.compression_type == CompressionType.GZIP:
                with gzip.open(backup_info.file_path, 'rt') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_info.file_path, 'r') as f:
                    backup_data = json.load(f)
            
            # Restore data to database
            restored_files = []
            
            if backup_info.dataset_id:
                # Restore specific dataset
                success = self._import_dataset(backup_data, backup_info.dataset_id)
                if success:
                    restored_files.append(f"dataset_{backup_info.dataset_id}")
            else:
                # Restore entire database
                success = self._import_database(backup_data, selective_restore)
                if success:
                    restored_files.extend(backup_data.keys() if isinstance(backup_data, dict) else ["database"])
            
            # Update restore info
            restore_info.restored_files = restored_files
            restore_info.completed_at = datetime.now()
            self._update_restore_info(restore_info)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to restore full backup: {e}")
            restore_info.errors.append(str(e))
            self._update_restore_info(restore_info)
            return False
    
    def _restore_incremental_backup(self, backup_info: BackupInfo, restore_info: RestoreInfo,
                                  selective_restore: List[str] = None) -> bool:
        """Restore from an incremental or differential backup."""
        try:
            # First, restore the parent backup
            if backup_info.parent_backup_id:
                parent_backup = self.get_backup_info(backup_info.parent_backup_id)
                if parent_backup:
                    parent_restore_id = self.restore_backup(
                        backup_info.parent_backup_id,
                        restore_info.target_location,
                        selective_restore
                    )
                    # Wait for parent restore to complete (simplified)
                    time.sleep(1)
            
            # Read incremental backup file
            if backup_info.compression_type == CompressionType.GZIP:
                with gzip.open(backup_info.file_path, 'rt') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_info.file_path, 'r') as f:
                    backup_data = json.load(f)
            
            # Apply changes
            changes = backup_data.get("changes", {})
            restored_files = []
            
            for table_name, table_changes in changes.items():
                if selective_restore and table_name not in selective_restore:
                    continue
                
                success = self._apply_table_changes(table_name, table_changes)
                if success:
                    restored_files.append(table_name)
            
            # Update restore info
            restore_info.restored_files = restored_files
            restore_info.completed_at = datetime.now()
            self._update_restore_info(restore_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore incremental backup: {e}")
            restore_info.errors.append(str(e))
            self._update_restore_info(restore_info)
            return False
    
    def _restore_snapshot_backup(self, backup_info: BackupInfo, restore_info: RestoreInfo,
                               selective_restore: List[str] = None) -> bool:
        """Restore from a snapshot backup."""
        try:
            target_dir = Path(restore_info.target_location or "data/restored")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract tar archive
            with tarfile.open(backup_info.file_path, 'r:gz') as tar:
                if selective_restore:
                    # Extract only selected files
                    members = [m for m in tar.getmembers() if any(sel in m.name for sel in selective_restore)]
                    tar.extractall(target_dir, members=members)
                    restored_files = [m.name for m in members]
                else:
                    # Extract all files
                    tar.extractall(target_dir)
                    restored_files = tar.getnames()
            
            # Update restore info
            restore_info.restored_files = restored_files
            restore_info.completed_at = datetime.now()
            self._update_restore_info(restore_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot backup: {e}")
            restore_info.errors.append(str(e))
            self._update_restore_info(restore_info)
            return False
    
    def _backup_scheduler_loop(self) -> None:
        """Background loop for scheduled backups."""
        while self._scheduler_running:
            try:
                # Check for scheduled backups
                scheduled_backups = self._get_scheduled_backups()
                
                for schedule in scheduled_backups:
                    if self._should_run_backup(schedule):
                        logger.info(f"Running scheduled backup: {schedule['schedule_id']}")
                        
                        backup_id = self.create_backup(
                            backup_type=BackupType(schedule['backup_type']),
                            dataset_id=schedule.get('dataset_id'),
                            description=f"Scheduled backup: {schedule['schedule_id']}"
                        )
                        
                        # Update last run time
                        self._update_schedule_last_run(schedule['schedule_id'])
                
                # Sleep for a while before checking again
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                time.sleep(3600)  # Wait before retrying
    
    def _generate_backup_id(self) -> str:
        """Generate a unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _generate_restore_id(self) -> str:
        """Generate a unique restore ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"restore_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _export_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Export a specific dataset."""
        # This is a simplified implementation
        # In practice, you'd export all related tables and data
        with self.connector.get_connection() as conn:
            # Export documents
            doc_query = "SELECT * FROM documents WHERE id IN (SELECT document_id FROM dataset_documents WHERE dataset_id = ?)"
            
            import sqlalchemy
            result = conn.execute(sqlalchemy.text(doc_query), (dataset_id,))
            
            documents = [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in result]
            
            return {
                "dataset_id": dataset_id,
                "documents": documents,
                "export_timestamp": datetime.now().isoformat()
            }
    
    def _export_database(self) -> Dict[str, Any]:
        """Export entire database."""
        # This is a simplified implementation
        # In practice, you'd export all tables
        with self.connector.get_connection() as conn:
            tables = ["documents", "datasets", "document_entities"]
            export_data = {}
            
            for table in tables:
                query = f"SELECT * FROM {table}"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query))
                else:
                    result = conn.execute(query)
                
                export_data[table] = [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in result]
            
            export_data["export_timestamp"] = datetime.now().isoformat()
            return export_data
    
    def _import_dataset(self, data: Dict[str, Any], dataset_id: str) -> bool:
        """Import dataset data."""
        # Simplified implementation
        try:
            with self.connector.get_connection() as conn:
                # Import documents
                for doc in data.get("documents", []):
                    # Insert or update document
                    # This would need proper SQL generation based on the data structure
                    pass
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to import dataset: {e}")
            return False
    
    def _import_database(self, data: Dict[str, Any], selective_restore: List[str] = None) -> bool:
        """Import database data."""
        # Simplified implementation
        try:
            with self.connector.get_connection() as conn:
                for table_name, table_data in data.items():
                    if table_name == "export_timestamp":
                        continue
                    
                    if selective_restore and table_name not in selective_restore:
                        continue
                    
                    # Import table data
                    # This would need proper SQL generation based on the data structure
                    pass
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False
    
    def _store_backup_info(self, backup_info: BackupInfo) -> None:
        """Store backup information in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO backups 
            (backup_id, backup_type, dataset_id, created_at, status, 
             compression_type, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                backup_info.backup_id,
                backup_info.backup_type.value,
                backup_info.dataset_id,
                backup_info.created_at.isoformat(),
                backup_info.status.value,
                backup_info.compression_type.value,
                backup_info.description,
                json.dumps(backup_info.metadata)
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _update_backup_info(self, backup_info: BackupInfo) -> None:
        """Update backup information in database."""
        with self.connector.get_connection() as conn:
            query = """
            UPDATE backups 
            SET completed_at = ?, status = ?, file_path = ?, 
                size_bytes = ?, compressed_size_bytes = ?, checksum = ?,
                parent_backup_id = ?
            WHERE backup_id = ?
            """
            
            params = (
                backup_info.completed_at.isoformat() if backup_info.completed_at else None,
                backup_info.status.value,
                backup_info.file_path,
                backup_info.size_bytes,
                backup_info.compressed_size_bytes,
                backup_info.checksum,
                backup_info.parent_backup_id,
                backup_info.backup_id
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _update_backup_status(self, backup_id: str, status: BackupStatus) -> None:
        """Update backup status."""
        with self.connector.get_connection() as conn:
            query = "UPDATE backups SET status = ? WHERE backup_id = ?"
            
            if self.connector.config.connection_type == "postgresql":
                import sqlalchemy
                conn.execute(sqlalchemy.text(query), (status.value, backup_id))
            else:
                conn.execute(query, (status.value, backup_id))
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _store_restore_info(self, restore_info: RestoreInfo) -> None:
        """Store restore information in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO restore_operations 
            (restore_id, backup_id, started_at, status, target_location, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                restore_info.restore_id,
                restore_info.backup_id,
                restore_info.started_at.isoformat(),
                restore_info.status,
                restore_info.target_location,
                json.dumps(restore_info.metadata)
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _update_restore_info(self, restore_info: RestoreInfo) -> None:
        """Update restore information in database."""
        with self.connector.get_connection() as conn:
            query = """
            UPDATE restore_operations 
            SET completed_at = ?, status = ?, restored_files = ?, errors = ?
            WHERE restore_id = ?
            """
            
            params = (
                restore_info.completed_at.isoformat() if restore_info.completed_at else None,
                restore_info.status,
                json.dumps(restore_info.restored_files),
                json.dumps(restore_info.errors),
                restore_info.restore_id
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _update_restore_status(self, restore_id: str, status: str) -> None:
        """Update restore status."""
        with self.connector.get_connection() as conn:
            query = "UPDATE restore_operations SET status = ? WHERE restore_id = ?"
            
            if self.connector.config.connection_type == "postgresql":
                import sqlalchemy
                conn.execute(sqlalchemy.text(query), (status, restore_id))
            else:
                conn.execute(query, (status, restore_id))
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _row_to_backup_info(self, row) -> BackupInfo:
        """Convert database row to BackupInfo object."""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        return BackupInfo(
            backup_id=row_dict["backup_id"],
            backup_type=BackupType(row_dict["backup_type"]),
            dataset_id=row_dict.get("dataset_id"),
            created_at=datetime.fromisoformat(row_dict["created_at"]),
            completed_at=datetime.fromisoformat(row_dict["completed_at"]) if row_dict.get("completed_at") else None,
            status=BackupStatus(row_dict.get("status", "pending")),
            file_path=row_dict.get("file_path"),
            size_bytes=row_dict.get("size_bytes", 0),
            compressed_size_bytes=row_dict.get("compressed_size_bytes", 0),
            compression_type=CompressionType(row_dict.get("compression_type", "gzip")),
            checksum=row_dict.get("checksum"),
            parent_backup_id=row_dict.get("parent_backup_id"),
            description=row_dict.get("description"),
            metadata=json.loads(row_dict.get("metadata", "{}"))
        )
    
    def _find_parent_backup(self, dataset_id: str, backup_type: BackupType) -> Optional[BackupInfo]:
        """Find the most recent parent backup."""
        backups = self.list_backups(dataset_id=dataset_id, status=BackupStatus.COMPLETED)
        
        # Filter by backup type
        if backup_type == BackupType.INCREMENTAL:
            # For incremental, find last full or incremental backup
            valid_backups = [b for b in backups if b.backup_type in [BackupType.FULL, BackupType.INCREMENTAL]]
        else:
            # For differential, find last full backup
            valid_backups = [b for b in backups if b.backup_type == BackupType.FULL]
        
        if valid_backups:
            # Return most recent
            return max(valid_backups, key=lambda b: b.created_at)
        
        return None
    
    def _get_changes_since_backup(self, parent_backup: BackupInfo) -> Dict[str, Any]:
        """Get changes since a parent backup."""
        # This is a simplified implementation
        # In practice, you'd compare current state with parent backup state
        return {
            "documents": {"added": [], "modified": [], "deleted": []},
            "datasets": {"added": [], "modified": [], "deleted": []},
            "since_timestamp": parent_backup.completed_at.isoformat() if parent_backup.completed_at else None
        }
    
    def _apply_table_changes(self, table_name: str, changes: Dict[str, Any]) -> bool:
        """Apply changes to a table."""
        # Simplified implementation
        try:
            with self.connector.get_connection() as conn:
                # Apply added records
                for record in changes.get("added", []):
                    # Insert record
                    pass
                
                # Apply modified records
                for record in changes.get("modified", []):
                    # Update record
                    pass
                
                # Apply deleted records
                for record in changes.get("deleted", []):
                    # Delete record
                    pass
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply changes to table {table_name}: {e}")
            return False
    
    def _get_datasets_with_backups(self) -> List[str]:
        """Get list of dataset IDs that have backups."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT DISTINCT dataset_id FROM backups WHERE dataset_id IS NOT NULL"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query))
                else:
                    result = conn.execute(query)
                
                return [row[0] for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get datasets with backups: {e}")
            return []
    
    def _get_dataset_files(self, dataset_id: str) -> List[str]:
        """Get list of files associated with a dataset."""
        # Simplified implementation - return empty list
        # In practice, you'd return paths to dataset-related files
        return []
    
    def _get_scheduled_backups(self) -> List[Dict[str, Any]]:
        """Get scheduled backup configurations."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM backup_schedule WHERE enabled = TRUE"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query))
                else:
                    result = conn.execute(query)
                
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get scheduled backups: {e}")
            return []
    
    def _should_run_backup(self, schedule: Dict[str, Any]) -> bool:
        """Check if a scheduled backup should run."""
        # Simplified implementation - check if enough time has passed
        if not schedule.get("last_run"):
            return True
        
        last_run = datetime.fromisoformat(schedule["last_run"])
        hours_since_last_run = (datetime.now() - last_run).total_seconds() / 3600
        
        return hours_since_last_run >= self.backup_schedule_hours
    
    def _update_schedule_last_run(self, schedule_id: str) -> None:
        """Update the last run time for a backup schedule."""
        try:
            with self.connector.get_connection() as conn:
                query = "UPDATE backup_schedule SET last_run = ? WHERE schedule_id = ?"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(query), (datetime.now().isoformat(), schedule_id))
                else:
                    conn.execute(query, (datetime.now().isoformat(), schedule_id))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update schedule last run: {e}")