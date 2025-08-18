"""
Incremental processor for efficient dataset updates.

This module provides incremental processing capabilities to efficiently
handle updates to large datasets without full reprocessing.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..models import Dataset, Document, ProcessingResult

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of incremental updates."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"


class ProcessingMode(Enum):
    """Processing modes for incremental updates."""
    FULL = "full"          # Full reprocessing
    INCREMENTAL = "incremental"  # Only process changes
    SMART = "smart"        # Intelligent decision based on changes


@dataclass
class FileChange:
    """Represents a change to a file."""
    file_path: str
    change_type: UpdateType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_size: Optional[int] = None
    new_size: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file change to dictionary."""
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class IncrementalUpdate:
    """Represents an incremental update operation."""
    update_id: str
    dataset_id: str
    changes: List[FileChange] = field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.INCREMENTAL
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incremental update to dictionary."""
        return {
            "update_id": self.update_id,
            "dataset_id": self.dataset_id,
            "changes": [change.to_dict() for change in self.changes],
            "processing_mode": self.processing_mode.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "results": self.results
        }


class IncrementalProcessor:
    """Handles incremental processing of dataset updates."""
    
    def __init__(self, database_connector, storage_path: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize incremental processor.
        
        Args:
            database_connector: DatabaseConnector instance
            storage_path: Path for storing incremental data
            config: Configuration for incremental processing
        """
        self.connector = database_connector
        self.storage_path = Path(storage_path or "data/incremental")
        self.config = config or {}
        
        # Configuration options
        self.hash_algorithm = self.config.get("hash_algorithm", "sha256")
        self.batch_size = self.config.get("batch_size", 100)
        self.max_file_size = self.config.get("max_file_size_bytes", 100 * 1024 * 1024)  # 100MB
        self.enable_smart_mode = self.config.get("enable_smart_mode", True)
        
        # File tracking
        self.file_hashes: Dict[str, str] = {}
        self.file_sizes: Dict[str, int] = {}
        self.last_scan_time: Optional[datetime] = None
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize incremental tables
        self._initialize_incremental_tables()
        
        # Load existing file tracking data
        self._load_file_tracking_data()
    
    def _initialize_incremental_tables(self) -> None:
        """Initialize database tables for incremental processing."""
        try:
            with self.connector.get_connection() as conn:
                # File tracking table
                file_tracking_sql = """
                CREATE TABLE IF NOT EXISTS file_tracking (
                    file_path VARCHAR(500) PRIMARY KEY,
                    file_hash VARCHAR(64) NOT NULL,
                    file_size BIGINT NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    last_processed TIMESTAMP,
                    document_id VARCHAR(255),
                    metadata TEXT
                )
                """
                
                # Incremental updates table
                updates_sql = """
                CREATE TABLE IF NOT EXISTS incremental_updates (
                    update_id VARCHAR(255) PRIMARY KEY,
                    dataset_id VARCHAR(255) NOT NULL,
                    processing_mode VARCHAR(20) NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    change_count INTEGER DEFAULT 0,
                    results TEXT,
                    metadata TEXT
                )
                """
                
                # File changes table
                changes_sql = """
                CREATE TABLE IF NOT EXISTS file_changes (
                    id VARCHAR(255) PRIMARY KEY,
                    update_id VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    change_type VARCHAR(20) NOT NULL,
                    old_hash VARCHAR(64),
                    new_hash VARCHAR(64),
                    old_size BIGINT,
                    new_size BIGINT,
                    timestamp TIMESTAMP NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(file_tracking_sql))
                    conn.execute(sqlalchemy.text(updates_sql))
                    conn.execute(sqlalchemy.text(changes_sql))
                else:
                    conn.execute(file_tracking_sql)
                    conn.execute(updates_sql)
                    conn.execute(changes_sql)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to initialize incremental tables: {e}")
    
    def scan_for_changes(self, source_paths: List[str]) -> List[FileChange]:
        """
        Scan source paths for file changes.
        
        Args:
            source_paths: List of paths to scan for changes
            
        Returns:
            List of detected file changes
        """
        changes = []
        current_files = set()
        
        logger.info(f"Scanning {len(source_paths)} paths for changes")
        
        for source_path in source_paths:
            path = Path(source_path)
            
            if path.is_file():
                # Single file
                change = self._check_file_change(path)
                if change:
                    changes.append(change)
                current_files.add(str(path))
                
            elif path.is_dir():
                # Directory - scan recursively
                for file_path in path.rglob("*"):
                    if file_path.is_file() and self._should_process_file(file_path):
                        change = self._check_file_change(file_path)
                        if change:
                            changes.append(change)
                        current_files.add(str(file_path))
        
        # Check for deleted files
        deleted_changes = self._check_deleted_files(current_files)
        changes.extend(deleted_changes)
        
        # Update last scan time
        self.last_scan_time = datetime.now()
        
        logger.info(f"Found {len(changes)} file changes")
        return changes
    
    def create_incremental_update(self, dataset_id: str, changes: List[FileChange],
                                processing_mode: ProcessingMode = None) -> str:
        """
        Create an incremental update operation.
        
        Args:
            dataset_id: ID of the dataset to update
            changes: List of file changes to process
            processing_mode: Processing mode (auto-detected if None)
            
        Returns:
            Update ID
        """
        update_id = self._generate_update_id()
        
        # Auto-detect processing mode if not specified
        if processing_mode is None:
            processing_mode = self._determine_processing_mode(changes)
        
        update = IncrementalUpdate(
            update_id=update_id,
            dataset_id=dataset_id,
            changes=changes,
            processing_mode=processing_mode
        )
        
        try:
            # Store update in database
            self._store_incremental_update(update)
            
            # Store file changes
            for change in changes:
                self._store_file_change(update_id, change)
            
            logger.info(f"Created incremental update {update_id} with {len(changes)} changes")
            return update_id
            
        except Exception as e:
            logger.error(f"Failed to create incremental update: {e}")
            raise
    
    def process_incremental_update(self, update_id: str, 
                                 processor_func: callable) -> ProcessingResult:
        """
        Process an incremental update.
        
        Args:
            update_id: ID of the update to process
            processor_func: Function to process individual files
            
        Returns:
            Processing result
        """
        try:
            # Get update info
            update = self._get_incremental_update(update_id)
            if not update:
                raise ValueError(f"Update {update_id} not found")
            
            # Update status
            self._update_status(update_id, "processing")
            
            results = {
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "errors": []
            }
            
            # Process changes in batches
            changes = self._get_file_changes(update_id)
            
            for i in range(0, len(changes), self.batch_size):
                batch = changes[i:i + self.batch_size]
                batch_results = self._process_change_batch(batch, processor_func)
                
                # Aggregate results
                results["processed_files"] += batch_results["processed_files"]
                results["successful_files"] += batch_results["successful_files"]
                results["failed_files"] += batch_results["failed_files"]
                results["errors"].extend(batch_results["errors"])
                
                # Update progress
                progress = min(100, (i + len(batch)) * 100 // len(changes))
                logger.info(f"Incremental update {update_id} progress: {progress}%")
            
            # Update completion status
            status = "completed" if results["failed_files"] == 0 else "completed_with_errors"
            self._update_status(update_id, status, results)
            
            logger.info(f"Completed incremental update {update_id}: {results}")
            
            return ProcessingResult(
                success=results["failed_files"] == 0,
                processing_time=0.0,  # TODO: Track actual processing time
                stage_results={"incremental_update": results}
            )
            
        except Exception as e:
            logger.error(f"Failed to process incremental update {update_id}: {e}")
            self._update_status(update_id, "failed", {"error": str(e)})
            raise
    
    def get_update_status(self, update_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an incremental update.
        
        Args:
            update_id: ID of the update
            
        Returns:
            Update status information
        """
        return self._get_incremental_update(update_id)
    
    def list_pending_updates(self, dataset_id: str = None) -> List[Dict[str, Any]]:
        """
        List pending incremental updates.
        
        Args:
            dataset_id: Filter by dataset ID (optional)
            
        Returns:
            List of pending updates
        """
        try:
            with self.connector.get_connection() as conn:
                if dataset_id:
                    query = """
                    SELECT * FROM incremental_updates 
                    WHERE dataset_id = ? AND status = 'pending'
                    ORDER BY started_at
                    """
                    params = (dataset_id,)
                else:
                    query = """
                    SELECT * FROM incremental_updates 
                    WHERE status = 'pending'
                    ORDER BY started_at
                    """
                    params = ()
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), params)
                
                updates = []
                for row in result:
                    row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                    updates.append(row_dict)
                
                return updates
                
        except Exception as e:
            logger.error(f"Failed to list pending updates: {e}")
            return []
    
    def optimize_file_tracking(self) -> int:
        """
        Optimize file tracking by removing stale entries.
        
        Returns:
            Number of entries removed
        """
        try:
            with self.connector.get_connection() as conn:
                # Remove entries for files that no longer exist
                query = "SELECT file_path FROM file_tracking"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query))
                else:
                    result = conn.execute(query)
                
                stale_files = []
                for row in result:
                    file_path = row[0]
                    if not Path(file_path).exists():
                        stale_files.append(file_path)
                
                # Delete stale entries
                if stale_files:
                    placeholders = ",".join(["?" for _ in stale_files])
                    delete_query = f"DELETE FROM file_tracking WHERE file_path IN ({placeholders})"
                    
                    if self.connector.config.connection_type == "postgresql":
                        conn.execute(sqlalchemy.text(delete_query), stale_files)
                    else:
                        conn.execute(delete_query, stale_files)
                    
                    if hasattr(conn, 'commit'):
                        conn.commit()
                
                logger.info(f"Removed {len(stale_files)} stale file tracking entries")
                return len(stale_files)
                
        except Exception as e:
            logger.error(f"Failed to optimize file tracking: {e}")
            return 0
    
    def get_processing_statistics(self, dataset_id: str = None, 
                                days: int = 30) -> Dict[str, Any]:
        """
        Get processing statistics for incremental updates.
        
        Args:
            dataset_id: Filter by dataset ID (optional)
            days: Number of days to include in statistics
            
        Returns:
            Processing statistics
        """
        try:
            cutoff_date = datetime.now().replace(
                day=datetime.now().day - days
            )
            
            with self.connector.get_connection() as conn:
                # Base query conditions
                conditions = ["started_at >= ?"]
                params = [cutoff_date.isoformat()]
                
                if dataset_id:
                    conditions.append("dataset_id = ?")
                    params.append(dataset_id)
                
                where_clause = " AND ".join(conditions)
                
                # Get update counts by status
                status_query = f"""
                SELECT status, COUNT(*) as count
                FROM incremental_updates
                WHERE {where_clause}
                GROUP BY status
                """
                
                # Get processing mode distribution
                mode_query = f"""
                SELECT processing_mode, COUNT(*) as count
                FROM incremental_updates
                WHERE {where_clause}
                GROUP BY processing_mode
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    status_result = conn.execute(sqlalchemy.text(status_query), params)
                    mode_result = conn.execute(sqlalchemy.text(mode_query), params)
                else:
                    status_result = conn.execute(status_query, params)
                    mode_result = conn.execute(mode_query, params)
                
                # Aggregate results
                status_counts = {row[0]: row[1] for row in status_result}
                mode_counts = {row[0]: row[1] for row in mode_result}
                
                return {
                    "period_days": days,
                    "dataset_id": dataset_id,
                    "status_distribution": status_counts,
                    "mode_distribution": mode_counts,
                    "total_updates": sum(status_counts.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {}
    
    def _check_file_change(self, file_path: Path) -> Optional[FileChange]:
        """Check if a file has changed."""
        try:
            if not file_path.exists():
                return None
            
            # Skip files that are too large
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
                return None
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            file_path_str = str(file_path)
            
            # Check if file is new or changed
            old_hash = self.file_hashes.get(file_path_str)
            old_size = self.file_sizes.get(file_path_str)
            
            if old_hash is None:
                # New file
                change = FileChange(
                    file_path=file_path_str,
                    change_type=UpdateType.ADD,
                    new_hash=file_hash,
                    new_size=file_size
                )
                
                # Update tracking
                self.file_hashes[file_path_str] = file_hash
                self.file_sizes[file_path_str] = file_size
                self._update_file_tracking(file_path_str, file_hash, file_size)
                
                return change
                
            elif old_hash != file_hash:
                # Changed file
                change = FileChange(
                    file_path=file_path_str,
                    change_type=UpdateType.UPDATE,
                    old_hash=old_hash,
                    new_hash=file_hash,
                    old_size=old_size,
                    new_size=file_size
                )
                
                # Update tracking
                self.file_hashes[file_path_str] = file_hash
                self.file_sizes[file_path_str] = file_size
                self._update_file_tracking(file_path_str, file_hash, file_size)
                
                return change
            
            # File unchanged
            return None
            
        except Exception as e:
            logger.error(f"Failed to check file change for {file_path}: {e}")
            return None
    
    def _check_deleted_files(self, current_files: Set[str]) -> List[FileChange]:
        """Check for deleted files."""
        deleted_changes = []
        
        for tracked_file in list(self.file_hashes.keys()):
            if tracked_file not in current_files:
                # File was deleted
                change = FileChange(
                    file_path=tracked_file,
                    change_type=UpdateType.DELETE,
                    old_hash=self.file_hashes[tracked_file],
                    old_size=self.file_sizes.get(tracked_file)
                )
                deleted_changes.append(change)
                
                # Remove from tracking
                del self.file_hashes[tracked_file]
                if tracked_file in self.file_sizes:
                    del self.file_sizes[tracked_file]
                self._remove_file_tracking(tracked_file)
        
        return deleted_changes
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Check file extension whitelist if configured
        allowed_extensions = self.config.get("allowed_extensions")
        if allowed_extensions:
            if file_path.suffix.lower() not in allowed_extensions:
                return False
        
        # Check file extension blacklist if configured
        blocked_extensions = self.config.get("blocked_extensions", [])
        if file_path.suffix.lower() in blocked_extensions:
            return False
        
        return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file."""
        hash_func = hashlib.new(self.hash_algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _determine_processing_mode(self, changes: List[FileChange]) -> ProcessingMode:
        """Determine optimal processing mode based on changes."""
        if not self.enable_smart_mode:
            return ProcessingMode.INCREMENTAL
        
        total_changes = len(changes)
        add_count = sum(1 for c in changes if c.change_type == UpdateType.ADD)
        update_count = sum(1 for c in changes if c.change_type == UpdateType.UPDATE)
        delete_count = sum(1 for c in changes if c.change_type == UpdateType.DELETE)
        
        # Thresholds for smart mode decision
        full_reprocess_threshold = self.config.get("full_reprocess_threshold", 0.5)
        
        # If more than threshold of files changed, do full reprocessing
        if total_changes > 0:
            change_ratio = (update_count + delete_count) / total_changes
            if change_ratio > full_reprocess_threshold:
                return ProcessingMode.FULL
        
        return ProcessingMode.INCREMENTAL
    
    def _process_change_batch(self, changes: List[FileChange], 
                            processor_func: callable) -> Dict[str, Any]:
        """Process a batch of file changes."""
        results = {
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "errors": []
        }
        
        for change in changes:
            try:
                results["processed_files"] += 1
                
                # Process the change
                if change.change_type == UpdateType.DELETE:
                    # Handle file deletion
                    success = self._handle_file_deletion(change)
                else:
                    # Process file (add or update)
                    success = processor_func(change.file_path, change)
                
                if success:
                    results["successful_files"] += 1
                    self._mark_change_processed(change)
                else:
                    results["failed_files"] += 1
                    
            except Exception as e:
                results["failed_files"] += 1
                error_msg = f"Failed to process {change.file_path}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def _handle_file_deletion(self, change: FileChange) -> bool:
        """Handle deletion of a file."""
        try:
            # Remove from database if document exists
            with self.connector.get_connection() as conn:
                # Find document ID for this file
                query = "SELECT document_id FROM file_tracking WHERE file_path = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (change.file_path,))
                
                row = result.fetchone()
                if row:
                    document_id = row[0]
                    
                    # Delete document and related records
                    delete_queries = [
                        "DELETE FROM document_entities WHERE document_id = ?",
                        "DELETE FROM dataset_documents WHERE document_id = ?",
                        "DELETE FROM documents WHERE id = ?"
                    ]
                    
                    for delete_query in delete_queries:
                        if self.connector.config.connection_type == "postgresql":
                            conn.execute(sqlalchemy.text(delete_query), (document_id,))
                        else:
                            conn.execute(delete_query, (document_id,))
                    
                    if hasattr(conn, 'commit'):
                        conn.commit()
                    
                    logger.info(f"Deleted document {document_id} for file {change.file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle file deletion for {change.file_path}: {e}")
            return False
    
    def _generate_update_id(self) -> str:
        """Generate a unique update ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"inc_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _load_file_tracking_data(self) -> None:
        """Load existing file tracking data from database."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT file_path, file_hash, file_size FROM file_tracking"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query))
                else:
                    result = conn.execute(query)
                
                for row in result:
                    file_path, file_hash, file_size = row
                    self.file_hashes[file_path] = file_hash
                    self.file_sizes[file_path] = file_size
                
                logger.info(f"Loaded tracking data for {len(self.file_hashes)} files")
                
        except Exception as e:
            logger.error(f"Failed to load file tracking data: {e}")
    
    def _update_file_tracking(self, file_path: str, file_hash: str, file_size: int) -> None:
        """Update file tracking in database."""
        try:
            with self.connector.get_connection() as conn:
                query = """
                INSERT OR REPLACE INTO file_tracking 
                (file_path, file_hash, file_size, last_modified)
                VALUES (?, ?, ?, ?)
                """
                
                if self.connector.config.connection_type == "postgresql":
                    query = """
                    INSERT INTO file_tracking 
                    (file_path, file_hash, file_size, last_modified)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (file_path) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    file_size = EXCLUDED.file_size,
                    last_modified = EXCLUDED.last_modified
                    """
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(query), (file_path, file_hash, file_size, datetime.now().isoformat()))
                else:
                    conn.execute(query, (file_path, file_hash, file_size, datetime.now().isoformat()))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update file tracking for {file_path}: {e}")
    
    def _remove_file_tracking(self, file_path: str) -> None:
        """Remove file tracking from database."""
        try:
            with self.connector.get_connection() as conn:
                query = "DELETE FROM file_tracking WHERE file_path = ?"
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(query), (file_path,))
                else:
                    conn.execute(query, (file_path,))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to remove file tracking for {file_path}: {e}")
    
    def _store_incremental_update(self, update: IncrementalUpdate) -> None:
        """Store incremental update in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO incremental_updates 
            (update_id, dataset_id, processing_mode, started_at, status, change_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                update.update_id,
                update.dataset_id,
                update.processing_mode.value,
                update.started_at.isoformat(),
                update.status,
                len(update.changes)
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _store_file_change(self, update_id: str, change: FileChange) -> None:
        """Store file change in database."""
        with self.connector.get_connection() as conn:
            change_id = f"{update_id}_{hashlib.md5(change.file_path.encode()).hexdigest()[:8]}"
            
            query = """
            INSERT INTO file_changes 
            (id, update_id, file_path, change_type, old_hash, new_hash, 
             old_size, new_size, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                change_id,
                update_id,
                change.file_path,
                change.change_type.value,
                change.old_hash,
                change.new_hash,
                change.old_size,
                change.new_size,
                change.timestamp.isoformat(),
                json.dumps(change.metadata)
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _get_incremental_update(self, update_id: str) -> Optional[Dict[str, Any]]:
        """Get incremental update from database."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM incremental_updates WHERE update_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (update_id,))
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get incremental update {update_id}: {e}")
            return None
    
    def _get_file_changes(self, update_id: str) -> List[FileChange]:
        """Get file changes for an update."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM file_changes WHERE update_id = ? ORDER BY timestamp"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (update_id,))
                
                changes = []
                for row in result:
                    row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                    
                    change = FileChange(
                        file_path=row_dict["file_path"],
                        change_type=UpdateType(row_dict["change_type"]),
                        old_hash=row_dict.get("old_hash"),
                        new_hash=row_dict.get("new_hash"),
                        old_size=row_dict.get("old_size"),
                        new_size=row_dict.get("new_size"),
                        timestamp=datetime.fromisoformat(row_dict["timestamp"]),
                        metadata=json.loads(row_dict.get("metadata", "{}"))
                    )
                    changes.append(change)
                
                return changes
                
        except Exception as e:
            logger.error(f"Failed to get file changes for update {update_id}: {e}")
            return []
    
    def _update_status(self, update_id: str, status: str, results: Dict[str, Any] = None) -> None:
        """Update status of an incremental update."""
        try:
            with self.connector.get_connection() as conn:
                if status in ["completed", "completed_with_errors", "failed"]:
                    query = """
                    UPDATE incremental_updates 
                    SET status = ?, completed_at = ?, results = ?
                    WHERE update_id = ?
                    """
                    params = (
                        status,
                        datetime.now().isoformat(),
                        json.dumps(results) if results else None,
                        update_id
                    )
                else:
                    query = "UPDATE incremental_updates SET status = ? WHERE update_id = ?"
                    params = (status, update_id)
                
                import sqlalchemy
                conn.execute(sqlalchemy.text(query), params)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update status for update {update_id}: {e}")
    
    def _mark_change_processed(self, change: FileChange) -> None:
        """Mark a file change as processed."""
        try:
            with self.connector.get_connection() as conn:
                query = """
                UPDATE file_changes 
                SET processed = TRUE 
                WHERE file_path = ? AND change_type = ?
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(query), (change.file_path, change.change_type.value))
                else:
                    conn.execute(query, (change.file_path, change.change_type.value))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to mark change processed for {change.file_path}: {e}")