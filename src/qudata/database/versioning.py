"""
Data versioning system for dataset snapshots and change tracking.

This module provides comprehensive data versioning capabilities including
snapshot creation, change tracking, rollback functionality, and version comparison.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..models import Dataset, Document

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes that can be tracked."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"


class VersionStatus(Enum):
    """Status of a version."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ChangeRecord:
    """Record of a change made to data."""
    id: str
    change_type: ChangeType
    entity_type: str  # "document", "dataset", etc.
    entity_id: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert change record to dictionary."""
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class VersionInfo:
    """Information about a data version."""
    version_id: str
    parent_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: VersionStatus = VersionStatus.ACTIVE
    checksum: Optional[str] = None
    size_bytes: int = 0
    change_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version info to dictionary."""
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "status": self.status.value,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "change_count": self.change_count,
            "metadata": self.metadata
        }


@dataclass
class Snapshot:
    """A complete snapshot of data at a point in time."""
    snapshot_id: str
    version_info: VersionInfo
    data: Dict[str, Any]
    changes_since_parent: List[ChangeRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "version_info": self.version_info.to_dict(),
            "data": self.data,
            "changes_since_parent": [
                change.to_dict() for change in self.changes_since_parent
            ]
        }


class DataVersioning:
    """Manages data versioning, snapshots, and change tracking."""
    
    def __init__(self, database_connector, storage_path: str = None, 
                 config: Dict[str, Any] = None):
        """
        Initialize data versioning system.
        
        Args:
            database_connector: DatabaseConnector instance
            storage_path: Path for storing version data
            config: Configuration for versioning system
        """
        self.connector = database_connector
        self.storage_path = Path(storage_path or "data/versions")
        self.config = config or {}
        self.current_version: Optional[str] = None
        self.change_log: List[ChangeRecord] = []
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize versioning tables if needed
        self._initialize_versioning_tables()
    
    def _initialize_versioning_tables(self) -> None:
        """Initialize database tables for versioning."""
        try:
            with self.connector.get_connection() as conn:
                # Create versions table
                versions_sql = """
                CREATE TABLE IF NOT EXISTS data_versions (
                    version_id VARCHAR(255) PRIMARY KEY,
                    parent_version VARCHAR(255),
                    created_at TIMESTAMP NOT NULL,
                    created_by VARCHAR(255),
                    description TEXT,
                    tags TEXT,
                    status VARCHAR(20) DEFAULT 'active',
                    checksum VARCHAR(64),
                    size_bytes BIGINT DEFAULT 0,
                    change_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
                """
                
                # Create change log table
                changes_sql = """
                CREATE TABLE IF NOT EXISTS change_log (
                    id VARCHAR(255) PRIMARY KEY,
                    version_id VARCHAR(255),
                    change_type VARCHAR(20) NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id VARCHAR(255) NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    user_id VARCHAR(255),
                    description TEXT,
                    metadata TEXT
                )
                """
                
                # Create snapshots table
                snapshots_sql = """
                CREATE TABLE IF NOT EXISTS data_snapshots (
                    snapshot_id VARCHAR(255) PRIMARY KEY,
                    version_id VARCHAR(255),
                    data_path VARCHAR(500),
                    created_at TIMESTAMP NOT NULL,
                    size_bytes BIGINT DEFAULT 0,
                    compression_type VARCHAR(20),
                    metadata TEXT
                )
                """
                
                import sqlalchemy
                conn.execute(sqlalchemy.text(versions_sql))
                conn.execute(sqlalchemy.text(changes_sql))
                conn.execute(sqlalchemy.text(snapshots_sql))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to initialize versioning tables: {e}")
    
    def create_version(self, description: str = None, tags: List[str] = None,
                      created_by: str = None) -> str:
        """
        Create a new version.
        
        Args:
            description: Description of the version
            tags: Tags to associate with the version
            created_by: User who created the version
            
        Returns:
            Version ID of the created version
        """
        version_id = self._generate_version_id()
        
        version_info = VersionInfo(
            version_id=version_id,
            parent_version=self.current_version,
            description=description,
            tags=tags or [],
            created_by=created_by
        )
        
        try:
            # Store version info in database
            self._store_version_info(version_info)
            
            # Update current version
            self.current_version = version_id
            
            logger.info(f"Created version {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise
    
    def create_snapshot(self, dataset: Dataset, version_id: str = None) -> str:
        """
        Create a snapshot of a dataset.
        
        Args:
            dataset: Dataset to snapshot
            version_id: Version ID (creates new version if not provided)
            
        Returns:
            Snapshot ID
        """
        if not version_id:
            version_id = self.create_version(
                description=f"Snapshot of dataset {dataset.name}"
            )
        
        snapshot_id = self._generate_snapshot_id()
        
        # Calculate checksum
        dataset_json = dataset.to_json()
        checksum = hashlib.sha256(dataset_json.encode()).hexdigest()
        
        # Store snapshot data
        snapshot_path = self.storage_path / f"{snapshot_id}.json"
        with open(snapshot_path, 'w') as f:
            f.write(dataset_json)
        
        # Create snapshot record
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            version_info=VersionInfo(
                version_id=version_id,
                checksum=checksum,
                size_bytes=len(dataset_json.encode())
            ),
            data=dataset.to_dict()
        )
        
        try:
            # Store snapshot info in database
            self._store_snapshot_info(snapshot, str(snapshot_path))
            
            # Update version info with checksum and size
            self._update_version_checksum(version_id, checksum, len(dataset_json.encode()))
            
            logger.info(f"Created snapshot {snapshot_id} for version {version_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            # Clean up snapshot file
            if snapshot_path.exists():
                snapshot_path.unlink()
            raise
    
    def load_snapshot(self, snapshot_id: str) -> Optional[Dataset]:
        """
        Load a dataset from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to load
            
        Returns:
            Dataset object if found, None otherwise
        """
        try:
            snapshot_info = self._get_snapshot_info(snapshot_id)
            if not snapshot_info:
                logger.error(f"Snapshot {snapshot_id} not found")
                return None
            
            snapshot_path = Path(snapshot_info["data_path"])
            if not snapshot_path.exists():
                logger.error(f"Snapshot file not found: {snapshot_path}")
                return None
            
            with open(snapshot_path, 'r') as f:
                dataset_data = json.load(f)
            
            # Reconstruct dataset from JSON data
            dataset = self._reconstruct_dataset(dataset_data)
            
            logger.info(f"Loaded snapshot {snapshot_id}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def track_change(self, change_type: ChangeType, entity_type: str,
                    entity_id: str, old_value: Dict[str, Any] = None,
                    new_value: Dict[str, Any] = None, description: str = None,
                    user_id: str = None) -> str:
        """
        Track a change to data.
        
        Args:
            change_type: Type of change
            entity_type: Type of entity being changed
            entity_id: ID of the entity
            old_value: Previous value (for updates/deletes)
            new_value: New value (for creates/updates)
            description: Description of the change
            user_id: User who made the change
            
        Returns:
            Change record ID
        """
        change_id = self._generate_change_id()
        
        change_record = ChangeRecord(
            id=change_id,
            change_type=change_type,
            entity_type=entity_type,
            entity_id=entity_id,
            old_value=old_value,
            new_value=new_value,
            description=description,
            user_id=user_id
        )
        
        try:
            # Store change record
            self._store_change_record(change_record)
            
            # Add to in-memory change log
            self.change_log.append(change_record)
            
            logger.debug(f"Tracked change {change_id}: {change_type.value} {entity_type} {entity_id}")
            return change_id
            
        except Exception as e:
            logger.error(f"Failed to track change: {e}")
            raise
    
    def get_version_history(self, entity_type: str = None, 
                           entity_id: str = None) -> List[VersionInfo]:
        """
        Get version history.
        
        Args:
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            
        Returns:
            List of version information
        """
        try:
            with self.connector.get_connection() as conn:
                if entity_type and entity_id:
                    # Get versions that affected specific entity
                    query = """
                    SELECT DISTINCT v.* FROM data_versions v
                    JOIN change_log c ON v.version_id = c.version_id
                    WHERE c.entity_type = ? AND c.entity_id = ?
                    ORDER BY v.created_at DESC
                    """
                    params = (entity_type, entity_id)
                else:
                    # Get all versions
                    query = """
                    SELECT * FROM data_versions
                    ORDER BY created_at DESC
                    """
                    params = ()
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), params)
                
                versions = []
                for row in result:
                    version_info = self._row_to_version_info(row)
                    versions.append(version_info)
                
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []
    
    def get_changes(self, version_id: str = None, entity_type: str = None,
                   entity_id: str = None, since: datetime = None) -> List[ChangeRecord]:
        """
        Get change records.
        
        Args:
            version_id: Filter by version ID
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            since: Filter changes since timestamp
            
        Returns:
            List of change records
        """
        try:
            with self.connector.get_connection() as conn:
                query_parts = ["SELECT * FROM change_log WHERE 1=1"]
                params = []
                
                if version_id:
                    query_parts.append("AND version_id = ?")
                    params.append(version_id)
                
                if entity_type:
                    query_parts.append("AND entity_type = ?")
                    params.append(entity_type)
                
                if entity_id:
                    query_parts.append("AND entity_id = ?")
                    params.append(entity_id)
                
                if since:
                    query_parts.append("AND timestamp >= ?")
                    params.append(since.isoformat())
                
                query_parts.append("ORDER BY timestamp DESC")
                query = " ".join(query_parts)
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), params)
                
                changes = []
                for row in result:
                    change_record = self._row_to_change_record(row)
                    changes.append(change_record)
                
                return changes
                
        except Exception as e:
            logger.error(f"Failed to get changes: {e}")
            return []
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            True if rollback was successful
        """
        try:
            # Get version info
            version_info = self._get_version_info(version_id)
            if not version_info:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Create rollback version
            rollback_version_id = self.create_version(
                description=f"Rollback to version {version_id}",
                tags=["rollback"]
            )
            
            # Track rollback change
            self.track_change(
                change_type=ChangeType.RESTORE,
                entity_type="version",
                entity_id=version_id,
                description=f"Rollback to version {version_id}"
            )
            
            logger.info(f"Rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version1: First version ID
            version2: Second version ID
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            changes1 = self.get_changes(version_id=version1)
            changes2 = self.get_changes(version_id=version2)
            
            # Group changes by entity
            entities1 = {f"{c.entity_type}:{c.entity_id}": c for c in changes1}
            entities2 = {f"{c.entity_type}:{c.entity_id}": c for c in changes2}
            
            comparison = {
                "version1": version1,
                "version2": version2,
                "added_entities": [],
                "removed_entities": [],
                "modified_entities": []
            }
            
            # Find added entities
            for entity_key in entities2:
                if entity_key not in entities1:
                    comparison["added_entities"].append(entity_key)
            
            # Find removed entities
            for entity_key in entities1:
                if entity_key not in entities2:
                    comparison["removed_entities"].append(entity_key)
            
            # Find modified entities
            for entity_key in entities1:
                if entity_key in entities2:
                    change1 = entities1[entity_key]
                    change2 = entities2[entity_key]
                    
                    if change1.new_value != change2.new_value:
                        comparison["modified_entities"].append({
                            "entity": entity_key,
                            "old_change": change1.to_dict(),
                            "new_change": change2.to_dict()
                        })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return {}
    
    def cleanup_old_versions(self, keep_count: int = 10, 
                           keep_days: int = 30) -> int:
        """
        Clean up old versions and snapshots.
        
        Args:
            keep_count: Number of recent versions to keep
            keep_days: Number of days to keep versions
            
        Returns:
            Number of versions cleaned up
        """
        try:
            cutoff_date = datetime.now().replace(
                day=datetime.now().day - keep_days
            )
            
            with self.connector.get_connection() as conn:
                # Get old versions to delete
                query = """
                SELECT version_id FROM data_versions
                WHERE created_at < ? AND status != 'active'
                ORDER BY created_at DESC
                OFFSET ?
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(
                        sqlalchemy.text(query), 
                        (cutoff_date.isoformat(), keep_count)
                    )
                else:
                    result = conn.execute(query, (cutoff_date.isoformat(), keep_count))
                
                old_versions = [row[0] for row in result]
                
                # Delete old versions and associated data
                deleted_count = 0
                for version_id in old_versions:
                    if self._delete_version(version_id):
                        deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} old versions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _generate_snapshot_id(self) -> str:
        """Generate a unique snapshot ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"snap_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _generate_change_id(self) -> str:
        """Generate a unique change ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"chg_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _execute_query(self, conn, query: str, params=None):
        """Execute a query with proper parameter handling for different database types."""
        import sqlalchemy
        if params:
            conn.execute(sqlalchemy.text(query), params)
        else:
            conn.execute(sqlalchemy.text(query))
    
    def _store_version_info(self, version_info: VersionInfo) -> None:
        """Store version information in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO data_versions 
            (version_id, parent_version, created_at, created_by, description, 
             tags, status, checksum, size_bytes, change_count, metadata)
            VALUES (:version_id, :parent_version, :created_at, :created_by, :description, 
                    :tags, :status, :checksum, :size_bytes, :change_count, :metadata)
            """
            
            params = {
                'version_id': version_info.version_id,
                'parent_version': version_info.parent_version,
                'created_at': version_info.created_at.isoformat(),
                'created_by': version_info.created_by,
                'description': version_info.description,
                'tags': json.dumps(version_info.tags),
                'status': version_info.status.value,
                'checksum': version_info.checksum,
                'size_bytes': version_info.size_bytes,
                'change_count': version_info.change_count,
                'metadata': json.dumps(version_info.metadata)
            }
            
            self._execute_query(conn, query, params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _store_snapshot_info(self, snapshot: Snapshot, data_path: str) -> None:
        """Store snapshot information in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO data_snapshots 
            (snapshot_id, version_id, data_path, created_at, size_bytes, metadata)
            VALUES (:snapshot_id, :version_id, :data_path, :created_at, :size_bytes, :metadata)
            """
            
            params = {
                'snapshot_id': snapshot.snapshot_id,
                'version_id': snapshot.version_info.version_id,
                'data_path': data_path,
                'created_at': datetime.now().isoformat(),
                'size_bytes': snapshot.version_info.size_bytes,
                'metadata': json.dumps(snapshot.version_info.metadata)
            }
            
            self._execute_query(conn, query, params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _store_change_record(self, change_record: ChangeRecord) -> None:
        """Store change record in database."""
        with self.connector.get_connection() as conn:
            query = """
            INSERT INTO change_log 
            (id, version_id, change_type, entity_type, entity_id, 
             old_value, new_value, timestamp, user_id, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                change_record.id,
                self.current_version,
                change_record.change_type.value,
                change_record.entity_type,
                change_record.entity_id,
                json.dumps(change_record.old_value) if change_record.old_value else None,
                json.dumps(change_record.new_value) if change_record.new_value else None,
                change_record.timestamp.isoformat(),
                change_record.user_id,
                change_record.description,
                json.dumps(change_record.metadata)
            )
            
            import sqlalchemy
            conn.execute(sqlalchemy.text(query), params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version information from database."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM data_versions WHERE version_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (version_id,))
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get version info: {e}")
            return None
    
    def _get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot information from database."""
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM data_snapshots WHERE snapshot_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (snapshot_id,))
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get snapshot info: {e}")
            return None
    
    def _update_version_checksum(self, version_id: str, checksum: str, size_bytes: int) -> None:
        """Update version checksum and size."""
        with self.connector.get_connection() as conn:
            query = """
            UPDATE data_versions 
            SET checksum = :checksum, size_bytes = :size_bytes
            WHERE version_id = :version_id
            """
            
            params = {
                'checksum': checksum,
                'size_bytes': size_bytes,
                'version_id': version_id
            }
            
            self._execute_query(conn, query, params)
            
            if hasattr(conn, 'commit'):
                conn.commit()
    
    def _row_to_version_info(self, row) -> VersionInfo:
        """Convert database row to VersionInfo object."""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        return VersionInfo(
            version_id=row_dict["version_id"],
            parent_version=row_dict.get("parent_version"),
            created_at=datetime.fromisoformat(row_dict["created_at"]),
            created_by=row_dict.get("created_by"),
            description=row_dict.get("description"),
            tags=json.loads(row_dict.get("tags", "[]")),
            status=VersionStatus(row_dict.get("status", "active")),
            checksum=row_dict.get("checksum"),
            size_bytes=row_dict.get("size_bytes", 0),
            change_count=row_dict.get("change_count", 0),
            metadata=json.loads(row_dict.get("metadata", "{}"))
        )
    
    def _row_to_change_record(self, row) -> ChangeRecord:
        """Convert database row to ChangeRecord object."""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        return ChangeRecord(
            id=row_dict["id"],
            change_type=ChangeType(row_dict["change_type"]),
            entity_type=row_dict["entity_type"],
            entity_id=row_dict["entity_id"],
            old_value=json.loads(row_dict["old_value"]) if row_dict.get("old_value") else None,
            new_value=json.loads(row_dict["new_value"]) if row_dict.get("new_value") else None,
            timestamp=datetime.fromisoformat(row_dict["timestamp"]),
            user_id=row_dict.get("user_id"),
            description=row_dict.get("description"),
            metadata=json.loads(row_dict.get("metadata", "{}"))
        )
    
    def _reconstruct_dataset(self, dataset_data: Dict[str, Any]) -> Dataset:
        """Reconstruct Dataset object from dictionary data."""
        from ..models import Dataset, Document, DocumentMetadata, DatasetMetadata, QualityMetrics
        
        # Reconstruct documents
        documents = []
        for doc_data in dataset_data.get("documents", []):
            # Reconstruct metadata
            metadata_data = doc_data["metadata"]
            metadata = DocumentMetadata(
                file_type=metadata_data["file_type"],
                size_bytes=metadata_data["size_bytes"],
                language=metadata_data["language"],
                author=metadata_data.get("author"),
                creation_date=datetime.fromisoformat(metadata_data["creation_date"]) if metadata_data.get("creation_date") else None,
                modification_date=datetime.fromisoformat(metadata_data["modification_date"]) if metadata_data.get("modification_date") else None,
                domain=metadata_data.get("domain", "uncategorized"),
                topics=metadata_data.get("topics", []),
                source_url=metadata_data.get("source_url"),
                encoding=metadata_data.get("encoding", "utf-8"),
                quality_score=metadata_data.get("quality_score", 0.0)
            )
            
            # Reconstruct document
            document = Document(
                id=doc_data["id"],
                source_path=doc_data["source_path"],
                content=doc_data["content"],
                metadata=metadata,
                processing_timestamp=datetime.fromisoformat(doc_data["processing_timestamp"]),
                version=doc_data.get("version", "1.0")
            )
            documents.append(document)
        
        # Reconstruct dataset metadata
        dataset_metadata_data = dataset_data.get("metadata", {})
        dataset_metadata = DatasetMetadata(
            creation_date=datetime.fromisoformat(dataset_metadata_data["creation_date"]) if dataset_metadata_data.get("creation_date") else datetime.now(),
            last_modified=datetime.fromisoformat(dataset_metadata_data["last_modified"]) if dataset_metadata_data.get("last_modified") else datetime.now(),
            description=dataset_metadata_data.get("description"),
            tags=dataset_metadata_data.get("tags", []),
            source=dataset_metadata_data.get("source"),
            license=dataset_metadata_data.get("license")
        )
        
        # Reconstruct quality metrics
        quality_data = dataset_data.get("quality_metrics", {})
        quality_metrics = QualityMetrics(
            overall_score=quality_data.get("overall_score", 0.0),
            length_score=quality_data.get("length_score", 0.0),
            language_score=quality_data.get("language_score", 0.0),
            coherence_score=quality_data.get("coherence_score", 0.0),
            uniqueness_score=quality_data.get("uniqueness_score", 0.0),
            completeness_score=quality_data.get("completeness_score", 0.0)
        )
        
        # Reconstruct dataset
        dataset = Dataset(
            id=dataset_data["id"],
            name=dataset_data["name"],
            version=dataset_data["version"],
            documents=documents,
            metadata=dataset_metadata,
            quality_metrics=quality_metrics
        )
        
        return dataset
    
    def _delete_version(self, version_id: str) -> bool:
        """Delete a version and associated data."""
        try:
            with self.connector.get_connection() as conn:
                # Delete change log entries
                conn.execute("DELETE FROM change_log WHERE version_id = ?", (version_id,))
                
                # Get and delete snapshots
                snapshot_query = "SELECT data_path FROM data_snapshots WHERE version_id = ?"
                result = conn.execute(snapshot_query, (version_id,))
                
                for row in result:
                    snapshot_path = Path(row[0])
                    if snapshot_path.exists():
                        snapshot_path.unlink()
                
                # Delete snapshot records
                conn.execute("DELETE FROM data_snapshots WHERE version_id = ?", (version_id,))
                
                # Delete version record
                conn.execute("DELETE FROM data_versions WHERE version_id = ?", (version_id,))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False