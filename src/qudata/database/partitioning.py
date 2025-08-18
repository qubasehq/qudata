"""
Partition manager for handling large datasets efficiently.

This module provides partitioning capabilities to manage large datasets
by splitting them into manageable chunks based on various strategies.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import math

from ..models import Dataset, Document

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """Partitioning strategies."""
    SIZE_BASED = "size_based"          # Partition by size
    COUNT_BASED = "count_based"        # Partition by document count
    DATE_BASED = "date_based"          # Partition by date
    DOMAIN_BASED = "domain_based"      # Partition by domain
    LANGUAGE_BASED = "language_based"  # Partition by language
    HASH_BASED = "hash_based"          # Partition by hash
    QUALITY_BASED = "quality_based"    # Partition by quality score


class PartitionStatus(Enum):
    """Status of a partition."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPACTING = "compacting"
    MERGING = "merging"
    SPLITTING = "splitting"


@dataclass
class PartitionInfo:
    """Information about a partition."""
    partition_id: str
    dataset_id: str
    strategy: PartitionStrategy
    partition_key: str
    document_count: int = 0
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    status: PartitionStatus = PartitionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert partition info to dictionary."""
        return {
            "partition_id": self.partition_id,
            "dataset_id": self.dataset_id,
            "strategy": self.strategy.value,
            "partition_key": self.partition_key,
            "document_count": self.document_count,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata
        }


@dataclass
class PartitionConfig:
    """Configuration for partitioning."""
    strategy: PartitionStrategy
    max_partition_size_bytes: int = 100 * 1024 * 1024  # 100MB
    max_partition_count: int = 1000
    min_partition_size_bytes: int = 1024 * 1024  # 1MB
    min_partition_count: int = 10
    auto_merge: bool = True
    auto_split: bool = True
    compression_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PartitionManager:
    """Manages dataset partitioning for efficient handling of large datasets."""
    
    def __init__(self, database_connector, storage_path: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize partition manager.
        
        Args:
            database_connector: DatabaseConnector instance
            storage_path: Path for storing partition data
            config: Configuration for partitioning
        """
        self.connector = database_connector
        self.storage_path = Path(storage_path or "data/partitions")
        self.config = config or {}
        
        # Default partition configurations
        self.default_configs = {
            PartitionStrategy.SIZE_BASED: PartitionConfig(
                strategy=PartitionStrategy.SIZE_BASED,
                max_partition_size_bytes=100 * 1024 * 1024  # 100MB
            ),
            PartitionStrategy.COUNT_BASED: PartitionConfig(
                strategy=PartitionStrategy.COUNT_BASED,
                max_partition_count=1000
            ),
            PartitionStrategy.DATE_BASED: PartitionConfig(
                strategy=PartitionStrategy.DATE_BASED,
                metadata={"date_field": "processing_timestamp", "interval": "month"}
            ),
            PartitionStrategy.DOMAIN_BASED: PartitionConfig(
                strategy=PartitionStrategy.DOMAIN_BASED
            ),
            PartitionStrategy.LANGUAGE_BASED: PartitionConfig(
                strategy=PartitionStrategy.LANGUAGE_BASED
            ),
            PartitionStrategy.HASH_BASED: PartitionConfig(
                strategy=PartitionStrategy.HASH_BASED,
                metadata={"hash_buckets": 16}
            ),
            PartitionStrategy.QUALITY_BASED: PartitionConfig(
                strategy=PartitionStrategy.QUALITY_BASED,
                metadata={"quality_thresholds": [0.3, 0.6, 0.8]}
            )
        }
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize partitioning tables
        self._initialize_partitioning_tables()
    
    def _initialize_partitioning_tables(self) -> None:
        """Initialize database tables for partitioning."""
        try:
            with self.connector.get_connection() as conn:
                # Partitions table
                partitions_sql = """
                CREATE TABLE IF NOT EXISTS partitions (
                    partition_id VARCHAR(255) PRIMARY KEY,
                    dataset_id VARCHAR(255) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    partition_key VARCHAR(255) NOT NULL,
                    document_count INTEGER DEFAULT 0,
                    size_bytes BIGINT DEFAULT 0,
                    created_at TIMESTAMP NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    status VARCHAR(20) DEFAULT 'active',
                    metadata TEXT
                )
                """
                
                # Partition documents table (maps documents to partitions)
                partition_docs_sql = """
                CREATE TABLE IF NOT EXISTS partition_documents (
                    partition_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255) NOT NULL,
                    added_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (partition_id, document_id)
                )
                """
                
                # Partition operations log
                operations_sql = """
                CREATE TABLE IF NOT EXISTS partition_operations (
                    operation_id VARCHAR(255) PRIMARY KEY,
                    partition_id VARCHAR(255) NOT NULL,
                    operation_type VARCHAR(50) NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    details TEXT,
                    metadata TEXT
                )
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    conn.execute(sqlalchemy.text(partitions_sql))
                    conn.execute(sqlalchemy.text(partition_docs_sql))
                    conn.execute(sqlalchemy.text(operations_sql))
                else:
                    conn.execute(partitions_sql)
                    conn.execute(partition_docs_sql)
                    conn.execute(operations_sql)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to initialize partitioning tables: {e}")
    
    def create_partitions(self, dataset: Dataset, strategy: PartitionStrategy,
                         config: PartitionConfig = None) -> List[str]:
        """
        Create partitions for a dataset.
        
        Args:
            dataset: Dataset to partition
            strategy: Partitioning strategy to use
            config: Partition configuration (uses default if None)
            
        Returns:
            List of created partition IDs
        """
        if config is None:
            config = self.default_configs.get(strategy, PartitionConfig(strategy=strategy))
        
        logger.info(f"Creating partitions for dataset {dataset.id} using {strategy.value} strategy")
        
        try:
            if strategy == PartitionStrategy.SIZE_BASED:
                partitions = self._create_size_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.COUNT_BASED:
                partitions = self._create_count_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.DATE_BASED:
                partitions = self._create_date_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.DOMAIN_BASED:
                partitions = self._create_domain_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.LANGUAGE_BASED:
                partitions = self._create_language_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.HASH_BASED:
                partitions = self._create_hash_based_partitions(dataset, config)
            elif strategy == PartitionStrategy.QUALITY_BASED:
                partitions = self._create_quality_based_partitions(dataset, config)
            else:
                raise ValueError(f"Unsupported partitioning strategy: {strategy}")
            
            # Store partitions in database
            partition_ids = []
            for partition_info, documents in partitions:
                partition_id = self._store_partition(partition_info, documents)
                partition_ids.append(partition_id)
            
            logger.info(f"Created {len(partition_ids)} partitions for dataset {dataset.id}")
            return partition_ids
            
        except Exception as e:
            logger.error(f"Failed to create partitions: {e}")
            raise
    
    def get_partition_info(self, partition_id: str) -> Optional[PartitionInfo]:
        """
        Get information about a partition.
        
        Args:
            partition_id: ID of the partition
            
        Returns:
            PartitionInfo object if found
        """
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT * FROM partitions WHERE partition_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (partition_id,))
                
                row = result.fetchone()
                if row:
                    return self._row_to_partition_info(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get partition info for {partition_id}: {e}")
            return None
    
    def list_partitions(self, dataset_id: str = None, 
                       strategy: PartitionStrategy = None) -> List[PartitionInfo]:
        """
        List partitions.
        
        Args:
            dataset_id: Filter by dataset ID (optional)
            strategy: Filter by strategy (optional)
            
        Returns:
            List of partition information
        """
        try:
            with self.connector.get_connection() as conn:
                query_parts = ["SELECT * FROM partitions WHERE 1=1"]
                params = []
                
                if dataset_id:
                    query_parts.append("AND dataset_id = ?")
                    params.append(dataset_id)
                
                if strategy:
                    query_parts.append("AND strategy = ?")
                    params.append(strategy.value)
                
                query_parts.append("ORDER BY created_at")
                query = " ".join(query_parts)
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), params)
                
                partitions = []
                for row in result:
                    partition_info = self._row_to_partition_info(row)
                    partitions.append(partition_info)
                
                return partitions
                
        except Exception as e:
            logger.error(f"Failed to list partitions: {e}")
            return []
    
    def get_partition_documents(self, partition_id: str) -> List[str]:
        """
        Get document IDs in a partition.
        
        Args:
            partition_id: ID of the partition
            
        Returns:
            List of document IDs
        """
        try:
            with self.connector.get_connection() as conn:
                query = "SELECT document_id FROM partition_documents WHERE partition_id = ?"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), (partition_id,))
                
                return [row[0] for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get partition documents for {partition_id}: {e}")
            return []
    
    def merge_partitions(self, partition_ids: List[str], 
                        new_partition_id: str = None) -> Optional[str]:
        """
        Merge multiple partitions into one.
        
        Args:
            partition_ids: List of partition IDs to merge
            new_partition_id: ID for the new partition (generated if None)
            
        Returns:
            ID of the merged partition
        """
        if len(partition_ids) < 2:
            raise ValueError("At least 2 partitions required for merging")
        
        if new_partition_id is None:
            new_partition_id = self._generate_partition_id()
        
        logger.info(f"Merging {len(partition_ids)} partitions into {new_partition_id}")
        
        try:
            # Get partition information
            partitions = []
            for partition_id in partition_ids:
                partition_info = self.get_partition_info(partition_id)
                if not partition_info:
                    raise ValueError(f"Partition {partition_id} not found")
                partitions.append(partition_info)
            
            # Validate that partitions can be merged
            if not self._can_merge_partitions(partitions):
                raise ValueError("Partitions cannot be merged (incompatible strategies or datasets)")
            
            # Create merged partition info
            merged_partition = PartitionInfo(
                partition_id=new_partition_id,
                dataset_id=partitions[0].dataset_id,
                strategy=partitions[0].strategy,
                partition_key=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_count=sum(p.document_count for p in partitions),
                size_bytes=sum(p.size_bytes for p in partitions),
                status=PartitionStatus.ACTIVE,
                metadata={"merged_from": partition_ids}
            )
            
            # Get all documents from source partitions
            all_documents = []
            for partition_id in partition_ids:
                documents = self.get_partition_documents(partition_id)
                all_documents.extend(documents)
            
            # Store merged partition
            self._store_partition(merged_partition, all_documents)
            
            # Mark source partitions as archived
            for partition_id in partition_ids:
                self._update_partition_status(partition_id, PartitionStatus.ARCHIVED)
            
            logger.info(f"Successfully merged partitions into {new_partition_id}")
            return new_partition_id
            
        except Exception as e:
            logger.error(f"Failed to merge partitions: {e}")
            return None
    
    def split_partition(self, partition_id: str, strategy: PartitionStrategy = None,
                       config: PartitionConfig = None) -> List[str]:
        """
        Split a partition into smaller partitions.
        
        Args:
            partition_id: ID of the partition to split
            strategy: Strategy for splitting (uses original if None)
            config: Configuration for splitting
            
        Returns:
            List of new partition IDs
        """
        logger.info(f"Splitting partition {partition_id}")
        
        try:
            # Get partition info
            partition_info = self.get_partition_info(partition_id)
            if not partition_info:
                raise ValueError(f"Partition {partition_id} not found")
            
            # Get documents in partition
            document_ids = self.get_partition_documents(partition_id)
            if not document_ids:
                logger.warning(f"No documents found in partition {partition_id}")
                return []
            
            # Load documents
            documents = self._load_documents(document_ids)
            
            # Create temporary dataset for splitting
            temp_dataset = Dataset(
                id=f"temp_{partition_id}",
                name=f"Temp dataset for splitting {partition_id}",
                version="1.0",
                documents=documents
            )
            
            # Use original strategy if not specified
            if strategy is None:
                strategy = partition_info.strategy
            
            # Create new partitions
            new_partition_ids = self.create_partitions(temp_dataset, strategy, config)
            
            # Mark original partition as archived
            self._update_partition_status(partition_id, PartitionStatus.ARCHIVED)
            
            logger.info(f"Split partition {partition_id} into {len(new_partition_ids)} new partitions")
            return new_partition_ids
            
        except Exception as e:
            logger.error(f"Failed to split partition {partition_id}: {e}")
            return []
    
    def optimize_partitions(self, dataset_id: str) -> Dict[str, Any]:
        """
        Optimize partitions for a dataset.
        
        Args:
            dataset_id: ID of the dataset to optimize
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing partitions for dataset {dataset_id}")
        
        results = {
            "merged_partitions": 0,
            "split_partitions": 0,
            "total_partitions_before": 0,
            "total_partitions_after": 0,
            "operations": []
        }
        
        try:
            # Get all partitions for dataset
            partitions = self.list_partitions(dataset_id=dataset_id)
            results["total_partitions_before"] = len(partitions)
            
            # Group partitions by strategy
            strategy_groups = {}
            for partition in partitions:
                if partition.strategy not in strategy_groups:
                    strategy_groups[partition.strategy] = []
                strategy_groups[partition.strategy].append(partition)
            
            # Optimize each strategy group
            for strategy, strategy_partitions in strategy_groups.items():
                config = self.default_configs.get(strategy)
                if not config:
                    continue
                
                # Find partitions that need merging (too small)
                small_partitions = [
                    p for p in strategy_partitions
                    if (p.size_bytes < config.min_partition_size_bytes or
                        p.document_count < config.min_partition_count)
                    and p.status == PartitionStatus.ACTIVE
                ]
                
                # Find partitions that need splitting (too large)
                large_partitions = [
                    p for p in strategy_partitions
                    if (p.size_bytes > config.max_partition_size_bytes or
                        p.document_count > config.max_partition_count)
                    and p.status == PartitionStatus.ACTIVE
                ]
                
                # Merge small partitions
                if len(small_partitions) >= 2 and config.auto_merge:
                    # Group small partitions for merging
                    merge_groups = self._group_partitions_for_merging(small_partitions, config)
                    
                    for merge_group in merge_groups:
                        if len(merge_group) >= 2:
                            partition_ids = [p.partition_id for p in merge_group]
                            merged_id = self.merge_partitions(partition_ids)
                            if merged_id:
                                results["merged_partitions"] += 1
                                results["operations"].append({
                                    "type": "merge",
                                    "source_partitions": partition_ids,
                                    "result_partition": merged_id
                                })
                
                # Split large partitions
                if large_partitions and config.auto_split:
                    for partition in large_partitions:
                        new_partition_ids = self.split_partition(partition.partition_id, strategy, config)
                        if new_partition_ids:
                            results["split_partitions"] += 1
                            results["operations"].append({
                                "type": "split",
                                "source_partition": partition.partition_id,
                                "result_partitions": new_partition_ids
                            })
            
            # Get final partition count
            final_partitions = self.list_partitions(dataset_id=dataset_id)
            active_partitions = [p for p in final_partitions if p.status == PartitionStatus.ACTIVE]
            results["total_partitions_after"] = len(active_partitions)
            
            logger.info(f"Partition optimization completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to optimize partitions: {e}")
            return results
    
    def get_partition_statistics(self, dataset_id: str = None) -> Dict[str, Any]:
        """
        Get partition statistics.
        
        Args:
            dataset_id: Filter by dataset ID (optional)
            
        Returns:
            Partition statistics
        """
        try:
            partitions = self.list_partitions(dataset_id=dataset_id)
            
            # Calculate statistics
            stats = {
                "total_partitions": len(partitions),
                "active_partitions": len([p for p in partitions if p.status == PartitionStatus.ACTIVE]),
                "archived_partitions": len([p for p in partitions if p.status == PartitionStatus.ARCHIVED]),
                "total_documents": sum(p.document_count for p in partitions),
                "total_size_bytes": sum(p.size_bytes for p in partitions),
                "average_partition_size": 0,
                "average_documents_per_partition": 0,
                "strategy_distribution": {},
                "size_distribution": {
                    "small": 0,    # < 10MB
                    "medium": 0,   # 10MB - 100MB
                    "large": 0,    # 100MB - 1GB
                    "xlarge": 0    # > 1GB
                }
            }
            
            if partitions:
                stats["average_partition_size"] = stats["total_size_bytes"] / len(partitions)
                stats["average_documents_per_partition"] = stats["total_documents"] / len(partitions)
                
                # Strategy distribution
                for partition in partitions:
                    strategy = partition.strategy.value
                    stats["strategy_distribution"][strategy] = stats["strategy_distribution"].get(strategy, 0) + 1
                
                # Size distribution
                for partition in partitions:
                    size_mb = partition.size_bytes / (1024 * 1024)
                    if size_mb < 10:
                        stats["size_distribution"]["small"] += 1
                    elif size_mb < 100:
                        stats["size_distribution"]["medium"] += 1
                    elif size_mb < 1024:
                        stats["size_distribution"]["large"] += 1
                    else:
                        stats["size_distribution"]["xlarge"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get partition statistics: {e}")
            return {}
    
    def cleanup_archived_partitions(self, older_than_days: int = 30) -> int:
        """
        Clean up archived partitions older than specified days.
        
        Args:
            older_than_days: Delete partitions archived more than this many days ago
            
        Returns:
            Number of partitions cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            with self.connector.get_connection() as conn:
                # Get archived partitions older than cutoff
                query = """
                SELECT partition_id FROM partitions 
                WHERE status = 'archived' AND last_modified < ?
                """
                
                if self.connector.config.connection_type == "postgresql":
                    import sqlalchemy
                    result = conn.execute(sqlalchemy.text(query), (cutoff_date.isoformat(),))
                else:
                    result = conn.execute(query, (cutoff_date.isoformat(),))
                
                old_partitions = [row[0] for row in result]
                
                # Delete old partitions
                deleted_count = 0
                for partition_id in old_partitions:
                    if self._delete_partition(partition_id):
                        deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} archived partitions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup archived partitions: {e}")
            return 0
    
    def _create_size_based_partitions(self, dataset: Dataset, 
                                    config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create size-based partitions."""
        partitions = []
        current_partition_docs = []
        current_size = 0
        partition_num = 1
        
        for document in dataset.documents:
            doc_size = len(document.content.encode('utf-8'))
            
            # Check if adding this document would exceed the size limit
            if (current_size + doc_size > config.max_partition_size_bytes and 
                current_partition_docs):
                
                # Create partition with current documents
                partition_info = PartitionInfo(
                    partition_id=f"{dataset.id}_size_{partition_num}",
                    dataset_id=dataset.id,
                    strategy=PartitionStrategy.SIZE_BASED,
                    partition_key=f"size_partition_{partition_num}",
                    document_count=len(current_partition_docs),
                    size_bytes=current_size
                )
                
                partitions.append((partition_info, current_partition_docs.copy()))
                
                # Reset for next partition
                current_partition_docs = []
                current_size = 0
                partition_num += 1
            
            current_partition_docs.append(document.id)
            current_size += doc_size
        
        # Create final partition if there are remaining documents
        if current_partition_docs:
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_size_{partition_num}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.SIZE_BASED,
                partition_key=f"size_partition_{partition_num}",
                document_count=len(current_partition_docs),
                size_bytes=current_size
            )
            partitions.append((partition_info, current_partition_docs))
        
        return partitions
    
    def _create_count_based_partitions(self, dataset: Dataset, 
                                     config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create count-based partitions."""
        partitions = []
        documents = [doc.id for doc in dataset.documents]
        
        # Calculate number of partitions needed
        total_docs = len(documents)
        docs_per_partition = config.max_partition_count
        num_partitions = math.ceil(total_docs / docs_per_partition)
        
        for i in range(num_partitions):
            start_idx = i * docs_per_partition
            end_idx = min((i + 1) * docs_per_partition, total_docs)
            partition_docs = documents[start_idx:end_idx]
            
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8')) 
                for doc in dataset.documents[start_idx:end_idx]
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_count_{i+1}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.COUNT_BASED,
                partition_key=f"count_partition_{i+1}",
                document_count=len(partition_docs),
                size_bytes=partition_size
            )
            
            partitions.append((partition_info, partition_docs))
        
        return partitions
    
    def _create_date_based_partitions(self, dataset: Dataset, 
                                    config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create date-based partitions."""
        partitions = {}
        date_field = config.metadata.get("date_field", "processing_timestamp")
        interval = config.metadata.get("interval", "month")
        
        for document in dataset.documents:
            # Get date from document
            if hasattr(document, date_field):
                doc_date = getattr(document, date_field)
            else:
                doc_date = document.processing_timestamp
            
            # Create partition key based on interval
            if interval == "day":
                partition_key = doc_date.strftime("%Y-%m-%d")
            elif interval == "week":
                partition_key = f"{doc_date.year}-W{doc_date.isocalendar()[1]:02d}"
            elif interval == "month":
                partition_key = doc_date.strftime("%Y-%m")
            elif interval == "year":
                partition_key = doc_date.strftime("%Y")
            else:
                partition_key = doc_date.strftime("%Y-%m")
            
            if partition_key not in partitions:
                partitions[partition_key] = []
            
            partitions[partition_key].append(document.id)
        
        # Convert to partition info objects
        result = []
        for partition_key, doc_ids in partitions.items():
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8'))
                for doc in dataset.documents
                if doc.id in doc_ids
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_date_{partition_key}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.DATE_BASED,
                partition_key=partition_key,
                document_count=len(doc_ids),
                size_bytes=partition_size
            )
            
            result.append((partition_info, doc_ids))
        
        return result
    
    def _create_domain_based_partitions(self, dataset: Dataset, 
                                      config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create domain-based partitions."""
        partitions = {}
        
        for document in dataset.documents:
            domain = document.metadata.domain
            
            if domain not in partitions:
                partitions[domain] = []
            
            partitions[domain].append(document.id)
        
        # Convert to partition info objects
        result = []
        for domain, doc_ids in partitions.items():
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8'))
                for doc in dataset.documents
                if doc.id in doc_ids
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_domain_{domain}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.DOMAIN_BASED,
                partition_key=domain,
                document_count=len(doc_ids),
                size_bytes=partition_size
            )
            
            result.append((partition_info, doc_ids))
        
        return result
    
    def _create_language_based_partitions(self, dataset: Dataset, 
                                        config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create language-based partitions."""
        partitions = {}
        
        for document in dataset.documents:
            language = document.metadata.language
            
            if language not in partitions:
                partitions[language] = []
            
            partitions[language].append(document.id)
        
        # Convert to partition info objects
        result = []
        for language, doc_ids in partitions.items():
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8'))
                for doc in dataset.documents
                if doc.id in doc_ids
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_lang_{language}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.LANGUAGE_BASED,
                partition_key=language,
                document_count=len(doc_ids),
                size_bytes=partition_size
            )
            
            result.append((partition_info, doc_ids))
        
        return result
    
    def _create_hash_based_partitions(self, dataset: Dataset, 
                                    config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create hash-based partitions."""
        hash_buckets = config.metadata.get("hash_buckets", 16)
        partitions = {i: [] for i in range(hash_buckets)}
        
        for document in dataset.documents:
            # Hash document ID to determine partition
            doc_hash = hashlib.md5(document.id.encode()).hexdigest()
            bucket = int(doc_hash, 16) % hash_buckets
            partitions[bucket].append(document.id)
        
        # Convert to partition info objects
        result = []
        for bucket, doc_ids in partitions.items():
            if not doc_ids:  # Skip empty partitions
                continue
                
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8'))
                for doc in dataset.documents
                if doc.id in doc_ids
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_hash_{bucket:02d}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.HASH_BASED,
                partition_key=f"hash_bucket_{bucket:02d}",
                document_count=len(doc_ids),
                size_bytes=partition_size
            )
            
            result.append((partition_info, doc_ids))
        
        return result
    
    def _create_quality_based_partitions(self, dataset: Dataset, 
                                       config: PartitionConfig) -> List[Tuple[PartitionInfo, List[str]]]:
        """Create quality-based partitions."""
        thresholds = config.metadata.get("quality_thresholds", [0.3, 0.6, 0.8])
        thresholds = sorted(thresholds)
        
        # Create quality ranges
        ranges = []
        for i, threshold in enumerate(thresholds):
            if i == 0:
                ranges.append((0.0, threshold, f"low_quality_0_{threshold}"))
            else:
                ranges.append((thresholds[i-1], threshold, f"medium_quality_{thresholds[i-1]}_{threshold}"))
        
        # Add high quality range
        if thresholds:
            ranges.append((thresholds[-1], 1.0, f"high_quality_{thresholds[-1]}_1.0"))
        else:
            ranges.append((0.0, 1.0, "all_quality"))
        
        partitions = {range_key: [] for _, _, range_key in ranges}
        
        for document in dataset.documents:
            quality_score = document.metadata.quality_score
            
            # Find appropriate quality range
            for min_qual, max_qual, range_key in ranges:
                if min_qual <= quality_score < max_qual or (max_qual == 1.0 and quality_score == 1.0):
                    partitions[range_key].append(document.id)
                    break
        
        # Convert to partition info objects
        result = []
        for range_key, doc_ids in partitions.items():
            if not doc_ids:  # Skip empty partitions
                continue
                
            # Calculate size
            partition_size = sum(
                len(doc.content.encode('utf-8'))
                for doc in dataset.documents
                if doc.id in doc_ids
            )
            
            partition_info = PartitionInfo(
                partition_id=f"{dataset.id}_quality_{range_key}",
                dataset_id=dataset.id,
                strategy=PartitionStrategy.QUALITY_BASED,
                partition_key=range_key,
                document_count=len(doc_ids),
                size_bytes=partition_size
            )
            
            result.append((partition_info, doc_ids))
        
        return result
    
    def _store_partition(self, partition_info: PartitionInfo, document_ids: List[str]) -> str:
        """Store partition in database."""
        try:
            with self.connector.get_connection() as conn:
                # Store partition info
                partition_query = """
                INSERT INTO partitions 
                (partition_id, dataset_id, strategy, partition_key, document_count, 
                 size_bytes, created_at, last_modified, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                partition_params = (
                    partition_info.partition_id,
                    partition_info.dataset_id,
                    partition_info.strategy.value,
                    partition_info.partition_key,
                    partition_info.document_count,
                    partition_info.size_bytes,
                    partition_info.created_at.isoformat(),
                    partition_info.last_modified.isoformat(),
                    partition_info.status.value,
                    json.dumps(partition_info.metadata)
                )
                
                import sqlalchemy
                conn.execute(sqlalchemy.text(partition_query), partition_params)
                
                # Store document mappings
                doc_query = """
                INSERT INTO partition_documents (partition_id, document_id, added_at)
                VALUES (?, ?, ?)
                """
                
                for doc_id in document_ids:
                    doc_params = (
                        partition_info.partition_id,
                        doc_id,
                        datetime.now().isoformat()
                    )
                    
                    if self.connector.config.connection_type == "postgresql":
                        conn.execute(sqlalchemy.text(doc_query), doc_params)
                    else:
                        conn.execute(doc_query, doc_params)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return partition_info.partition_id
                
        except Exception as e:
            logger.error(f"Failed to store partition: {e}")
            raise
    
    def _generate_partition_id(self) -> str:
        """Generate a unique partition ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"part_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _row_to_partition_info(self, row) -> PartitionInfo:
        """Convert database row to PartitionInfo object."""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        return PartitionInfo(
            partition_id=row_dict["partition_id"],
            dataset_id=row_dict["dataset_id"],
            strategy=PartitionStrategy(row_dict["strategy"]),
            partition_key=row_dict["partition_key"],
            document_count=row_dict.get("document_count", 0),
            size_bytes=row_dict.get("size_bytes", 0),
            created_at=datetime.fromisoformat(row_dict["created_at"]),
            last_modified=datetime.fromisoformat(row_dict["last_modified"]),
            status=PartitionStatus(row_dict.get("status", "active")),
            metadata=json.loads(row_dict.get("metadata", "{}"))
        )
    
    def _can_merge_partitions(self, partitions: List[PartitionInfo]) -> bool:
        """Check if partitions can be merged."""
        if not partitions:
            return False
        
        # All partitions must have same dataset and strategy
        first_partition = partitions[0]
        for partition in partitions[1:]:
            if (partition.dataset_id != first_partition.dataset_id or
                partition.strategy != first_partition.strategy):
                return False
        
        return True
    
    def _group_partitions_for_merging(self, partitions: List[PartitionInfo], 
                                    config: PartitionConfig) -> List[List[PartitionInfo]]:
        """Group small partitions for merging."""
        groups = []
        current_group = []
        current_size = 0
        
        # Sort partitions by size
        sorted_partitions = sorted(partitions, key=lambda p: p.size_bytes)
        
        for partition in sorted_partitions:
            # Check if adding this partition would exceed the max size
            if (current_size + partition.size_bytes > config.max_partition_size_bytes and 
                current_group):
                
                groups.append(current_group)
                current_group = []
                current_size = 0
            
            current_group.append(partition)
            current_size += partition.size_bytes
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _load_documents(self, document_ids: List[str]) -> List[Document]:
        """Load documents from database."""
        documents = []
        
        try:
            with self.connector.get_connection() as conn:
                # Create placeholders for IN clause
                placeholders = ",".join(["?" for _ in document_ids])
                query = f"SELECT * FROM documents WHERE id IN ({placeholders})"
                
                import sqlalchemy
                result = conn.execute(sqlalchemy.text(query), document_ids)
                
                for row in result:
                    # Convert row to Document object
                    # This is a simplified version - in practice you'd need full reconstruction
                    row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                    
                    from ..models import Document, DocumentMetadata
                    
                    metadata = DocumentMetadata(
                        file_type=row_dict["file_type"],
                        size_bytes=row_dict["size_bytes"],
                        language=row_dict["language"],
                        domain=row_dict.get("domain", "uncategorized"),
                        quality_score=row_dict.get("quality_score", 0.0)
                    )
                    
                    document = Document(
                        id=row_dict["id"],
                        source_path=row_dict["source_path"],
                        content=row_dict["content"],
                        metadata=metadata,
                        processing_timestamp=datetime.fromisoformat(row_dict["processing_timestamp"]),
                        version=row_dict.get("version", "1.0")
                    )
                    
                    documents.append(document)
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return []
    
    def _update_partition_status(self, partition_id: str, status: PartitionStatus) -> None:
        """Update partition status."""
        try:
            with self.connector.get_connection() as conn:
                query = """
                UPDATE partitions 
                SET status = ?, last_modified = ?
                WHERE partition_id = ?
                """
                
                params = (status.value, datetime.now().isoformat(), partition_id)
                
                import sqlalchemy
                conn.execute(sqlalchemy.text(query), params)
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update partition status: {e}")
    
    def _delete_partition(self, partition_id: str) -> bool:
        """Delete a partition and its data."""
        try:
            with self.connector.get_connection() as conn:
                # Delete partition documents
                conn.execute("DELETE FROM partition_documents WHERE partition_id = ?", (partition_id,))
                
                # Delete partition operations
                conn.execute("DELETE FROM partition_operations WHERE partition_id = ?", (partition_id,))
                
                # Delete partition
                conn.execute("DELETE FROM partitions WHERE partition_id = ?", (partition_id,))
                
                if hasattr(conn, 'commit'):
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete partition {partition_id}: {e}")
            return False