"""
Deduplication engine for exact and near-duplicate detection.

This module provides functionality to detect and remove duplicate content
using various similarity algorithms and configurable thresholds.
"""

import hashlib
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from ..models import ProcessingError, ErrorSeverity


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate documents."""
    representative_id: str  # ID of the document to keep
    duplicate_ids: List[str] = field(default_factory=list)  # IDs of duplicates to remove
    similarity_scores: Dict[str, float] = field(default_factory=dict)  # Similarity scores
    duplicate_type: str = "exact"  # Type of duplication: exact, near, fuzzy
    
    def get_all_ids(self) -> List[str]:
        """Get all document IDs in this group."""
        return [self.representative_id] + self.duplicate_ids
    
    def get_duplicate_count(self) -> int:
        """Get the number of duplicates (excluding representative)."""
        return len(self.duplicate_ids)


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    original_count: int
    unique_count: int
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    removed_ids: Set[str] = field(default_factory=set)
    
    def get_duplicate_count(self) -> int:
        """Get total number of duplicates removed."""
        return len(self.removed_ids)
    
    def get_deduplication_ratio(self) -> float:
        """Get ratio of duplicates removed."""
        if self.original_count == 0:
            return 0.0
        return self.get_duplicate_count() / self.original_count


class DeduplicationEngine:
    """
    Engine for detecting and removing duplicate content.
    
    Supports exact matching, near-duplicate detection using text similarity,
    and fuzzy matching with configurable thresholds.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DeduplicationEngine.
        
        Args:
            config: Configuration dictionary with deduplication settings
        """
        self.config = config or {}
        
        # Load deduplication-specific config if available
        dedup_config = self.config.get('deduplication', {})
        
        # Similarity thresholds
        self.exact_match_threshold = dedup_config.get('exact_match_threshold', 
                                                     self.config.get('exact_match_threshold', 1.0))
        self.near_duplicate_threshold = dedup_config.get('near_duplicate_threshold', 
                                                        self.config.get('near_duplicate_threshold', 0.85))
        self.fuzzy_match_threshold = dedup_config.get('fuzzy_match_threshold', 
                                                     self.config.get('fuzzy_match_threshold', 0.7))
        
        # Content preprocessing settings
        self.normalize_whitespace = dedup_config.get('normalize_whitespace', 
                                                    self.config.get('normalize_whitespace', True))
        self.case_sensitive = dedup_config.get('case_sensitive', 
                                              self.config.get('case_sensitive', False))
        self.ignore_punctuation = dedup_config.get('ignore_punctuation', 
                                                  self.config.get('ignore_punctuation', True))
        self.min_content_length = dedup_config.get('min_content_length', 
                                                  self.config.get('min_content_length', 50))
        
        # Performance settings
        self.use_hashing = dedup_config.get('use_hashing', 
                                           self.config.get('use_hashing', True))
        self.chunk_size = dedup_config.get('chunk_size', 
                                          self.config.get('chunk_size', 1000))
        
        # Caches for performance
        self._content_hashes: Dict[str, str] = {}
        self._normalized_content: Dict[str, str] = {}
    
    def deduplicate_documents(self, documents: Dict[str, str]) -> DeduplicationResult:
        """
        Deduplicate a collection of documents.
        
        Args:
            documents: Dictionary mapping document IDs to content
            
        Returns:
            DeduplicationResult with duplicate groups and statistics
        """
        if not documents:
            return DeduplicationResult(original_count=0, unique_count=0)
        
        result = DeduplicationResult(
            original_count=len(documents),
            unique_count=len(documents)
        )
        
        try:
            # Filter out documents that are too short
            filtered_docs = self._filter_short_documents(documents)
            
            # Step 1: Exact duplicate detection using hashes
            exact_groups = self._find_exact_duplicates(filtered_docs)
            result.duplicate_groups.extend(exact_groups)
            
            # Remove exact duplicates from further processing
            remaining_docs = self._remove_processed_documents(filtered_docs, exact_groups)
            
            # Step 2: Near-duplicate detection using similarity
            if self.near_duplicate_threshold < 1.0:
                near_groups = self._find_near_duplicates(remaining_docs)
                result.duplicate_groups.extend(near_groups)
                remaining_docs = self._remove_processed_documents(remaining_docs, near_groups)
            
            # Step 3: Fuzzy matching for very similar content
            if self.fuzzy_match_threshold < self.near_duplicate_threshold:
                fuzzy_groups = self._find_fuzzy_duplicates(remaining_docs)
                result.duplicate_groups.extend(fuzzy_groups)
            
            # Calculate final statistics
            for group in result.duplicate_groups:
                result.removed_ids.update(group.duplicate_ids)
            
            result.unique_count = result.original_count - len(result.removed_ids)
            
        except Exception as e:
            raise ProcessingError(
                stage="deduplication",
                error_type="DeduplicationError",
                message=f"Failed to deduplicate documents: {str(e)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return result
    
    def _filter_short_documents(self, documents: Dict[str, str]) -> Dict[str, str]:
        """Filter out documents that are too short to be meaningful."""
        return {
            doc_id: content for doc_id, content in documents.items()
            if len(content.strip()) >= self.min_content_length
        }
    
    def _find_exact_duplicates(self, documents: Dict[str, str]) -> List[DuplicateGroup]:
        """Find exact duplicates using content hashes."""
        hash_to_docs: Dict[str, List[str]] = {}
        
        for doc_id, content in documents.items():
            content_hash = self._get_content_hash(doc_id, content)
            if content_hash not in hash_to_docs:
                hash_to_docs[content_hash] = []
            hash_to_docs[content_hash].append(doc_id)
        
        # Create duplicate groups for hashes with multiple documents
        duplicate_groups = []
        for doc_ids in hash_to_docs.values():
            if len(doc_ids) > 1:
                # Use the first document as representative
                representative = doc_ids[0]
                duplicates = doc_ids[1:]
                
                group = DuplicateGroup(
                    representative_id=representative,
                    duplicate_ids=duplicates,
                    duplicate_type="exact"
                )
                
                # All have similarity score of 1.0 for exact matches
                for dup_id in duplicates:
                    group.similarity_scores[dup_id] = 1.0
                
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _find_near_duplicates(self, documents: Dict[str, str]) -> List[DuplicateGroup]:
        """Find near-duplicates using text similarity."""
        duplicate_groups = []
        processed_docs = set()
        
        doc_ids = list(documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            if doc_id1 in processed_docs:
                continue
            
            group_duplicates = []
            similarity_scores = {}
            
            content1 = self._get_normalized_content(doc_id1, documents[doc_id1])
            
            for j in range(i + 1, len(doc_ids)):
                doc_id2 = doc_ids[j]
                if doc_id2 in processed_docs:
                    continue
                
                content2 = self._get_normalized_content(doc_id2, documents[doc_id2])
                similarity = self._calculate_similarity(content1, content2)
                
                if similarity >= self.near_duplicate_threshold:
                    group_duplicates.append(doc_id2)
                    similarity_scores[doc_id2] = similarity
                    processed_docs.add(doc_id2)
            
            if group_duplicates:
                group = DuplicateGroup(
                    representative_id=doc_id1,
                    duplicate_ids=group_duplicates,
                    similarity_scores=similarity_scores,
                    duplicate_type="near"
                )
                duplicate_groups.append(group)
                processed_docs.add(doc_id1)
        
        return duplicate_groups
    
    def _find_fuzzy_duplicates(self, documents: Dict[str, str]) -> List[DuplicateGroup]:
        """Find fuzzy duplicates using more lenient similarity matching."""
        duplicate_groups = []
        processed_docs = set()
        
        doc_ids = list(documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            if doc_id1 in processed_docs:
                continue
            
            group_duplicates = []
            similarity_scores = {}
            
            content1 = self._get_normalized_content(doc_id1, documents[doc_id1])
            
            for j in range(i + 1, len(doc_ids)):
                doc_id2 = doc_ids[j]
                if doc_id2 in processed_docs:
                    continue
                
                content2 = self._get_normalized_content(doc_id2, documents[doc_id2])
                similarity = self._calculate_fuzzy_similarity(content1, content2)
                
                if similarity >= self.fuzzy_match_threshold:
                    group_duplicates.append(doc_id2)
                    similarity_scores[doc_id2] = similarity
                    processed_docs.add(doc_id2)
            
            if group_duplicates:
                group = DuplicateGroup(
                    representative_id=doc_id1,
                    duplicate_ids=group_duplicates,
                    similarity_scores=similarity_scores,
                    duplicate_type="fuzzy"
                )
                duplicate_groups.append(group)
                processed_docs.add(doc_id1)
        
        return duplicate_groups
    
    def _remove_processed_documents(self, documents: Dict[str, str], 
                                   groups: List[DuplicateGroup]) -> Dict[str, str]:
        """Remove documents that have been processed in duplicate groups."""
        processed_ids = set()
        for group in groups:
            processed_ids.update(group.get_all_ids())
        
        return {
            doc_id: content for doc_id, content in documents.items()
            if doc_id not in processed_ids
        }
    
    def _get_content_hash(self, doc_id: str, content: str) -> str:
        """Get or compute content hash for a document."""
        if doc_id not in self._content_hashes:
            normalized = self._get_normalized_content(doc_id, content)
            self._content_hashes[doc_id] = hashlib.md5(
                normalized.encode('utf-8')
            ).hexdigest()
        return self._content_hashes[doc_id]
    
    def _get_normalized_content(self, doc_id: str, content: str) -> str:
        """Get or compute normalized content for similarity comparison."""
        if doc_id not in self._normalized_content:
            normalized = content
            
            # Case normalization
            if not self.case_sensitive:
                normalized = normalized.lower()
            
            # Whitespace normalization
            if self.normalize_whitespace:
                normalized = re.sub(r'\s+', ' ', normalized.strip())
            
            # Punctuation removal
            if self.ignore_punctuation:
                normalized = re.sub(r'[^\w\s]', '', normalized)
            
            self._normalized_content[doc_id] = normalized
        
        return self._normalized_content[doc_id]
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        if not content1 or not content2:
            return 0.0
        
        # Use SequenceMatcher for similarity calculation
        matcher = SequenceMatcher(None, content1, content2)
        return matcher.ratio()
    
    def _calculate_fuzzy_similarity(self, content1: str, content2: str) -> float:
        """Calculate fuzzy similarity using word-level comparison."""
        if not content1 or not content2:
            return 0.0
        
        # Split into words and compare
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_duplicate_statistics(self, result: DeduplicationResult) -> Dict[str, Any]:
        """Get detailed statistics about deduplication results."""
        stats = {
            'original_count': result.original_count,
            'unique_count': result.unique_count,
            'duplicate_count': result.get_duplicate_count(),
            'deduplication_ratio': result.get_deduplication_ratio(),
            'duplicate_groups': len(result.duplicate_groups),
            'by_type': {}
        }
        
        # Statistics by duplicate type
        for group in result.duplicate_groups:
            dup_type = group.duplicate_type
            if dup_type not in stats['by_type']:
                stats['by_type'][dup_type] = {
                    'groups': 0,
                    'duplicates_removed': 0,
                    'avg_similarity': 0.0
                }
            
            stats['by_type'][dup_type]['groups'] += 1
            stats['by_type'][dup_type]['duplicates_removed'] += group.get_duplicate_count()
            
            # Calculate average similarity for this type
            if group.similarity_scores:
                avg_sim = sum(group.similarity_scores.values()) / len(group.similarity_scores)
                current_avg = stats['by_type'][dup_type]['avg_similarity']
                group_count = stats['by_type'][dup_type]['groups']
                stats['by_type'][dup_type]['avg_similarity'] = (
                    (current_avg * (group_count - 1) + avg_sim) / group_count
                )
        
        return stats
    
    def clear_cache(self):
        """Clear internal caches to free memory."""
        self._content_hashes.clear()
        self._normalized_content.clear()
