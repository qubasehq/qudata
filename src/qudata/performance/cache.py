"""
Caching layer for expensive operation caching.

Provides multi-level caching with LRU, TTL, and persistent storage
options for optimizing expensive operations like NLP processing.
"""

import hashlib
import json
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_entries: int = 10000
    default_ttl: Optional[float] = 3600  # 1 hour
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_disk_cache: bool = True
    disk_cache_dir: str = ".cache"
    enable_compression: bool = True
    enable_stats: bool = True


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache."""
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used
                self._cache.popitem(last=False)
            
            self._cache[key] = value
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)


class DiskCache:
    """
    Disk-based cache using SQLite for metadata and files for data.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize disk cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self._lock = threading.Lock()
        self._closed = False
        self._init_database()
    
    def close(self):
        """Close the disk cache and cleanup resources."""
        self._closed = True
    
    def _get_connection(self):
        """Get a database connection with proper settings."""
        if self._closed:
            raise RuntimeError("DiskCache has been closed")
        
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
        return conn
    
    def _init_database(self):
        """Initialize SQLite database."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    ttl REAL,
                    size_bytes INTEGER
                )
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT filename, ttl FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    filename, ttl = row
                    
                    # Check TTL
                    if ttl and time.time() > ttl:
                        self.remove(key)
                        return None
                    
                    # Load data from file
                    file_path = self.cache_dir / filename
                    if not file_path.exists():
                        self.remove(key)
                        return None
                    
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Update access time
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                        (time.time(), key)
                    )
                    conn.commit()
                    
                    return data
                    
            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in disk cache."""
        with self._lock:
            try:
                # Generate filename
                filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
                file_path = self.cache_dir / filename
                
                # Save data to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = file_path.stat().st_size
                current_time = time.time()
                ttl_timestamp = current_time + ttl if ttl else None
                
                # Save metadata to database
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, filename, created_at, accessed_at, ttl, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (key, filename, current_time, current_time, ttl_timestamp, size_bytes))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Disk cache put error: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove entry from disk cache."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT filename FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        filename = row[0]
                        file_path = self.cache_dir / filename
                        
                        # Remove file
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Remove from database
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        return True
                        
                return False
                
            except Exception as e:
                logger.error(f"Disk cache remove error: {e}")
                return False
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT filename FROM cache_entries")
                    for row in cursor:
                        filename = row[0]
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()
                    
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Disk cache clear error: {e}")
    
    def cleanup_expired(self):
        """Clean up expired entries."""
        with self._lock:
            try:
                current_time = time.time()
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT key, filename FROM cache_entries WHERE ttl IS NOT NULL AND ttl < ?",
                        (current_time,)
                    )
                    
                    for key, filename in cursor:
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()
                    
                    conn.execute("DELETE FROM cache_entries WHERE ttl IS NOT NULL AND ttl < ?", (current_time,))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Disk cache cleanup error: {e}")


class CacheLayer:
    """
    Multi-level cache with memory and disk storage.
    
    Provides efficient caching for expensive operations with configurable
    eviction policies, TTL, and persistent storage.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache layer."""
        self.config = config or CacheConfig()
        
        # Initialize caches
        self._memory_cache = LRUCache(max_size=self.config.max_entries)
        self._disk_cache = DiskCache(self.config.disk_cache_dir) if self.config.enable_disk_cache else None
        
        # Statistics
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Function cache decorators
        self._function_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            # Try memory cache first
            value = self._memory_cache.get(key)
            if value is not None:
                self._stats.hits += 1
                return value
            
            # Try disk cache
            if self._disk_cache:
                value = self._disk_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    self._memory_cache.put(key, value)
                    self._stats.hits += 1
                    return value
            
            self._stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self._lock:
            # Store in memory cache
            self._memory_cache.put(key, value)
            
            # Store in disk cache if enabled
            if self._disk_cache:
                cache_ttl = ttl or self.config.default_ttl
                self._disk_cache.put(key, value, cache_ttl)
    
    def remove(self, key: str) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed, False otherwise
        """
        with self._lock:
            memory_removed = self._memory_cache.remove(key)
            disk_removed = self._disk_cache.remove(key) if self._disk_cache else False
            return memory_removed or disk_removed
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            if self._disk_cache:
                self._disk_cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats.hits + self._stats.misses
            hit_rate = (self._stats.hits / total_requests) if total_requests > 0 else 0.0
            
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                memory_usage=0,  # TODO: Calculate actual memory usage
                entry_count=self._memory_cache.size(),
                hit_rate=hit_rate
            )
    
    def cleanup_expired(self):
        """Clean up expired entries."""
        if self._disk_cache:
            self._disk_cache.cleanup_expired()
    
    def cache_function(self, 
                      ttl: Optional[float] = None,
                      key_func: Optional[Callable] = None):
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time to live for cached results
            key_func: Function to generate cache key from arguments
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_function_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def _generate_function_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        try:
            key_data = {
                'function': func_name,
                'args': str(args),  # Convert to string to handle non-serializable objects
                'kwargs': str(sorted(kwargs.items()))
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            # Fallback to simpler key generation
            return hashlib.md5(f"{func_name}_{str(args)}_{str(kwargs)}".encode()).hexdigest()
    
    def memoize(self, func: Callable, ttl: Optional[float] = None) -> Callable:
        """
        Memoize a function with caching.
        
        Args:
            func: Function to memoize
            ttl: Time to live for cached results
            
        Returns:
            Memoized function
        """
        return self.cache_function(ttl=ttl)(func)
    
    def cache_expensive_operation(self, 
                                 operation_name: str,
                                 operation: Callable,
                                 *args,
                                 ttl: Optional[float] = None,
                                 **kwargs) -> Any:
        """
        Cache the result of an expensive operation.
        
        Args:
            operation_name: Name of the operation for cache key
            operation: Function to execute
            *args: Arguments for the operation
            ttl: Time to live for cached result
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result (cached or computed)
        """
        cache_key = self._generate_function_key(operation_name, args, kwargs)
        
        # Try to get from cache
        result = self.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit for operation: {operation_name}")
            return result
        
        # Execute operation and cache result
        logger.debug(f"Cache miss for operation: {operation_name}, executing...")
        result = operation(*args, **kwargs)
        self.put(cache_key, result, ttl)
        
        return result


class CacheManager:
    """
    Global cache manager for coordinating multiple cache instances.
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self._caches: Dict[str, CacheLayer] = {}
        self._lock = threading.Lock()
    
    def get_cache(self, name: str, config: Optional[CacheConfig] = None) -> CacheLayer:
        """
        Get or create a named cache.
        
        Args:
            name: Cache name
            config: Cache configuration
            
        Returns:
            CacheLayer instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = CacheLayer(config)
            return self._caches[name]
    
    def clear_all_caches(self):
        """Clear all managed caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_global_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}
    
    def cleanup_all_expired(self):
        """Clean up expired entries in all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.cleanup_expired()


# Global cache manager instance
cache_manager = CacheManager()


def get_cache(name: str = "default", config: Optional[CacheConfig] = None) -> CacheLayer:
    """
    Get a named cache instance.
    
    Args:
        name: Cache name
        config: Cache configuration
        
    Returns:
        CacheLayer instance
    """
    return cache_manager.get_cache(name, config)


def cached(cache_name: str = "default", 
          ttl: Optional[float] = None,
          key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use
        ttl: Time to live for cached results
        key_func: Function to generate cache key
        
    Returns:
        Decorated function
    """
    cache = get_cache(cache_name)
    return cache.cache_function(ttl=ttl, key_func=key_func)