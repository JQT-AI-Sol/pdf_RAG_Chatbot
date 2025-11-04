"""
Embedding cache module for caching text embeddings to reduce latency and API costs.

This module provides LRU memory cache and persistent disk cache for embeddings.
"""

import hashlib
import pickle
from typing import List, Optional
from pathlib import Path
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Embedding cache with LRU memory cache and persistent disk cache.

    This class caches text embeddings to avoid redundant API calls, reducing both
    latency and costs. It uses a two-tier caching strategy:
    1. Memory cache (LRU): Fast access for frequently used embeddings
    2. Disk cache: Persistent storage for long-term caching
    """

    def __init__(self, cache_dir: str = "./cache/embeddings", max_memory_items: int = 1000):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory path for disk cache storage
            max_memory_items: Maximum number of items to keep in memory cache (LRU)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use OrderedDict for LRU cache implementation
        self._memory_cache = OrderedDict()
        self._max_memory_items = max_memory_items

        # Statistics for monitoring
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "saves": 0
        }

        logger.info(f"EmbeddingCache initialized: {cache_dir} (max_memory_items={max_memory_items})")

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key (SHA256 hash) from text.

        Args:
            text: Input text to hash

        Returns:
            str: SHA256 hash of the text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from cache.

        Checks memory cache first, then disk cache. If found in disk cache,
        it will be promoted to memory cache.

        Args:
            text: Input text to retrieve embedding for

        Returns:
            List[float] if cached, None if not found
        """
        key = self._get_cache_key(text)

        # Check memory cache (OrderedDict maintains insertion order)
        if key in self._memory_cache:
            # Move to end (most recently used)
            self._memory_cache.move_to_end(key)
            self.stats["memory_hits"] += 1
            logger.debug(f"Memory cache hit: {key[:8]}...")
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)

                self.stats["disk_hits"] += 1
                logger.debug(f"Disk cache hit: {key[:8]}...")

                # Promote to memory cache
                self._add_to_memory_cache(key, embedding)
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache from disk: {e}")

        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key[:8]}...")
        return None

    def set(self, text: str, embedding: List[float]):
        """
        Save an embedding to cache (both memory and disk).

        Args:
            text: Input text
            embedding: Embedding vector to cache
        """
        key = self._get_cache_key(text)

        # Add to memory cache
        self._add_to_memory_cache(key, embedding)

        # Persist to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            self.stats["saves"] += 1
            logger.debug(f"Cached embedding: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _add_to_memory_cache(self, key: str, value: List[float]):
        """
        Add an item to memory cache using LRU eviction policy.

        Args:
            key: Cache key
            value: Embedding vector
        """
        # If key already exists, move it to end (most recently used)
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self._memory_cache[key] = value
        else:
            # Add new item
            self._memory_cache[key] = value

            # Evict oldest if cache is full
            if len(self._memory_cache) > self._max_memory_items:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
                logger.debug(f"Evicted from memory cache: {oldest_key[:8]}...")

    def clear_memory_cache(self):
        """Clear the memory cache (disk cache remains intact)."""
        self._memory_cache.clear()
        logger.info("Memory cache cleared")

    def clear_disk_cache(self):
        """Clear the disk cache (memory cache remains intact)."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Disk cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    def clear_all(self):
        """Clear both memory and disk caches."""
        self.clear_memory_cache()
        self.clear_disk_cache()
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "saves": 0
        }
        logger.info("All caches cleared")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including hit rates and cache sizes
        """
        total_requests = self.stats["memory_hits"] + self.stats["disk_hits"] + self.stats["misses"]
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.stats["memory_hits"] + self.stats["disk_hits"]) / total_requests

        return {
            **self.stats,
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.pkl"))),
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }
