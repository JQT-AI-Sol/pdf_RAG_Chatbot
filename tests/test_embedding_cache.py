"""
Tests for the embedding cache module
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.embedding_cache import EmbeddingCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache(temp_cache_dir):
    """Create an EmbeddingCache instance with temporary directory"""
    return EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=3)


def test_cache_initialization(temp_cache_dir):
    """Test that cache initializes correctly"""
    cache = EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=10)
    assert cache is not None
    assert cache.cache_dir.exists()
    assert cache._max_memory_items == 10
    print("\n✓ Cache initialized successfully")


def test_cache_miss(cache):
    """Test cache miss behavior"""
    text = "This is a test query"
    result = cache.get(text)

    assert result is None
    assert cache.stats["misses"] == 1
    assert cache.stats["memory_hits"] == 0
    assert cache.stats["disk_hits"] == 0
    print("\n✓ Cache miss handled correctly")


def test_cache_hit_memory(cache):
    """Test memory cache hit"""
    text = "This is a test query"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Set cache
    cache.set(text, embedding)

    # Get from cache (should hit memory)
    result = cache.get(text)

    assert result == embedding
    assert cache.stats["memory_hits"] == 1
    assert cache.stats["disk_hits"] == 0
    assert cache.stats["misses"] == 0
    print("\n✓ Memory cache hit works correctly")


def test_cache_persistence(cache, temp_cache_dir):
    """Test that cache persists to disk"""
    text = "This is a test query"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Set cache
    cache.set(text, embedding)

    # Verify disk file exists
    cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
    assert len(cache_files) == 1
    print(f"\n✓ Cache persisted to disk: {cache_files[0].name}")

    # Create new cache instance (clear memory)
    new_cache = EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=3)

    # Get from cache (should hit disk and promote to memory)
    result = new_cache.get(text)

    assert result == embedding
    assert new_cache.stats["disk_hits"] == 1
    assert new_cache.stats["memory_hits"] == 0
    print("✓ Disk cache hit and promotion to memory works correctly")


def test_lru_eviction(cache):
    """Test LRU eviction when memory cache is full"""
    embeddings = {
        "text1": [0.1, 0.2, 0.3],
        "text2": [0.4, 0.5, 0.6],
        "text3": [0.7, 0.8, 0.9],
        "text4": [1.0, 1.1, 1.2],  # This should evict text1
    }

    # Fill cache (max_memory_items=3)
    for text, embedding in embeddings.items():
        cache.set(text, embedding)

    # Memory cache should have only 3 items
    assert len(cache._memory_cache) == 3

    # text1 should be evicted from memory (but still on disk)
    result = cache.get("text1")
    assert result == [0.1, 0.2, 0.3]
    assert cache.stats["disk_hits"] == 1  # Retrieved from disk
    print("\n✓ LRU eviction works correctly")


def test_cache_key_generation(cache):
    """Test that cache key generation is consistent"""
    text = "Test query"
    key1 = cache._get_cache_key(text)
    key2 = cache._get_cache_key(text)

    assert key1 == key2
    assert len(key1) == 64  # SHA256 produces 64 hex characters
    print(f"\n✓ Cache key generation is consistent: {key1[:16]}...")


def test_cache_stats(cache):
    """Test cache statistics tracking"""
    # Initial stats
    stats = cache.get_stats()
    assert stats["memory_hits"] == 0
    assert stats["disk_hits"] == 0
    assert stats["misses"] == 0
    assert stats["saves"] == 0
    assert stats["hit_rate"] == 0.0

    # Cache miss
    cache.get("text1")

    # Cache set
    cache.set("text1", [0.1, 0.2])

    # Cache hit
    cache.get("text1")

    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["saves"] == 1
    assert stats["memory_hits"] == 1
    assert stats["total_requests"] == 2
    assert stats["hit_rate"] == 0.5  # 1 hit out of 2 requests

    print(f"\n✓ Cache statistics tracked correctly:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Memory hits: {stats['memory_hits']}")
    print(f"  Disk hits: {stats['disk_hits']}")
    print(f"  Misses: {stats['misses']}")


def test_cache_clear(cache):
    """Test cache clearing functionality"""
    # Add some items
    cache.set("text1", [0.1, 0.2])
    cache.set("text2", [0.3, 0.4])

    # Clear memory cache
    cache.clear_memory_cache()
    assert len(cache._memory_cache) == 0

    # Disk cache should still have files
    assert len(list(cache.cache_dir.glob("*.pkl"))) > 0

    # Clear all
    cache.clear_all()
    assert len(cache._memory_cache) == 0
    assert len(list(cache.cache_dir.glob("*.pkl"))) == 0

    print("\n✓ Cache clearing works correctly")


if __name__ == "__main__":
    print("Running embedding cache tests...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Run tests manually
        test_cache_initialization(temp_dir)

        cache_instance = EmbeddingCache(cache_dir=temp_dir, max_memory_items=3)
        test_cache_miss(cache_instance)
        test_cache_hit_memory(cache_instance)
        test_cache_persistence(cache_instance, temp_dir)

        # Create fresh cache for LRU test
        cache_instance2 = EmbeddingCache(cache_dir=temp_dir + "_lru", max_memory_items=3)
        test_lru_eviction(cache_instance2)

        cache_instance3 = EmbeddingCache(cache_dir=temp_dir + "_key", max_memory_items=3)
        test_cache_key_generation(cache_instance3)

        cache_instance4 = EmbeddingCache(cache_dir=temp_dir + "_stats", max_memory_items=3)
        test_cache_stats(cache_instance4)

        cache_instance5 = EmbeddingCache(cache_dir=temp_dir + "_clear", max_memory_items=3)
        test_cache_clear(cache_instance5)

        print("\n✓ All embedding cache tests passed!")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_dir + "_lru", ignore_errors=True)
        shutil.rmtree(temp_dir + "_key", ignore_errors=True)
        shutil.rmtree(temp_dir + "_stats", ignore_errors=True)
        shutil.rmtree(temp_dir + "_clear", ignore_errors=True)
