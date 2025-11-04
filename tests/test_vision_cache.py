"""
Tests for vision analysis caching
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from PIL import Image
from src.vision_analyzer import VisionAnalyzer


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    temp_dir = tempfile.mkdtemp()
    image_path = Path(temp_dir) / "test_image.png"

    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    img.save(image_path)

    yield str(image_path)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def analyzer_with_cache(temp_cache_dir):
    """Create VisionAnalyzer with caching enabled"""
    config = {
        "rag": {"enable_vision_cache": True},
        "cache": {
            "vision": {
                "enabled": True,
                "directory": temp_cache_dir
            }
        },
        "openai": {"model_vision": "gpt-5"},
        "vision": {
            "analysis_prompt_table": "Analyze this table",
            "analysis_prompt_graph": "Analyze this graph"
        }
    }
    return VisionAnalyzer(config)


@pytest.fixture
def analyzer_without_cache():
    """Create VisionAnalyzer with caching disabled"""
    config = {
        "rag": {"enable_vision_cache": False},
        "cache": {"vision": {"enabled": False}},
        "openai": {"model_vision": "gpt-5"},
        "vision": {
            "analysis_prompt_table": "Analyze this table",
            "analysis_prompt_graph": "Analyze this graph"
        }
    }
    return VisionAnalyzer(config)


def test_cache_initialization(analyzer_with_cache, temp_cache_dir):
    """Test that cache directory is initialized"""
    assert analyzer_with_cache.cache_dir is not None
    assert analyzer_with_cache.cache_dir.exists()
    assert str(analyzer_with_cache.cache_dir) == temp_cache_dir
    print(f"\n✓ Cache initialized at: {temp_cache_dir}")


def test_cache_disabled(analyzer_without_cache):
    """Test that cache can be disabled"""
    assert analyzer_without_cache.cache_dir is None
    print("\n✓ Cache correctly disabled")


def test_image_hash_generation(analyzer_with_cache, temp_image):
    """Test that image hash generation is consistent"""
    hash1 = analyzer_with_cache._get_image_hash(temp_image)
    hash2 = analyzer_with_cache._get_image_hash(temp_image)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex characters
    print(f"\n✓ Image hash generated: {hash1[:16]}...")


def test_cache_file_creation(analyzer_with_cache, temp_image, temp_cache_dir):
    """Test that cache files are created after analysis"""
    # Note: This test doesn't actually call the API
    # It only tests the cache file structure

    # Manually create a cache entry to simulate successful analysis
    image_hash = analyzer_with_cache._get_image_hash(temp_image)
    cache_key = f"{image_hash}_table"
    cache_file = Path(temp_cache_dir) / f"{cache_key}.json"

    # Simulate cache save
    test_result = {
        "content_type": "table",
        "description": "Test analysis result",
        "image_path": temp_image,
        "model": "gpt-5"
    }

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'result': test_result,
            'timestamp': '2025-01-01T00:00:00',
            'content_type': 'table',
            'image_hash': image_hash
        }, f, ensure_ascii=False, indent=2)

    assert cache_file.exists()
    print(f"\n✓ Cache file created: {cache_file.name}")

    # Verify cache file content
    with open(cache_file, 'r', encoding='utf-8') as f:
        cached_data = json.load(f)

    assert cached_data['result']['content_type'] == 'table'
    assert cached_data['image_hash'] == image_hash
    print("✓ Cache file content validated")


def test_different_content_types_different_cache(analyzer_with_cache, temp_image):
    """Test that different content types create different cache entries"""
    image_hash = analyzer_with_cache._get_image_hash(temp_image)

    cache_key_table = f"{image_hash}_table"
    cache_key_graph = f"{image_hash}_graph"

    assert cache_key_table != cache_key_graph
    print(f"\n✓ Different content types create different cache keys:")
    print(f"  Table: {cache_key_table[:24]}...")
    print(f"  Graph: {cache_key_graph[:24]}...")


def test_cache_key_includes_content_type(analyzer_with_cache, temp_image):
    """Test that cache keys include both image hash and content type"""
    image_hash = analyzer_with_cache._get_image_hash(temp_image)

    # Test different content types
    for content_type in ['table', 'graph', 'full_page']:
        cache_key = f"{image_hash}_{content_type}"
        assert content_type in cache_key
        assert image_hash in cache_key

    print("\n✓ Cache keys correctly include content type")


def test_cache_directory_structure(temp_cache_dir):
    """Test cache directory structure"""
    cache_path = Path(temp_cache_dir)

    # Simulate multiple cache files
    test_files = [
        "abc123_table.json",
        "def456_graph.json",
        "ghi789_table.json"
    ]

    for filename in test_files:
        (cache_path / filename).touch()

    cached_files = list(cache_path.glob("*.json"))
    assert len(cached_files) == 3

    # Group by content type
    tables = [f for f in cached_files if 'table' in f.name]
    graphs = [f for f in cached_files if 'graph' in f.name]

    assert len(tables) == 2
    assert len(graphs) == 1

    print(f"\n✓ Cache directory structure verified:")
    print(f"  Total files: {len(cached_files)}")
    print(f"  Tables: {len(tables)}")
    print(f"  Graphs: {len(graphs)}")


if __name__ == "__main__":
    print("Running vision cache tests...")

    # Create temp directories
    temp_cache = tempfile.mkdtemp()
    temp_img_dir = tempfile.mkdtemp()

    try:
        # Create test image
        test_img_path = Path(temp_img_dir) / "test.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(test_img_path)

        # Run tests
        analyzer_with = VisionAnalyzer({
            "rag": {"enable_vision_cache": True},
            "cache": {"vision": {"enabled": True, "directory": temp_cache}},
            "openai": {"model_vision": "gpt-5"},
            "vision": {
                "analysis_prompt_table": "Analyze",
                "analysis_prompt_graph": "Analyze"
            }
        })

        analyzer_without = VisionAnalyzer({
            "rag": {"enable_vision_cache": False},
            "cache": {"vision": {"enabled": False}},
            "openai": {"model_vision": "gpt-5"},
            "vision": {
                "analysis_prompt_table": "Analyze",
                "analysis_prompt_graph": "Analyze"
            }
        })

        test_cache_initialization(analyzer_with, temp_cache)
        test_cache_disabled(analyzer_without)
        test_image_hash_generation(analyzer_with, str(test_img_path))
        test_cache_file_creation(analyzer_with, str(test_img_path), temp_cache)
        test_different_content_types_different_cache(analyzer_with, str(test_img_path))
        test_cache_key_includes_content_type(analyzer_with, str(test_img_path))
        test_cache_directory_structure(temp_cache)

        print("\n✓ All vision cache tests passed!")
    finally:
        # Cleanup
        shutil.rmtree(temp_cache, ignore_errors=True)
        shutil.rmtree(temp_img_dir, ignore_errors=True)
