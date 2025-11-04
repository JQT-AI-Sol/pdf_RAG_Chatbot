"""
Tests for BM25 hybrid search module
"""

import pytest
from src.hybrid_search import HybridSearcher


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties",
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for documents"""
    return [
        {"id": 1, "category": "ML"},
        {"id": 2, "category": "DL"},
        {"id": 3, "category": "NLP"},
        {"id": 4, "category": "CV"},
        {"id": 5, "category": "RL"},
    ]


@pytest.fixture
def searcher():
    """Create a HybridSearcher instance"""
    return HybridSearcher(alpha=0.7)


def test_initialization(searcher):
    """Test that HybridSearcher initializes correctly"""
    assert searcher is not None
    assert searcher.alpha == 0.7
    assert searcher.bm25_index is None
    assert searcher.tokenizer_type in ["mecab", "simple"]
    print(f"\n✓ HybridSearcher initialized (tokenizer: {searcher.tokenizer_type})")


def test_simple_tokenizer(searcher):
    """Test simple tokenizer"""
    text = "This is a test document"
    tokens = searcher._tokenize(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
    print(f"\n✓ Tokenized: '{text}' -> {tokens}")


def test_bm25_index_building(searcher, sample_documents, sample_metadata):
    """Test BM25 index construction"""
    searcher.build_bm25_index(sample_documents, sample_metadata)

    assert searcher.bm25_index is not None
    assert len(searcher.bm25_docs) == len(sample_documents)
    assert len(searcher.bm25_doc_metadata) == len(sample_metadata)
    print(f"\n✓ BM25 index built for {len(sample_documents)} documents")


def test_bm25_search_keyword_query(searcher, sample_documents, sample_metadata):
    """Test BM25 search with keyword query"""
    searcher.build_bm25_index(sample_documents, sample_metadata)

    # Search for specific keyword
    results = searcher.search_bm25("neural networks", top_k=3)

    assert len(results) > 0
    # Check that results contain tuples of (doc, score, metadata)
    for doc, score, metadata in results:
        assert isinstance(doc, str)
        assert isinstance(score, float)
        assert isinstance(metadata, dict)
        assert score > 0

    # The document about deep learning should rank high
    top_doc = results[0][0]
    assert "neural" in top_doc.lower() or "deep learning" in top_doc.lower()

    print(f"\n✓ BM25 search found {len(results)} results for 'neural networks'")
    print(f"  Top result: {top_doc[:60]}...")


def test_score_normalization(searcher):
    """Test score normalization"""
    scores = [1.0, 2.5, 5.0, 0.5, 3.0]
    normalized = searcher._normalize_scores(scores)

    assert len(normalized) == len(scores)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    print(f"\n✓ Scores normalized: {scores} -> {normalized.tolist()}")


def test_hybrid_search(searcher, sample_documents, sample_metadata):
    """Test hybrid search combining BM25 and vector results"""
    searcher.build_bm25_index(sample_documents, sample_metadata)

    # Simulate vector search results (doc, score, metadata)
    vector_results = [
        (sample_documents[1], 0.9, sample_metadata[1]),  # Deep learning
        (sample_documents[0], 0.7, sample_metadata[0]),  # Machine learning
        (sample_documents[2], 0.6, sample_metadata[2]),  # NLP
    ]

    # Perform hybrid search
    query = "neural networks deep learning"
    hybrid_results = searcher.hybrid_search(query, vector_results, top_k=3)

    assert len(hybrid_results) > 0
    assert len(hybrid_results) <= 3

    # Check result format
    for doc, score, metadata in hybrid_results:
        assert isinstance(doc, str)
        assert isinstance(score, float)
        assert isinstance(metadata, dict)
        assert 0 <= score <= 1  # Normalized score

    print(f"\n✓ Hybrid search completed: {len(hybrid_results)} results")
    print(f"  Query: '{query}'")
    print(f"  Top result: {hybrid_results[0][0][:50]}...")


def test_hybrid_search_without_bm25_index(searcher):
    """Test hybrid search fallback when BM25 index not built"""
    # Vector results only
    vector_results = [
        ("Document 1", 0.9, {"id": 1}),
        ("Document 2", 0.7, {"id": 2}),
    ]

    query = "test query"
    hybrid_results = searcher.hybrid_search(query, vector_results, top_k=2)

    # Should return vector results as-is
    assert len(hybrid_results) == len(vector_results)
    assert hybrid_results == vector_results

    print("\n✓ Hybrid search correctly falls back to vector-only when BM25 not built")


def test_different_alpha_values():
    """Test hybrid search with different alpha values"""
    documents = ["machine learning", "deep learning", "neural networks"]
    metadata = [{"id": i} for i in range(3)]

    # Test with different alpha values
    alphas = [0.3, 0.5, 0.7, 0.9]

    print("\n✓ Testing different alpha values:")
    for alpha in alphas:
        searcher = HybridSearcher(alpha=alpha)
        searcher.build_bm25_index(documents, metadata)

        vector_results = [(documents[0], 0.8, metadata[0])]
        query = "machine learning"

        results = searcher.hybrid_search(query, vector_results, top_k=2)

        print(f"  alpha={alpha}: {len(results)} results")
        assert len(results) > 0


def test_get_stats(searcher, sample_documents, sample_metadata):
    """Test statistics retrieval"""
    # Before building index
    stats_before = searcher.get_stats()
    assert stats_before["index_built"] is False
    assert stats_before["num_documents"] == 0

    # After building index
    searcher.build_bm25_index(sample_documents, sample_metadata)
    stats_after = searcher.get_stats()

    assert stats_after["index_built"] is True
    assert stats_after["num_documents"] == len(sample_documents)
    assert stats_after["alpha"] == 0.7
    assert stats_after["tokenizer"] in ["mecab", "simple"]

    print(f"\n✓ Statistics retrieved successfully:")
    print(f"  Index built: {stats_after['index_built']}")
    print(f"  Documents: {stats_after['num_documents']}")
    print(f"  Tokenizer: {stats_after['tokenizer']}")
    print(f"  Alpha: {stats_after['alpha']}")


if __name__ == "__main__":
    print("Running BM25 hybrid search tests...")

    # Create fixtures
    docs = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties",
    ]
    meta = [{"id": i, "category": f"cat{i}"} for i in range(len(docs))]
    searcher_inst = HybridSearcher(alpha=0.7)

    # Run tests
    test_initialization(searcher_inst)
    test_simple_tokenizer(searcher_inst)
    test_bm25_index_building(searcher_inst, docs, meta)

    searcher_inst2 = HybridSearcher(alpha=0.7)
    test_bm25_search_keyword_query(searcher_inst2, docs, meta)

    searcher_inst3 = HybridSearcher(alpha=0.7)
    test_score_normalization(searcher_inst3)

    searcher_inst4 = HybridSearcher(alpha=0.7)
    test_hybrid_search(searcher_inst4, docs, meta)

    searcher_inst5 = HybridSearcher(alpha=0.7)
    test_hybrid_search_without_bm25_index(searcher_inst5)

    test_different_alpha_values()

    searcher_inst6 = HybridSearcher(alpha=0.7)
    test_get_stats(searcher_inst6, docs, meta)

    print("\n✓ All BM25 hybrid search tests passed!")
