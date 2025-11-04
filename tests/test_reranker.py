"""
Tests for the Reranker module
"""

import pytest
from src.reranker import Reranker


def test_reranker_initialization():
    """Test that Reranker can be initialized"""
    reranker = Reranker()
    assert reranker is not None
    assert reranker.model is not None


def test_reranker_basic_functionality():
    """Test basic reranking functionality"""
    reranker = Reranker()

    query = "How to create a database table?"
    documents = [
        "To create a graph, select the data and choose Insert > Chart.",
        "To create a table in SQL, use the CREATE TABLE statement.",
        "To insert an image, click on Insert > Picture.",
        "Database tables store data in rows and columns.",
        "Graphics and charts help visualize data.",
    ]

    # Rerank documents
    results = reranker.rerank(query, documents, top_k=3)

    # Assertions
    assert len(results) == 3
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)

    # The most relevant documents should be about databases/tables (indices 1 and 3)
    top_indices = [idx for idx, _ in results[:2]]
    assert 1 in top_indices or 3 in top_indices, "Expected database-related documents to be ranked highly"

    print(f"\nQuery: {query}")
    print(f"Top 3 results:")
    for idx, score in results:
        print(f"  [{idx}] Score: {score:.4f} - {documents[idx][:50]}...")


def test_reranker_empty_documents():
    """Test reranker with empty document list"""
    reranker = Reranker()

    query = "test query"
    documents = []

    results = reranker.rerank(query, documents, top_k=5)
    assert results == []


def test_reranker_fewer_docs_than_topk():
    """Test reranker when documents are fewer than top_k"""
    reranker = Reranker()

    query = "test"
    documents = ["doc1", "doc2"]

    results = reranker.rerank(query, documents, top_k=5)
    assert len(results) == 2  # Should return all available documents


def test_reranker_relevance_scoring():
    """Test that relevance scores are reasonable"""
    reranker = Reranker()

    query = "Python programming language"
    documents = [
        "Python is a high-level programming language.",
        "The python snake is found in tropical regions.",
        "Java is another programming language.",
    ]

    results = reranker.rerank(query, documents, top_k=3)

    # First result should be most relevant (Python programming)
    assert results[0][0] == 0, "Most relevant document should be ranked first"

    # Scores should be in descending order
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    print(f"\nQuery: {query}")
    print(f"Results:")
    for idx, score in results:
        print(f"  [{idx}] Score: {score:.4f} - {documents[idx]}")


if __name__ == "__main__":
    print("Running Reranker tests...")

    # Run tests manually
    test_reranker_initialization()
    print("✓ Initialization test passed")

    test_reranker_basic_functionality()
    print("✓ Basic functionality test passed")

    test_reranker_empty_documents()
    print("✓ Empty documents test passed")

    test_reranker_fewer_docs_than_topk()
    print("✓ Fewer docs than top_k test passed")

    test_reranker_relevance_scoring()
    print("✓ Relevance scoring test passed")

    print("\nAll tests passed!")
