"""
Cohere Reranker module for improved search relevance using Cohere's Rerank API.

This module provides reranking functionality using Cohere's multilingual reranking models,
which offer superior performance for Japanese text compared to local cross-encoder models.
"""

import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Reranks search results using Cohere's Rerank API.

    Cohere's rerank models provide state-of-the-art relevance scoring with excellent
    multilingual support, particularly for Japanese text.

    Supported models:
    - rerank-multilingual-v3.0: Best for Japanese and multilingual content
    - rerank-english-v3.0: Optimized for English-only content
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'rerank-multilingual-v3.0'
    ):
        """
        Initialize the Cohere Reranker.

        Args:
            api_key: Cohere API key. If None, will read from COHERE_API_KEY env variable.
            model: Model name to use for reranking.
                   Options: 'rerank-multilingual-v3.0' (default), 'rerank-english-v3.0'

        Raises:
            ValueError: If API key is not provided and not found in environment
            ImportError: If cohere package is not installed
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install with: pip install cohere"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('COHERE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Cohere API key not provided. "
                "Set COHERE_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.client = cohere.Client(self.api_key)

        logger.info(f"Initialized CohereReranker with model: {model}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        max_chunks_per_doc: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to the query using Cohere API.

        Args:
            query: The search query
            documents: List of document strings to rerank
            top_k: Number of top results to return (default: 5)
            max_chunks_per_doc: Optional parameter for long document handling
                                (not used in current implementation)

        Returns:
            List of tuples (document_index, relevance_score) sorted by relevance.
            Scores are normalized to [0, 1], higher is more relevant.

        Example:
            >>> reranker = CohereReranker(api_key="your-key")
            >>> query = "日雇派遣労働者数は何人ですか？"
            >>> docs = [
            ...     "表２－３ 日雇派遣の状況\\n日雇派遣労働者数 | 39,053人",
            ...     "製造業の労働者数について...",
            ...     "医療ベッド数の統計..."
            ... ]
            >>> results = reranker.rerank(query, docs, top_k=1)
            >>> print(results)  # [(0, 0.95)]  # Document 0 is most relevant
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        if len(documents) <= top_k:
            logger.debug(
                f"Document count ({len(documents)}) <= top_k ({top_k}), "
                "returning all documents"
            )
            # Still call API to get scores
            top_k = len(documents)

        try:
            logger.debug(
                f"Calling Cohere Rerank API: {len(documents)} documents, "
                f"model={self.model}, top_k={top_k}"
            )

            # Call Cohere Rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=False  # We already have the documents
            )

            # Extract results
            # response.results is a list of RerankResult objects with:
            # - index: original document index
            # - relevance_score: score (typically 0-1 range)
            results = [
                (result.index, result.relevance_score)
                for result in response.results
            ]

            logger.info(
                f"Cohere reranking completed: {len(documents)} docs -> {len(results)} results",
                extra={
                    "top_scores": [score for _, score in results[:3]],
                    "score_range": f"[{min(s for _, s in results):.3f}, {max(s for _, s in results):.3f}]"
                }
            )

            return results

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}", exc_info=True)

            # Fallback: return original order with zero scores
            logger.warning("Falling back to original document order")
            return [(i, 0.0) for i in range(min(len(documents), top_k))]

    def health_check(self) -> bool:
        """
        Check if Cohere API is accessible and API key is valid.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test with minimal API usage
            test_query = "test"
            test_docs = ["test document"]

            response = self.client.rerank(
                model=self.model,
                query=test_query,
                documents=test_docs,
                top_n=1
            )

            logger.info("Cohere API health check: OK")
            return True

        except Exception as e:
            logger.error(f"Cohere API health check failed: {e}")
            return False
