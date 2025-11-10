"""
Reranker module for improving search relevance using Cross-Encoder models.

This module provides reranking functionality to improve the quality of search results
by reordering them based on semantic relevance to the query.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks search results using a Cross-Encoder model.

    Cross-encoders provide more accurate relevance scoring than bi-encoders
    by jointly encoding the query and document, at the cost of higher latency.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the Reranker with a Cross-Encoder model.

        Args:
            model_name: Name of the Cross-Encoder model from Hugging Face.
                       Default: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (lightweight, fast)
                       Alternative: 'cross-encoder/ms-marco-electra-base' (more accurate)

        Note:
            The model will be downloaded on first use (~80MB for default model).
            GPU will be used automatically if available (CUDA).
        """
        logger.info(f"Initializing Reranker with model: {model_name}")
        try:
            import torch
            import os

            # Disable meta device initialization to avoid "copy out of meta tensor" error
            os.environ['TRANSFORMERS_OFFLINE'] = '0'

            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")

            # Initialize CrossEncoder without device parameter (auto-detection)
            # Then move to target device if needed
            self.model = CrossEncoder(model_name, max_length=512)

            # Move model to target device if not already there
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(device)

            logger.info("Reranker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of document strings to rerank
            top_k: Number of top results to return (default: 5)

        Returns:
            List of tuples (document_index, relevance_score) sorted by relevance.
            Scores are typically in range [-10, 10], higher is more relevant.

        Example:
            >>> reranker = Reranker()
            >>> query = "How to create a table?"
            >>> docs = [
            ...     "To create a graph, select the data...",
            ...     "To create a table, go to Insert menu...",
            ...     "To insert an image, click..."
            ... ]
            >>> results = reranker.rerank(query, docs, top_k=1)
            >>> print(results)  # [(1, 8.5)]  # Document at index 1 is most relevant
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        if len(documents) <= top_k:
            # If we have fewer documents than top_k, no need to rerank
            logger.debug(f"Document count ({len(documents)}) <= top_k ({top_k}), returning all")
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)
            ranked_indices = np.argsort(scores)[::-1]
            return list(zip(ranked_indices.tolist(), scores[ranked_indices].tolist()))

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Compute relevance scores
            logger.debug(f"Computing reranking scores for {len(documents)} documents")
            scores = self.model.predict(pairs)

            # Sort by score (descending) and take top_k
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            ranked_scores = scores[ranked_indices]

            logger.debug(
                f"Reranking completed: {len(documents)} docs -> top {top_k}",
                extra={
                    "top_scores": ranked_scores.tolist()[:3],  # Log top 3 scores
                    "score_range": f"[{scores.min():.2f}, {scores.max():.2f}]"
                }
            )

            return list(zip(ranked_indices.tolist(), ranked_scores.tolist()))

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return original order
            logger.warning("Falling back to original document order")
            return [(i, 0.0) for i in range(min(len(documents), top_k))]
