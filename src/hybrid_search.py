"""
Hybrid search module combining BM25 and vector search for improved recall.

This module provides hybrid search functionality that combines:
- BM25 (keyword-based) search for exact term matching
- Vector search for semantic similarity
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

# Try to import MeCab for Japanese tokenization
try:
    import MeCab
    MECAB_AVAILABLE = True
except (ImportError, RuntimeError):
    MECAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid searcher combining BM25 (keyword) and vector (semantic) search.

    This class implements hybrid retrieval by:
    1. Performing both BM25 and vector searches
    2. Normalizing scores from both methods
    3. Combining scores with configurable weights
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize the hybrid searcher.

        Args:
            alpha: Weight for vector search (0-1). BM25 weight = 1-alpha.
                   Higher alpha means more weight on semantic search.
                   Default: 0.7 (70% vector, 30% BM25)
        """
        self.alpha = alpha
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_doc_metadata = []

        # Initialize tokenizer (MeCab if available, else simple tokenizer)
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger("-Owakati")
                self.tokenizer_type = "mecab"
                logger.info("Using MeCab tokenizer for Japanese support")
            except RuntimeError:
                self.mecab = None
                self.tokenizer_type = "simple"
                logger.warning("MeCab initialization failed, falling back to simple tokenizer")
        else:
            self.mecab = None
            self.tokenizer_type = "simple"
            logger.info("MeCab not available, using simple tokenizer")

        logger.info(f"HybridSearcher initialized (alpha={alpha}, tokenizer={self.tokenizer_type})")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using available tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if self.mecab:
            try:
                return self.mecab.parse(text).strip().split()
            except Exception as e:
                logger.warning(f"MeCab tokenization failed: {e}, falling back to simple tokenizer")

        # Simple tokenizer (works for English and basic Japanese)
        # Lowercase, split by whitespace and common punctuation
        text = text.lower()
        # Split by whitespace, punctuation, and special characters
        tokens = re.findall(r'\w+', text)
        return tokens

    def build_bm25_index(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts (same length as documents)
        """
        if not documents:
            logger.warning("No documents provided for BM25 index")
            return

        logger.info(f"Building BM25 index for {len(documents)} documents...")

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        self.bm25_docs = documents
        self.bm25_doc_metadata = metadata or [{}] * len(documents)
        self.bm25_index = BM25Okapi(tokenized_docs)

        logger.info(f"BM25 index built successfully ({self.tokenizer_type} tokenizer)")

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.bm25_index:
            logger.warning("BM25 index not built")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include results with non-zero scores
                results.append((
                    self.bm25_docs[idx],
                    float(bm25_scores[idx]),
                    self.bm25_doc_metadata[idx]
                ))

        logger.debug(f"BM25 search: {len(results)} results (query: '{query[:50]}...')")
        return results

    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        Normalize scores using min-max normalization.

        Args:
            scores: List of scores to normalize

        Returns:
            Normalized scores (0-1 range)
        """
        if not scores:
            return np.array([])

        scores_array = np.array(scores)

        # Handle edge cases
        if len(scores_array) == 1:
            return np.array([1.0])

        min_score = scores_array.min()
        max_score = scores_array.max()

        if max_score - min_score == 0:
            return np.ones(len(scores_array))

        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized

    def hybrid_search(
        self,
        query: str,
        vector_results: List[Tuple[str, float, Dict]],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid search combining BM25 and vector search results.

        Args:
            query: Search query
            vector_results: Results from vector search as (doc, score, metadata) tuples
            top_k: Number of final results to return

        Returns:
            List of (document, hybrid_score, metadata) tuples sorted by score
        """
        # If BM25 index not built, return vector results only
        if not self.bm25_index:
            logger.debug("BM25 index not available, returning vector results only")
            return vector_results[:top_k]

        # Perform BM25 search
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        if not bm25_results:
            logger.debug("No BM25 results, returning vector results only")
            return vector_results[:top_k]

        # Normalize scores
        vector_scores = [score for _, score, _ in vector_results]
        bm25_scores = [score for _, score, _ in bm25_results]

        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Combine scores
        combined_scores = {}

        # Add vector search results
        for idx, (doc, _, metadata) in enumerate(vector_results):
            combined_scores[doc] = {
                "score": self.alpha * vector_scores_norm[idx],
                "metadata": metadata
            }

        # Add BM25 results
        for idx, (doc, _, metadata) in enumerate(bm25_results):
            if doc in combined_scores:
                # Document found in both searches - combine scores
                combined_scores[doc]["score"] += (1 - self.alpha) * bm25_scores_norm[idx]
            else:
                # Document only found in BM25 search
                combined_scores[doc] = {
                    "score": (1 - self.alpha) * bm25_scores_norm[idx],
                    "metadata": metadata
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:top_k]

        # Convert back to list of tuples
        final_results = [
            (doc, data["score"], data["metadata"])
            for doc, data in sorted_results
        ]

        logger.info(
            f"Hybrid search completed: {len(final_results)} results "
            f"(vector: {len(vector_results)}, bm25: {len(bm25_results)})"
        )

        return final_results

    def get_stats(self) -> Dict:
        """
        Get statistics about the BM25 index.

        Returns:
            Dict with index statistics
        """
        return {
            "index_built": self.bm25_index is not None,
            "num_documents": len(self.bm25_docs),
            "tokenizer": self.tokenizer_type,
            "alpha": self.alpha,
            "mecab_available": MECAB_AVAILABLE
        }
