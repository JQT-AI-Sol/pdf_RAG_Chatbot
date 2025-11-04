"""
Text embedding module using OpenAI API
"""

import logging
import os
from typing import List
from openai import OpenAI as OpenAIClient

# Langfuse統合
try:
    from langfuse.openai import OpenAI
    LANGFUSE_AVAILABLE = True
except ImportError:
    OpenAI = OpenAIClient
    LANGFUSE_AVAILABLE = False

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class TextEmbedder:
    """OpenAI APIを使用してテキストをエンベディングするクラス"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: OpenAI設定
        """
        self.config = config
        self.openai_config = config.get('openai', {})
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = self.openai_config.get('model_embedding', 'text-embedding-3-small')

        # キャッシュ初期化
        self.cache = None
        rag_config = config.get('rag', {})
        cache_config = config.get('cache', {}).get('embedding', {})

        if rag_config.get('enable_embedding_cache', False):
            try:
                cache_dir = cache_config.get('directory', './cache/embeddings')
                max_items = cache_config.get('max_memory_items', 1000)
                self.cache = EmbeddingCache(cache_dir=cache_dir, max_memory_items=max_items)
                logger.info(f"Embedding cache enabled (max_memory_items={max_items})")
            except Exception as e:
                logger.error(f"Failed to initialize embedding cache: {e}")
                logger.warning("Continuing without caching")

    def embed_text(self, text: str) -> List[float]:
        """
        テキストをエンベディング（キャッシュ対応）

        Args:
            text: エンベディングするテキスト

        Returns:
            list: エンベディングベクトル
        """
        # キャッシュチェック
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        try:
            # キャッシュミス: API呼び出し
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)})")

            # キャッシュに保存
            if self.cache:
                self.cache.set(text, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        クエリをエンベディング（embed_textのエイリアス）

        Args:
            text: エンベディングするクエリテキスト

        Returns:
            list: エンベディングベクトル
        """
        return self.embed_text(text)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        複数のテキストをバッチでエンベディング（キャッシュ対応）

        Args:
            texts: エンベディングするテキストのリスト
            batch_size: 1回のAPIリクエストで処理するテキスト数（デフォルト: 100）

        Returns:
            list: エンベディングベクトルのリスト
        """
        if not texts:
            return []

        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        # キャッシュヒット/ミスを判定
        if self.cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)

            logger.debug(f"Cache hit: {len(texts) - len(uncached_texts)}/{len(texts)} embeddings")
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # キャッシュミスのテキストをバッチ処理
        if uncached_texts:
            try:
                all_embeddings = []

                # バッチサイズごとに分割して処理
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]

                    # OpenAI APIは1回のリクエストで複数テキストを処理可能
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )

                    # レスポンスから埋め込みを抽出
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                    logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch ({i+1}-{i+len(batch)}/{len(uncached_texts)})")

                # 結果をキャッシュに保存 & results配列に格納
                for idx, text, embedding in zip(uncached_indices, uncached_texts, all_embeddings):
                    if self.cache:
                        self.cache.set(text, embedding)
                    results[idx] = embedding

                logger.info(f"Generated {len(all_embeddings)} new embeddings (total: {len(texts)})")

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return results
