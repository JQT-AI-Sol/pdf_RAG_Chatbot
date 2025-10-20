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

    def embed_text(self, text: str) -> List[float]:
        """
        テキストをエンベディング

        Args:
            text: エンベディングするテキスト

        Returns:
            list: エンベディングベクトル
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)})")
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
        複数のテキストをバッチでエンベディング（真のバッチ処理）

        Args:
            texts: エンベディングするテキストのリスト
            batch_size: 1回のAPIリクエストで処理するテキスト数（デフォルト: 100）

        Returns:
            list: エンベディングベクトルのリスト
        """
        if not texts:
            return []

        all_embeddings = []

        try:
            # バッチサイズごとに分割して処理
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # OpenAI APIは1回のリクエストで複数テキストを処理可能
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                # レスポンスから埋め込みを抽出
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch ({i+1}-{i+len(batch)}/{len(texts)})")

            logger.info(f"Generated {len(all_embeddings)} embeddings in total using batch processing")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
