"""
Text embedding module using OpenAI API
"""

import logging
import os
from typing import List
from openai import OpenAI


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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストをバッチでエンベディング

        Args:
            texts: エンベディングするテキストのリスト

        Returns:
            list: エンベディングベクトルのリスト
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
