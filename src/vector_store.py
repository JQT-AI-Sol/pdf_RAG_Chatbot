"""
Vector store module using ChromaDB
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDBを使用したベクトルストアクラス"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: ChromaDB設定
        """
        self.config = config
        self.chroma_config = config.get('chromadb', {})

        # ChromaDBクライアントの初期化
        self.client = chromadb.PersistentClient(
            path=self.chroma_config.get('persist_directory', './data/chroma_db')
        )

        # コレクション名
        self.text_collection_name = self.chroma_config.get('collection_name_text', 'pdf_text_chunks')
        self.image_collection_name = self.chroma_config.get('collection_name_images', 'pdf_image_contents')

        # コレクション取得または作成
        self.text_collection = self.client.get_or_create_collection(
            name=self.text_collection_name
        )
        self.image_collection = self.client.get_or_create_collection(
            name=self.image_collection_name
        )

        logger.info("Vector store initialized")

    def add_text_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        テキストチャンクをベクトルストアに追加

        Args:
            chunks: テキストチャンクのリスト
            embeddings: 対応するエンベディングのリスト
        """
        try:
            ids = [f"text_{i}" for i in range(len(chunks))]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'source_file': chunk['source_file'],
                    'page_number': chunk['page_number'],
                    'category': chunk['category'],
                    'content_type': 'text'
                }
                for chunk in chunks
            ]

            self.text_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added {len(chunks)} text chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding text chunks: {e}")
            raise

    def add_image_content(self, image_data: Dict[str, Any], embedding: List[float]):
        """
        画像コンテンツ（Vision解析結果）をベクトルストアに追加

        Args:
            image_data: 画像データと解析結果
            embedding: エンベディング
        """
        try:
            image_id = f"image_{image_data['image_path']}"

            metadata = {
                'source_file': image_data.get('source_file', ''),
                'page_number': image_data.get('page_number', 0),
                'category': image_data.get('category', ''),
                'content_type': image_data.get('content_type', 'image'),
                'image_path': image_data['image_path']
            }

            self.image_collection.add(
                ids=[image_id],
                embeddings=[embedding],
                documents=[image_data.get('description', '')],
                metadatas=[metadata]
            )

            logger.info(f"Added image content to vector store: {image_id}")

        except Exception as e:
            logger.error(f"Error adding image content: {e}")
            raise

    def search(self, query_embedding: List[float], category: Optional[str] = None,
               top_k: int = 5, search_type: str = 'both') -> Dict[str, List[Dict[str, Any]]]:
        """
        ベクトル検索を実行

        Args:
            query_embedding: クエリのエンベディング
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            top_k: 取得する結果の数
            search_type: 検索タイプ ('text', 'image', 'both')

        Returns:
            dict: 検索結果（テキストと画像）
        """
        results = {
            'text': [],
            'images': []
        }

        try:
            # カテゴリーフィルター
            where = {'category': category} if category else None

            # テキスト検索
            if search_type in ['text', 'both']:
                text_results = self.text_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where
                )
                results['text'] = self._format_results(text_results)

            # 画像検索
            if search_type in ['image', 'both']:
                image_results = self.image_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where
                )
                results['images'] = self._format_results(image_results)

            logger.info(f"Search completed: {len(results['text'])} text, {len(results['images'])} images")

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

        return results

    def _format_results(self, raw_results: dict) -> List[Dict[str, Any]]:
        """
        検索結果をフォーマット

        Args:
            raw_results: ChromaDBの検索結果

        Returns:
            list: フォーマット済み結果
        """
        formatted = []

        if not raw_results or not raw_results.get('ids'):
            return formatted

        for i in range(len(raw_results['ids'][0])):
            formatted.append({
                'id': raw_results['ids'][0][i],
                'document': raw_results['documents'][0][i],
                'metadata': raw_results['metadatas'][0][i],
                'distance': raw_results['distances'][0][i] if 'distances' in raw_results else None
            })

        return formatted
