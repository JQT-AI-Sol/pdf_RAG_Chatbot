"""
Vector store module using ChromaDB
"""

import logging
import os
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

        # Streamlit Cloud環境を検出
        is_streamlit_cloud = (
            os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
            os.path.exists('/mount/src') or
            'STREAMLIT_SHARING_MODE' in os.environ
        )

        # ChromaDBクライアントの初期化
        if is_streamlit_cloud:
            # Streamlit Cloudではメモリ内のEphemeralClientを使用
            logger.warning("Running on Streamlit Cloud - using EphemeralClient (data will not persist)")
            self.client = chromadb.EphemeralClient()
        else:
            # ローカル環境ではPersistentClientを使用
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
            import uuid
            # ユニークなIDを生成（UUID使用）
            ids = [f"text_{uuid.uuid4().hex[:16]}_{i}" for i in range(len(chunks))]
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
            import hashlib
            # image_pathをハッシュ化してユニークIDを生成
            image_id = f"image_{hashlib.md5(image_data['image_path'].encode()).hexdigest()}"

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

    def add_image_contents_batch(self, image_data_list: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        複数の画像コンテンツをバッチでベクトルストアに追加

        Args:
            image_data_list: 画像データと解析結果のリスト
            embeddings: 対応するエンベディングのリスト
        """
        try:
            if not image_data_list or not embeddings:
                return

            import hashlib
            # image_pathをハッシュ化してユニークIDを生成
            ids = [f"image_{hashlib.md5(img_data['image_path'].encode()).hexdigest()}" for img_data in image_data_list]
            documents = [img_data.get('description', '') for img_data in image_data_list]
            metadatas = [
                {
                    'source_file': img_data.get('source_file', ''),
                    'page_number': img_data.get('page_number', 0),
                    'category': img_data.get('category', ''),
                    'content_type': img_data.get('content_type', 'image'),
                    'image_path': img_data['image_path']
                }
                for img_data in image_data_list
            ]

            self.image_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added {len(image_data_list)} image contents to vector store in batch")

        except Exception as e:
            logger.error(f"Error adding image contents in batch: {e}")
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

    def get_registered_pdfs(self) -> List[Dict[str, Any]]:
        """
        登録済みPDFのリストを取得

        Returns:
            list: PDFファイルごとの情報
                [{
                    'source_file': str,
                    'category': str,
                    'text_count': int,
                    'image_count': int,
                    'total_count': int
                }, ...]
        """
        try:
            pdf_info = {}

            # テキストコレクションから取得
            text_data = self.text_collection.get()
            if text_data and text_data.get('metadatas'):
                for metadata in text_data['metadatas']:
                    source_file = metadata.get('source_file', '')
                    if source_file:
                        if source_file not in pdf_info:
                            pdf_info[source_file] = {
                                'source_file': source_file,
                                'category': metadata.get('category', ''),
                                'text_count': 0,
                                'image_count': 0
                            }
                        pdf_info[source_file]['text_count'] += 1

            # イメージコレクションから取得
            image_data = self.image_collection.get()
            if image_data and image_data.get('metadatas'):
                for metadata in image_data['metadatas']:
                    source_file = metadata.get('source_file', '')
                    if source_file:
                        if source_file not in pdf_info:
                            pdf_info[source_file] = {
                                'source_file': source_file,
                                'category': metadata.get('category', ''),
                                'text_count': 0,
                                'image_count': 0
                            }
                        pdf_info[source_file]['image_count'] += 1

            # 合計カウントを追加
            result = []
            for pdf_data in pdf_info.values():
                pdf_data['total_count'] = pdf_data['text_count'] + pdf_data['image_count']
                result.append(pdf_data)

            # ファイル名でソート
            result.sort(key=lambda x: x['source_file'])

            logger.info(f"Found {len(result)} registered PDFs")
            return result

        except Exception as e:
            logger.error(f"Error getting registered PDFs: {e}")
            return []

    def delete_by_source_file(self, source_file: str) -> Dict[str, int]:
        """
        特定のPDFファイルに関連する全てのベクトルデータを削除

        Args:
            source_file: 削除対象のPDFファイル名

        Returns:
            dict: 削除件数 {'text_deleted': int, 'image_deleted': int}
        """
        deleted_counts = {
            'text_deleted': 0,
            'image_deleted': 0
        }

        try:
            # テキストコレクションから削除
            text_data = self.text_collection.get(
                where={'source_file': source_file}
            )
            if text_data and text_data.get('ids'):
                text_ids = text_data['ids']
                if text_ids:
                    self.text_collection.delete(ids=text_ids)
                    deleted_counts['text_deleted'] = len(text_ids)
                    logger.info(f"Deleted {len(text_ids)} text chunks for {source_file}")

            # イメージコレクションから削除
            image_data = self.image_collection.get(
                where={'source_file': source_file}
            )
            if image_data and image_data.get('ids'):
                image_ids = image_data['ids']
                if image_ids:
                    self.image_collection.delete(ids=image_ids)
                    deleted_counts['image_deleted'] = len(image_ids)
                    logger.info(f"Deleted {len(image_ids)} image contents for {source_file}")

            logger.info(f"Successfully deleted all data for {source_file}")

        except Exception as e:
            logger.error(f"Error deleting data for {source_file}: {e}")
            raise

        return deleted_counts
