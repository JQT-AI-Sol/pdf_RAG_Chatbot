"""
Vector store module supporting both ChromaDB and Supabase
"""

import logging
import os
from typing import List, Dict, Any, Optional
import uuid
import hashlib

logger = logging.getLogger(__name__)


class VectorStore:
    """ベクトルストアクラス（ChromaDB / Supabase対応）"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        self.vs_config = config.get('vector_store', {})
        self.provider = self.vs_config.get('provider', 'chromadb')

        if self.provider == 'supabase':
            self._init_supabase()
        else:
            self._init_chromadb()

        logger.info(f"Vector store initialized with provider: {self.provider}")

    def _init_supabase(self):
        """Supabaseクライアントの初期化"""
        try:
            from supabase import create_client, Client

            # 環境変数から接続情報を取得
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_KEY')

            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

            self.client: Client = create_client(supabase_url, supabase_key)

            # テーブル名
            supabase_config = self.vs_config.get('supabase', {})
            self.text_table = supabase_config.get('table_name_text', 'pdf_text_chunks')
            self.image_table = supabase_config.get('table_name_images', 'pdf_image_contents')
            self.pdf_table = supabase_config.get('table_name_pdfs', 'registered_pdfs')
            self.match_threshold = supabase_config.get('match_threshold', 0.7)
            self.storage_bucket = supabase_config.get('storage_bucket', 'pdf-images')

            logger.info(f"Supabase client initialized (URL: {supabase_url})")

            # Storageバケットの確認・作成
            try:
                # バケットが存在するか確認
                buckets = self.client.storage.list_buckets()
                bucket_names = [b.name for b in buckets]

                if self.storage_bucket not in bucket_names:
                    # バケットが存在しない場合は作成
                    self.client.storage.create_bucket(
                        self.storage_bucket,
                        options={"public": False}  # プライベートバケット
                    )
                    logger.info(f"Created Supabase Storage bucket: {self.storage_bucket}")
                else:
                    logger.info(f"Using existing Supabase Storage bucket: {self.storage_bucket}")
            except Exception as e:
                logger.warning(f"Could not verify/create storage bucket: {e}. Continuing anyway...")

        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
            raise

    def _init_chromadb(self):
        """ChromaDBクライアントの初期化"""
        import chromadb

        chroma_config = self.vs_config.get('chromadb', {})

        # Streamlit Cloud環境を検出
        is_streamlit_cloud = (
            os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
            os.path.exists('/mount/src') or
            'STREAMLIT_SHARING_MODE' in os.environ
        )

        if is_streamlit_cloud:
            persist_dir = '/tmp/chroma_db'
            logger.warning(f"Running on Streamlit Cloud - using PersistentClient with temporary directory: {persist_dir}")
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.PersistentClient(
                path=chroma_config.get('persist_directory', './data/chroma_db')
            )

        # コレクション名
        self.text_collection_name = chroma_config.get('collection_name_text', 'pdf_text_chunks')
        self.image_collection_name = chroma_config.get('collection_name_images', 'pdf_image_contents')

        # コレクション取得または作成
        self.text_collection = self.client.get_or_create_collection(
            name=self.text_collection_name
        )
        self.image_collection = self.client.get_or_create_collection(
            name=self.image_collection_name
        )

    def add_text_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        テキストチャンクをベクトルストアに追加

        Args:
            chunks: テキストチャンクのリスト
            embeddings: 対応するエンベディングのリスト
        """
        if self.provider == 'supabase':
            self._add_text_chunks_supabase(chunks, embeddings)
        else:
            self._add_text_chunks_chromadb(chunks, embeddings)

    def _add_text_chunks_supabase(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Supabaseにテキストチャンクを追加"""
        try:
            records = []
            for chunk, embedding in zip(chunks, embeddings):
                records.append({
                    'id': f"text_{uuid.uuid4().hex[:16]}",
                    'content': chunk['text'],
                    'embedding': embedding,
                    'source_file': chunk['source_file'],
                    'page_number': chunk['page_number'],
                    'category': chunk['category'],
                    'content_type': 'text'
                })

            self.client.table(self.text_table).insert(records).execute()
            logger.info(f"Added {len(chunks)} text chunks to Supabase")

        except Exception as e:
            logger.error(f"Error adding text chunks to Supabase: {e}")
            raise

    def _add_text_chunks_chromadb(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """ChromaDBにテキストチャンクを追加"""
        try:
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

            logger.info(f"Added {len(chunks)} text chunks to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding text chunks to ChromaDB: {e}")
            raise

    def add_image_contents_batch(self, image_data_list: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        複数の画像コンテンツをバッチでベクトルストアに追加

        Args:
            image_data_list: 画像データと解析結果のリスト
            embeddings: 対応するエンベディングのリスト
        """
        if not image_data_list or not embeddings:
            return

        if self.provider == 'supabase':
            self._add_image_contents_supabase(image_data_list, embeddings)
        else:
            self._add_image_contents_chromadb(image_data_list, embeddings)

    def _add_image_contents_supabase(self, image_data_list: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Supabaseに画像コンテンツを追加（画像はStorageにアップロード）"""
        try:
            from pathlib import Path

            records = []
            for img_data, embedding in zip(image_data_list, embeddings):
                image_id = f"image_{hashlib.md5(img_data['image_path'].encode()).hexdigest()}"

                # 画像をSupabase Storageにアップロード
                local_image_path = img_data['image_path']
                storage_path = None

                if Path(local_image_path).exists():
                    try:
                        # Storageパスを生成（category/filename形式）
                        category = img_data.get('category', 'uncategorized')
                        filename = Path(local_image_path).name
                        storage_path = f"{category}/{filename}"

                        # 画像ファイルをアップロード
                        with open(local_image_path, 'rb') as f:
                            image_bytes = f.read()

                        self.client.storage.from_(self.storage_bucket).upload(
                            storage_path,
                            image_bytes,
                            file_options={"content-type": "image/png", "upsert": "true"}
                        )
                        logger.debug(f"Uploaded image to Storage: {storage_path}")

                    except Exception as upload_error:
                        logger.warning(f"Failed to upload image {local_image_path} to Storage: {upload_error}")
                        # アップロード失敗時はローカルパスをそのまま使用
                        storage_path = local_image_path
                else:
                    # ファイルが存在しない場合はローカルパスをそのまま使用
                    logger.warning(f"Image file not found: {local_image_path}")
                    storage_path = local_image_path

                records.append({
                    'id': image_id,
                    'content': img_data.get('description', ''),
                    'embedding': embedding,
                    'source_file': img_data.get('source_file', ''),
                    'page_number': img_data.get('page_number', 0),
                    'category': img_data.get('category', ''),
                    'content_type': img_data.get('content_type', 'image'),
                    'image_path': storage_path  # Storage pathを保存
                })

            # upsertでon_conflictを明示的に指定（主キーidで競合解決）
            self.client.table(self.image_table).upsert(
                records,
                on_conflict='id'
            ).execute()
            logger.info(f"Upserted {len(image_data_list)} image contents to Supabase")

        except Exception as e:
            logger.error(f"Error adding image contents to Supabase: {e}")
            raise

    def _add_image_contents_chromadb(self, image_data_list: List[Dict[str, Any]], embeddings: List[List[float]]):
        """ChromaDBに画像コンテンツを追加"""
        try:
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

            logger.info(f"Added {len(image_data_list)} image contents to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding image contents to ChromaDB: {e}")
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
        if self.provider == 'supabase':
            return self._search_supabase(query_embedding, category, top_k, search_type)
        else:
            return self._search_chromadb(query_embedding, category, top_k, search_type)

    def _search_supabase(self, query_embedding: List[float], category: Optional[str],
                        top_k: int, search_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Supabaseでベクトル検索"""
        results = {'text': [], 'images': []}

        try:
            # テキスト検索
            if search_type in ['text', 'both']:
                response = self.client.rpc(
                    'match_text_chunks',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': self.match_threshold,
                        'match_count': top_k,
                        'filter_category': category
                    }
                ).execute()

                # デバッグ: 実際のレスポンスを確認
                if response.data and len(response.data) > 0:
                    logger.info(f"Supabase text result - Keys: {list(response.data[0].keys())}")
                    logger.info(f"Supabase text result - Sample data: source_file={response.data[0].get('source_file')}, page={response.data[0].get('page_number')}, category={response.data[0].get('category')}")

                if response.data:
                    results['text'] = [
                        {
                            'id': row.get('id', ''),
                            'content': row.get('content', ''),
                            'source_file': row.get('source_file', ''),
                            'page_number': row.get('page_number', 0),
                            'category': row.get('category', ''),
                            'content_type': 'text',  # テキストは常に'text'
                            'distance': row.get('distance', 1 - row.get('similarity', 0)),
                            'metadata': {
                                'source_file': row.get('source_file', ''),
                                'page_number': row.get('page_number', 0),
                                'category': row.get('category', ''),
                                'content_type': 'text'
                            }
                        }
                        for row in response.data
                    ]

            # 画像検索
            if search_type in ['image', 'both']:
                response = self.client.rpc(
                    'match_image_contents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': self.match_threshold,
                        'match_count': top_k,
                        'filter_category': category
                    }
                ).execute()

                # デバッグ: 実際のレスポンスを確認
                if response.data and len(response.data) > 0:
                    logger.info(f"Supabase image result - Keys: {list(response.data[0].keys())}")
                    logger.info(f"Supabase image result - Sample data: source_file={response.data[0].get('source_file')}, page={response.data[0].get('page_number')}, category={response.data[0].get('category')}")

                if response.data:
                    results['images'] = [
                        {
                            'id': row.get('id', ''),
                            'description': row.get('content', ''),
                            'source_file': row.get('source_file', ''),
                            'page_number': row.get('page_number', 0),
                            'category': row.get('category', ''),
                            'content_type': row.get('content_type', 'image'),  # DBから取得、デフォルトは'image'
                            'path': row.get('image_path', ''),
                            'distance': row.get('distance', 1 - row.get('similarity', 0)),
                            'metadata': {
                                'source_file': row.get('source_file', ''),
                                'page_number': row.get('page_number', 0),
                                'category': row.get('category', ''),
                                'content_type': row.get('content_type', 'image')
                            }
                        }
                        for row in response.data
                    ]

            logger.info(f"Search completed: {len(results['text'])} text, {len(results['images'])} images")

        except Exception as e:
            logger.error(f"Error during Supabase search: {e}")
            raise

        return results

    def _search_chromadb(self, query_embedding: List[float], category: Optional[str],
                        top_k: int, search_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """ChromaDBでベクトル検索"""
        results = {'text': [], 'images': []}

        try:
            where = {'category': category} if category else None

            if search_type in ['text', 'both']:
                text_results = self.text_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where
                )
                results['text'] = self._format_chromadb_results(text_results)

            if search_type in ['image', 'both']:
                image_results = self.image_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where
                )
                results['images'] = self._format_chromadb_results(image_results)

            logger.info(f"Search completed: {len(results['text'])} text, {len(results['images'])} images")

        except Exception as e:
            logger.error(f"Error during ChromaDB search: {e}")
            raise

        return results

    def _format_chromadb_results(self, raw_results: dict) -> List[Dict[str, Any]]:
        """ChromaDB検索結果をフォーマット"""
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

    def get_all_categories(self) -> List[str]:
        """
        登録済みPDFから一意のカテゴリーリストを取得

        Returns:
            list: カテゴリー名のリスト（重複なし、ソート済み）
        """
        try:
            pdfs = self.get_registered_pdfs()
            categories = list(set(pdf['category'] for pdf in pdfs if pdf.get('category')))
            return sorted(categories)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_registered_pdfs(self) -> List[Dict[str, Any]]:
        """
        登録済みPDFのリストを取得

        Returns:
            list: PDFファイルごとの情報
        """
        if self.provider == 'supabase':
            return self._get_registered_pdfs_supabase()
        else:
            return self._get_registered_pdfs_chromadb()

    def _get_registered_pdfs_supabase(self) -> List[Dict[str, Any]]:
        """Supabaseから登録済みPDF一覧を取得"""
        try:
            # registered_pdfsテーブルから取得
            response = self.client.table(self.pdf_table).select('*').execute()

            if not response.data:
                return []

            result = []
            for row in response.data:
                # テキストと画像の件数を集計
                text_count = self.client.table(self.text_table)\
                    .select('id', count='exact')\
                    .eq('source_file', row['filename'])\
                    .execute()

                image_count = self.client.table(self.image_table)\
                    .select('id', count='exact')\
                    .eq('source_file', row['filename'])\
                    .execute()

                result.append({
                    'source_file': row['filename'],
                    'category': row['category'],
                    'text_count': text_count.count if text_count else 0,
                    'image_count': image_count.count if image_count else 0,
                    'total_count': (text_count.count if text_count else 0) + (image_count.count if image_count else 0)
                })

            logger.info(f"Found {len(result)} registered PDFs")
            return result

        except Exception as e:
            logger.error(f"Error getting registered PDFs from Supabase: {e}")
            return []

    def _get_registered_pdfs_chromadb(self) -> List[Dict[str, Any]]:
        """ChromaDBから登録済みPDF一覧を取得"""
        try:
            pdf_info = {}

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

            result = []
            for pdf_data in pdf_info.values():
                pdf_data['total_count'] = pdf_data['text_count'] + pdf_data['image_count']
                result.append(pdf_data)

            result.sort(key=lambda x: x['source_file'])

            logger.info(f"Found {len(result)} registered PDFs")
            return result

        except Exception as e:
            logger.error(f"Error getting registered PDFs from ChromaDB: {e}")
            return []

    def delete_by_source_file(self, source_file: str) -> Dict[str, int]:
        """
        特定のPDFファイルに関連する全てのベクトルデータを削除

        Args:
            source_file: 削除対象のPDFファイル名

        Returns:
            dict: 削除件数 {'text_deleted': int, 'image_deleted': int}
        """
        if self.provider == 'supabase':
            return self._delete_by_source_file_supabase(source_file)
        else:
            return self._delete_by_source_file_chromadb(source_file)

    def _delete_by_source_file_supabase(self, source_file: str) -> Dict[str, int]:
        """Supabaseから特定PDFのデータを削除"""
        deleted_counts = {'text_deleted': 0, 'image_deleted': 0}

        try:
            # テキスト削除
            text_response = self.client.table(self.text_table)\
                .delete()\
                .eq('source_file', source_file)\
                .execute()
            deleted_counts['text_deleted'] = len(text_response.data) if text_response.data else 0

            # 画像削除
            image_response = self.client.table(self.image_table)\
                .delete()\
                .eq('source_file', source_file)\
                .execute()
            deleted_counts['image_deleted'] = len(image_response.data) if image_response.data else 0

            # PDF登録情報削除
            self.client.table(self.pdf_table)\
                .delete()\
                .eq('filename', source_file)\
                .execute()

            logger.info(f"Successfully deleted all data for {source_file} from Supabase")

        except Exception as e:
            logger.error(f"Error deleting data from Supabase for {source_file}: {e}")
            raise

        return deleted_counts

    def _delete_by_source_file_chromadb(self, source_file: str) -> Dict[str, int]:
        """ChromaDBから特定PDFのデータを削除"""
        deleted_counts = {'text_deleted': 0, 'image_deleted': 0}

        try:
            text_data = self.text_collection.get(where={'source_file': source_file})
            if text_data and text_data.get('ids'):
                text_ids = text_data['ids']
                if text_ids:
                    self.text_collection.delete(ids=text_ids)
                    deleted_counts['text_deleted'] = len(text_ids)

            image_data = self.image_collection.get(where={'source_file': source_file})
            if image_data and image_data.get('ids'):
                image_ids = image_data['ids']
                if image_ids:
                    self.image_collection.delete(ids=image_ids)
                    deleted_counts['image_deleted'] = len(image_ids)

            logger.info(f"Successfully deleted all data for {source_file} from ChromaDB")

        except Exception as e:
            logger.error(f"Error deleting data from ChromaDB for {source_file}: {e}")
            raise

        return deleted_counts

    def register_pdf(self, filename: str, category: str):
        """
        PDFをregistered_pdfsテーブルに登録

        Args:
            filename: PDFファイル名
            category: カテゴリー
        """
        if self.provider == 'supabase':
            try:
                self.client.table(self.pdf_table).upsert({
                    'filename': filename,
                    'category': category
                }).execute()
                logger.info(f"Registered PDF in Supabase: {filename}")
            except Exception as e:
                logger.error(f"Error registering PDF in Supabase: {e}")
                raise
