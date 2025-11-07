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

        # BM25ハイブリッド検索用のトークナイザー初期化
        self._init_tokenizer()

        logger.info(f"Vector store initialized with provider: {self.provider} (v1.1)")

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
            self.pdf_storage_bucket = supabase_config.get('pdf_storage_bucket', 'pdf-files')

            logger.info(f"Supabase client initialized (URL: {supabase_url})")

            # Storageバケットの確認・作成
            try:
                # バケットが存在するか確認
                buckets = self.client.storage.list_buckets()
                bucket_names = [b.name for b in buckets]

                # 画像用バケット
                if self.storage_bucket not in bucket_names:
                    # バケットが存在しない場合は作成
                    self.client.storage.create_bucket(
                        self.storage_bucket,
                        options={"public": False}  # プライベートバケット
                    )
                    logger.info(f"Created Supabase Storage bucket: {self.storage_bucket}")
                else:
                    logger.info(f"Using existing Supabase Storage bucket: {self.storage_bucket}")

                # PDF用バケット
                if self.pdf_storage_bucket not in bucket_names:
                    self.client.storage.create_bucket(
                        self.pdf_storage_bucket,
                        options={"public": False}  # プライベートバケット
                    )
                    logger.info(f"Created Supabase Storage bucket for PDFs: {self.pdf_storage_bucket}")
                else:
                    logger.info(f"Using existing Supabase Storage bucket for PDFs: {self.pdf_storage_bucket}")
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

    def _init_tokenizer(self):
        """
        BM25ハイブリッド検索用の日本語トークナイザー初期化
        """
        try:
            import MeCab
            self.tokenizer = MeCab.Tagger("-Owakati")
            logger.info("MeCab tokenizer initialized for BM25 hybrid search")
        except ImportError:
            logger.warning("MeCab not available, using simple space-based tokenization as fallback")
            self.tokenizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize MeCab: {e}. Using simple tokenization as fallback")
            self.tokenizer = None

    def _tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化（日本語対応）

        Args:
            text: トークン化するテキスト

        Returns:
            list: トークンのリスト
        """
        if not text:
            return []

        if self.tokenizer:
            # MeCabで分かち書き
            try:
                tokens = self.tokenizer.parse(text).strip().split()
                logger.debug(f"MeCab tokenization: '{text}' -> {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                return [token for token in tokens if len(token) > 1]  # 1文字のトークンは除外
            except Exception as e:
                logger.warning(f"MeCab tokenization failed: {e}, falling back to regex-based split")

        # フォールバック: Regex-based smart tokenization
        # 英数字、ひらがな、カタカナ、漢字をそれぞれまとまりとして抽出
        import re

        # 重要な略語（2-4文字の英語）を先に保護
        important_keywords = ['SNS', 'AI', 'IT', 'PC', 'OS', 'API', 'URL', 'VPN', 'DNS', 'HTTP', 'HTTPS']

        # パターンマッチングで単語を抽出
        # [a-zA-Z0-9]+: 連続する英数字（SNS、API、123など）
        # [ぁ-ん]+: 連続するひらがな（利用、注意など）
        # [ァ-ヴー]+: 連続するカタカナ（セキュリティなど）
        # [一-龥]+: 連続する漢字（注意点、利用など）
        words = re.findall(r'[a-zA-Z0-9]+|[ぁ-ん]+|[ァ-ヴー]+|[一-龥]+', text)

        tokens = []
        for word in words:
            # 英数字の場合
            if word.isascii():
                # 重要な略語は大文字で保持
                upper_word = word.upper()
                if upper_word in important_keywords:
                    tokens.append(upper_word)
                # 2文字以上の英数字は小文字化して追加
                elif len(word) >= 2:
                    tokens.append(word.lower())
            # 日本語の場合は2文字以上のトークンのみ
            elif len(word) >= 2:
                tokens.append(word)

        logger.debug(f"Regex tokenization: '{text}' -> {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        return tokens

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
            # デバッグ: Embeddingサイズを確認
            if embeddings and len(embeddings) > 0:
                first_emb = embeddings[0]
                first_emb_dim = len(first_emb)
                logger.info(f"🔍 DEBUG: Saving {len(embeddings)} embeddings, first dimension: {first_emb_dim}")
                logger.info(f"🔍 DEBUG: Embedding type before save: {type(first_emb)}")
                logger.info(f"🔍 DEBUG: First 3 values: {first_emb[:3]}")
                if first_emb_dim != 3072:
                    logger.error(f"❌ DEBUG: ABNORMAL embedding dimension before save! Expected 3072, got {first_emb_dim}")

            records = []
            for chunk, embedding in zip(chunks, embeddings):
                records.append({
                    'id': f"text_{uuid.uuid4().hex[:16]}",
                    'content': chunk['text'],
                    'embedding': embedding,  # List[float]のまま渡す（Supabaseが自動変換）
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

            # デバッグ: Embeddingサイズを確認
            if embeddings and len(embeddings) > 0:
                first_emb_dim = len(embeddings[0])
                logger.info(f"🔍 DEBUG: Saving {len(embeddings)} image embeddings, first dimension: {first_emb_dim}")
                if first_emb_dim != 3072:
                    logger.error(f"❌ DEBUG: ABNORMAL image embedding dimension before save! Expected 3072, got {first_emb_dim}")

            records = []
            for img_data, embedding in zip(image_data_list, embeddings):
                image_id = f"image_{hashlib.md5(img_data['image_path'].encode()).hexdigest()}"

                # 画像をSupabase Storageにアップロード
                local_image_path = img_data['image_path']
                storage_path = None

                if Path(local_image_path).exists():
                    try:
                        # Storageパスを生成（URL-safe形式）
                        # 日本語を含むカテゴリー名・ファイル名はSupabase Storageで使えないため、
                        # ハッシュベースのパスを使用
                        category = img_data.get('category', 'uncategorized')
                        filename = Path(local_image_path).name

                        # カテゴリーとファイル名をURL-safeにエンコード
                        # さらにハッシュを付けてユニーク性を保証
                        category_hash = hashlib.md5(category.encode('utf-8')).hexdigest()[:8]
                        file_ext = Path(filename).suffix
                        file_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:16]
                        storage_path = f"cat_{category_hash}/img_{file_hash}{file_ext}"

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
                    'embedding': embedding,  # List[float]のまま渡す（Supabaseが自動変換）
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
               top_k: int = 5, search_type: str = 'both', query_text: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        ベクトル検索を実行（BM25ハイブリッド検索対応）

        Args:
            query_embedding: クエリのエンベディング
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            top_k: 取得する結果の数
            search_type: 検索タイプ ('text', 'image', 'both')
            query_text: クエリテキスト（BM25ハイブリッド検索用、オプション）

        Returns:
            dict: 検索結果（テキストと画像）
        """
        # BM25ハイブリッド検索の有効化を確認
        hybrid_config = self.config.get('hybrid_search', {})
        use_hybrid = (
            hybrid_config.get('enabled', False) and
            query_text and
            self.provider == 'supabase' and
            search_type in ['text', 'both']
        )

        if use_hybrid:
            logger.info("Using BM25 hybrid search for text results")
            return self._hybrid_search_supabase(query_text, query_embedding, category, top_k, search_type)
        else:
            # 従来のベクトル検索のみ
            if self.provider == 'supabase':
                return self._search_supabase(query_embedding, category, top_k, search_type)
            else:
                return self._search_chromadb(query_embedding, category, top_k, search_type)

    def _reciprocal_rank_fusion(self, vector_results: List[Dict], bm25_results: List[Dict],
                               alpha: float = 0.7, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) でベクトル検索とBM25の結果を統合

        Args:
            vector_results: ベクトル検索結果のリスト
            bm25_results: BM25検索結果のリスト（スコア付き）
            alpha: ベクトル検索の重み（0-1）、BM25の重みは1-alpha
            k: RRFパラメータ（デフォルト60）

        Returns:
            list: 統合された検索結果（スコア順）
        """
        # IDでアクセスできるように辞書化
        vector_dict = {r['id']: (i, r) for i, r in enumerate(vector_results)}
        bm25_dict = {r['id']: (i, r) for i, r in enumerate(bm25_results)}

        # 全てのユニークなIDを取得
        all_ids = set(vector_dict.keys()) | set(bm25_dict.keys())

        # RRFスコアを計算
        fused_results = []
        for doc_id in all_ids:
            score = 0.0

            # ベクトル検索のランクからスコア計算
            if doc_id in vector_dict:
                vector_rank, vector_result = vector_dict[doc_id]
                score += alpha * (1.0 / (k + vector_rank + 1))
                result_data = vector_result
            else:
                result_data = None

            # BM25のランクからスコア計算
            if doc_id in bm25_dict:
                bm25_rank, bm25_result = bm25_dict[doc_id]
                score += (1 - alpha) * (1.0 / (k + bm25_rank + 1))
                if result_data is None:
                    result_data = bm25_result

            # 結果を追加
            if result_data:
                result_with_score = result_data.copy()
                result_with_score['hybrid_score'] = score
                fused_results.append(result_with_score)

        # スコア順にソート
        fused_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        logger.debug(f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(fused_results)} merged results")
        if fused_results:
            top_scores = [r['hybrid_score'] for r in fused_results[:3]]
            logger.info(f"Top 3 hybrid scores: {top_scores}")

        return fused_results

    def _hybrid_search_supabase(self, query_text: str, query_embedding: List[float],
                               category: Optional[str], top_k: int, search_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        BM25 + ベクトル検索のハイブリッド検索（Supabase）

        Args:
            query_text: クエリテキスト
            query_embedding: クエリのエンベディング
            category: 検索対象カテゴリー
            top_k: 最終的に返す結果の数
            search_type: 検索タイプ ('text', 'image', 'both')

        Returns:
            dict: 検索結果（テキストと画像）
        """
        from rank_bm25 import BM25Okapi

        results = {'text': [], 'images': []}

        try:
            # === 1. テキスト検索（ハイブリッド） ===
            if search_type in ['text', 'both']:
                # 1.1 全候補を取得（カテゴリーフィルタあり、thresholdなし）
                query_builder = self.client.table(self.text_table).select('id, content, source_file, page_number, category')
                if category:
                    query_builder = query_builder.eq('category', category)

                all_candidates_response = query_builder.execute()

                if not all_candidates_response.data:
                    logger.warning(f"No text candidates found for category: {category}")
                else:
                    all_candidates = all_candidates_response.data
                    logger.info(f"Retrieved {len(all_candidates)} text candidates for hybrid search")

                    # 1.2 ベクトル検索（上位候補を多めに取得）
                    top_k_vector = min(top_k * 3, len(all_candidates))  # top_kの3倍（最大で全候補）
                    vector_results_dict = self._search_supabase(
                        query_embedding, category, top_k_vector, 'text'
                    )
                    vector_results = vector_results_dict.get('text', [])
                    logger.info(f"Vector search returned {len(vector_results)} results")

                    # 1.3 BM25検索
                    # 候補のテキストをトークン化
                    corpus = [candidate['content'] for candidate in all_candidates]
                    tokenized_corpus = [self._tokenize(doc) for doc in corpus]

                    # BM25インデックス構築
                    bm25 = BM25Okapi(tokenized_corpus)

                    # クエリをトークン化してBM25スコア計算
                    tokenized_query = self._tokenize(query_text)
                    bm25_scores = bm25.get_scores(tokenized_query)

                    # スコア順にソート
                    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

                    # 上位候補を取得
                    top_k_bm25 = min(top_k * 3, len(all_candidates))
                    bm25_results = []
                    for idx in bm25_ranked_indices[:top_k_bm25]:
                        candidate = all_candidates[idx]
                        bm25_results.append({
                            'id': candidate['id'],
                            'content': candidate['content'],
                            'source_file': candidate['source_file'],
                            'page_number': candidate['page_number'],
                            'category': candidate['category'],
                            'content_type': 'text',
                            'bm25_score': float(bm25_scores[idx]),
                            'metadata': {
                                'source_file': candidate['source_file'],
                                'page_number': candidate['page_number'],
                                'category': candidate['category'],
                                'content_type': 'text'
                            }
                        })

                    logger.info(f"BM25 search returned {len(bm25_results)} results (top score: {bm25_results[0]['bm25_score'] if bm25_results else 0:.2f})")

                    # 1.4 Reciprocal Rank Fusion (RRF)
                    hybrid_config = self.config.get('hybrid_search', {})
                    alpha = hybrid_config.get('alpha', 0.7)

                    fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results, alpha=alpha)

                    # 上位top_k件を返す
                    results['text'] = fused_results[:top_k]

                    logger.info(f"Hybrid search completed: {len(results['text'])} final text results")

            # === 2. 画像検索（従来のベクトル検索のみ） ===
            if search_type in ['image', 'both']:
                image_results_dict = self._search_supabase(query_embedding, category, top_k, 'image')
                results['images'] = image_results_dict.get('images', [])

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}", exc_info=True)
            # エラー時は従来のベクトル検索にフォールバック
            logger.warning("Falling back to vector-only search")
            return self._search_supabase(query_embedding, category, top_k, search_type)

        return results

    def _search_supabase(self, query_embedding: List[float], category: Optional[str],
                        top_k: int, search_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Supabaseでベクトル検索"""
        results = {'text': [], 'images': []}

        try:
            # デバッグ: データが存在するか確認
            if category:
                count_response = self.client.table(self.text_table)\
                    .select('id', count='exact')\
                    .eq('category', category)\
                    .execute()
                logger.info(f"🔍 DEBUG: Found {count_response.count} text chunks with category='{category}' in database")

            # テキスト検索
            if search_type in ['text', 'both']:
                logger.info(f"Calling match_text_chunks with category={category}, top_k={top_k}, threshold={self.match_threshold}")

                # 🔍 デバッグ: query_embeddingを確認
                logger.info(f"🔍 DEBUG: query_embedding type={type(query_embedding)}, len={len(query_embedding) if query_embedding else 0}")
                if query_embedding and len(query_embedding) > 0:
                    logger.info(f"🔍 DEBUG: First 3 values: {query_embedding[:3]}")
                else:
                    logger.error(f"❌ DEBUG: query_embedding is empty or None!")

                # RPCパラメータを準備
                rpc_params = {
                    'query_embedding': query_embedding,  # List[float]のまま渡す（Supabaseが自動変換）
                    'match_threshold': self.match_threshold,
                    'match_count': top_k,
                    'filter_category': category
                }

                logger.info(f"🔍 DEBUG: RPC params prepared - threshold={self.match_threshold}, count={top_k}, category={category}")

                response = self.client.rpc('match_text_chunks', rpc_params).execute()

                logger.info(f"Text search response received: {len(response.data) if response.data else 0} results")

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
                # デバッグ: データが存在するか確認
                if category:
                    count_response = self.client.table(self.image_table)\
                        .select('id', count='exact')\
                        .eq('category', category)\
                        .execute()
                    logger.info(f"🔍 DEBUG: Found {count_response.count} image contents with category='{category}' in database")

                logger.info(f"Calling match_image_contents with category={category}, top_k={top_k}, threshold={self.match_threshold}")

                response = self.client.rpc(
                    'match_image_contents',
                    {
                        'query_embedding': query_embedding,  # List[float]のまま渡す（Supabaseが自動変換）
                        'match_threshold': self.match_threshold,
                        'match_count': top_k,
                        'filter_category': category
                    }
                ).execute()

                logger.info(f"Image search response received: {len(response.data) if response.data else 0} results")

                # デバッグ: 実際のレスポンスを確認
                if response.data and len(response.data) > 0:
                    logger.info(f"Supabase image result - Keys: {list(response.data[0].keys())}")
                    logger.info(f"Supabase image result - Sample data: source_file={response.data[0].get('source_file')}, page={response.data[0].get('page_number')}, category={response.data[0].get('category')}")
                else:
                    logger.warning("No image results returned from Supabase RPC")

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

    def register_pdf(self, filename: str, category: str, storage_path: Optional[str] = None):
        """
        PDFをregistered_pdfsテーブルに登録

        Args:
            filename: PDFファイル名
            category: カテゴリー
            storage_path: Supabase Storageパス（オプション）
        """
        if self.provider == 'supabase':
            try:
                data = {
                    'filename': filename,
                    'category': category
                }
                if storage_path:
                    data['storage_path'] = storage_path

                self.client.table(self.pdf_table).upsert(data).execute()
                logger.info(f"Registered PDF in Supabase: {filename} (storage_path: {storage_path})")
            except Exception as e:
                logger.error(f"Error registering PDF in Supabase: {e}")
                raise

    def upload_pdf_to_storage(self, pdf_path: str, filename: str, category: str) -> str:
        """
        PDFファイルをSupabase Storageにアップロード

        Args:
            pdf_path: ローカルのPDFファイルパス
            filename: PDFファイル名
            category: カテゴリー

        Returns:
            str: Storageパス
        """
        if self.provider != 'supabase':
            logger.warning("PDF upload to storage is only supported for Supabase provider")
            return ""

        try:
            from pathlib import Path
            import hashlib
            from datetime import datetime

            # Storageパスを生成（日本語を避けるため、ハッシュベースのパスを使用）
            # カテゴリーと日時のハッシュでディレクトリ作成
            category_hash = hashlib.md5(category.encode('utf-8')).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d")

            # ファイル名の拡張子を保持
            file_extension = Path(filename).suffix
            filename_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:16]

            # 英数字のみのパスを生成: cat_{hash}/file_{hash}_{timestamp}.pdf
            storage_path = f"cat_{category_hash}/file_{filename_hash}_{timestamp}{file_extension}"

            # PDFファイルを読み込み
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # Supabase Storageにアップロード
            self.client.storage.from_(self.pdf_storage_bucket).upload(
                storage_path,
                pdf_bytes,
                file_options={"content-type": "application/pdf", "upsert": "true"}
            )
            logger.info(f"Uploaded PDF to Supabase Storage: {storage_path}")

            return storage_path

        except Exception as e:
            logger.error(f"Error uploading PDF to Supabase Storage: {e}")
            raise

    def get_pdf_url_from_storage(self, filename: str) -> Optional[str]:
        """
        Supabase StorageからPDFの署名付きURLを取得

        Args:
            filename: PDFファイル名

        Returns:
            str: 署名付きURL（有効期限: 1時間）、取得失敗時はNone
        """
        if self.provider != 'supabase':
            return None

        try:
            # registered_pdfsテーブルからstorage_pathを取得
            response = self.client.table(self.pdf_table)\
                .select('storage_path')\
                .eq('filename', filename)\
                .execute()

            if not response.data or len(response.data) == 0:
                logger.warning(f"PDF not found in database: {filename}")
                return None

            storage_path = response.data[0].get('storage_path')
            if not storage_path:
                logger.warning(f"No storage_path found for PDF: {filename}")
                return None

            # 署名付きURLを生成（有効期限: 3600秒 = 1時間）
            url_response = self.client.storage.from_(self.pdf_storage_bucket)\
                .create_signed_url(storage_path, 3600)

            if url_response and 'signedURL' in url_response:
                logger.info(f"Generated signed URL for PDF: {filename}")
                return url_response['signedURL']
            else:
                logger.error(f"Failed to generate signed URL for PDF: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error getting PDF URL from Supabase Storage: {e}")
            return None

    def download_pdf_from_storage(self, filename: str, destination_path: str) -> bool:
        """
        Supabase StorageからPDFをダウンロード

        Args:
            filename: PDFファイル名
            destination_path: ダウンロード先パス

        Returns:
            bool: 成功時True、失敗時False
        """
        if self.provider != 'supabase':
            return False

        try:
            # registered_pdfsテーブルからstorage_pathを取得
            response = self.client.table(self.pdf_table)\
                .select('storage_path')\
                .eq('filename', filename)\
                .execute()

            if not response.data or len(response.data) == 0:
                logger.warning(f"PDF not found in database: {filename}")
                return False

            storage_path = response.data[0].get('storage_path')
            if not storage_path:
                logger.warning(f"No storage_path found for PDF: {filename}")
                return False

            # PDFをダウンロード
            pdf_bytes = self.client.storage.from_(self.pdf_storage_bucket)\
                .download(storage_path)

            # ファイルに保存
            with open(destination_path, 'wb') as f:
                f.write(pdf_bytes)

            logger.info(f"Downloaded PDF from Supabase Storage: {filename} -> {destination_path}")
            return True

        except Exception as e:
            logger.error(f"Error downloading PDF from Supabase Storage: {e}")
            return False

    def debug_inspect_data(self, category: str, limit: int = 5) -> Dict[str, Any]:
        """
        デバッグ用: カテゴリーのデータを確認

        Args:
            category: カテゴリー名
            limit: 取得する件数

        Returns:
            dict: サンプルデータ
        """
        if self.provider != 'supabase':
            logger.warning("debug_inspect_data is only supported for Supabase")
            return {}

        try:
            result = {
                'category': category,
                'text_chunks': [],
                'images': []
            }

            # テキストチャンクのサンプル取得
            text_response = self.client.table(self.text_table)\
                .select('id, content, source_file, page_number, category')\
                .eq('category', category)\
                .limit(limit)\
                .execute()

            if text_response.data:
                result['text_chunks'] = text_response.data
                logger.info(f"📊 DEBUG: Sample text chunks for '{category}':")
                for i, chunk in enumerate(text_response.data[:3], 1):
                    logger.info(f"  [{i}] {chunk['source_file']} (page {chunk['page_number']})")
                    logger.info(f"      Content preview: {chunk['content'][:100]}...")

            # 画像のサンプル取得
            image_response = self.client.table(self.image_table)\
                .select('id, content, source_file, page_number, category, content_type')\
                .eq('category', category)\
                .limit(limit)\
                .execute()

            if image_response.data:
                result['images'] = image_response.data
                logger.info(f"📊 DEBUG: Sample images for '{category}':")
                for i, img in enumerate(image_response.data[:3], 1):
                    logger.info(f"  [{i}] {img['source_file']} (page {img['page_number']}, type: {img.get('content_type', 'image')})")
                    logger.info(f"      Description preview: {img['content'][:100]}...")

            # embeddingフィールドが存在するか確認
            text_with_emb = self.client.table(self.text_table)\
                .select('id, embedding')\
                .eq('category', category)\
                .limit(1)\
                .execute()

            if text_with_emb.data and len(text_with_emb.data) > 0:
                embedding = text_with_emb.data[0].get('embedding')
                if embedding:
                    logger.info(f"✅ DEBUG: Embedding exists, dimension: {len(embedding)}")
                    logger.info(f"🔍 DEBUG: Embedding type: {type(embedding)}")
                    logger.info(f"🔍 DEBUG: First 5 elements: {embedding[:5]}")

                    # 異常な次元数の場合は警告
                    if len(embedding) != 3072:
                        logger.error(f"❌ DEBUG: ABNORMAL embedding dimension! Expected 3072, got {len(embedding)}")
                        logger.error(f"   This will cause vector search to fail!")
                else:
                    logger.error(f"❌ DEBUG: Embedding field is NULL!")

            return result

        except Exception as e:
            logger.error(f"Error inspecting data: {e}")
            return {}
