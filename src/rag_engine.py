"""
RAG (Retrieval-Augmented Generation) engine with LangChain
"""

import logging
import os
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import google.generativeai as genai

from .prompt_templates import RAG_PROMPT_TEMPLATE
from .reranker import Reranker

logger = logging.getLogger(__name__)

# Langfuse統合
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class RAGEngine:
    """RAGエンジン - LangChain統合版（OpenAI & Gemini対応）"""

    def __init__(self, config: dict, vector_store, embedder):
        """
        初期化

        Args:
            config: 設定辞書
            vector_store: VectorStoreインスタンス
            embedder: TextEmbedderインスタンス
        """
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder

        self.openai_config = config.get("openai", {})
        self.gemini_config = config.get("gemini", {})
        self.search_config = config.get("search", {})
        self.langfuse_config = config.get("langfuse", {})
        self.chat_config = config.get("chat", {})
        self.rag_config = config.get("rag", {})
        self.reranking_config = config.get("reranking", {})

        # Langfuse有効化チェック
        langfuse_available = LANGFUSE_AVAILABLE
        config_enabled = self.langfuse_config.get("enabled", True)
        has_public_key = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        has_secret_key = bool(os.getenv("LANGFUSE_SECRET_KEY"))

        self.langfuse_enabled = langfuse_available and config_enabled and has_public_key and has_secret_key

        if self.langfuse_enabled:
            self.langfuse_handler = LangfuseCallbackHandler()
            logger.info("Langfuse tracing enabled")
        else:
            self.langfuse_handler = None
            logger.info(
                f"Langfuse tracing disabled (available={langfuse_available}, config={config_enabled}, public_key={has_public_key}, secret_key={has_secret_key})"
            )

        # LangChain LLM初期化
        self._init_llms()

        # Reranker初期化（有効な場合のみ）
        self.reranker = None
        if self.rag_config.get("enable_reranking", False):
            try:
                model_name = self.reranking_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.reranker = Reranker(model_name=model_name)
                logger.info(f"Reranker initialized: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Reranker: {e}")
                logger.warning("Continuing without reranking")

        # 会話履歴管理の設定
        self.max_history_messages = self.chat_config.get("max_history_messages", 10)
        self.include_images_in_history = self.chat_config.get("include_images_in_history", False)

        logger.info(f"RAGEngine initialized with max_history_messages={self.max_history_messages}")

    def _init_llms(self):
        """LLMの初期化 (OpenAI: LangChain, Gemini: Native SDK)"""
        # OpenAI (LangChain)
        openai_kwargs = {
            "model": self.openai_config.get("model_chat", "gpt-4o-mini"),
            "temperature": self.openai_config.get("temperature", 0.7),
            "max_tokens": self.openai_config.get("max_tokens", 16000),
            "streaming": True,
        }
        if self.langfuse_enabled:
            openai_kwargs["callbacks"] = [self.langfuse_handler]

        self.openai_llm = ChatOpenAI(**openai_kwargs)

        # Gemini (Native SDK - API Key方式)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=self.gemini_config.get("model_chat", "gemini-2.5-flash")
            )
            logger.info(f"Gemini initialized with model: {self.gemini_config.get('model_chat', 'gemini-2.5-flash')}")
        else:
            self.gemini_model = None
            logger.warning("GEMINI_API_KEY not set - Gemini will be unavailable")

        logger.info("LLMs initialized (OpenAI: LangChain, Gemini: Native SDK)")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        画像をbase64エンコード（ローカルファイルまたはSupabase Storage URL）

        Args:
            image_path: 画像ファイルのパスまたはStorage URL

        Returns:
            str: base64エンコードされた画像データ
        """
        try:
            from pathlib import Path
            import io

            # Supabase Storage URLの場合（category/filename形式）
            logger.info(f"📸 _encode_image_to_base64: image_path={image_path}, exists={Path(image_path).exists()}, has_slash={'/' in image_path}")
            if not Path(image_path).exists() and '/' in image_path:
                # Supabase Storageから画像をダウンロード
                try:
                    logger.info(f"📸 _encode_image_to_base64: Downloading from Supabase Storage: {image_path}")
                    storage_bucket = self.config.get('vector_store', {}).get('supabase', {}).get('storage_bucket', 'pdf-images')

                    # vector_storeインスタンスを取得（既に初期化済みのはず）
                    if hasattr(self, 'vector_store') and hasattr(self.vector_store, 'client'):
                        response = self.vector_store.client.storage.from_(storage_bucket).download(image_path)
                        logger.info(f"📸 _encode_image_to_base64: Successfully downloaded and encoded image from Storage")
                        return base64.b64encode(response).decode("utf-8")
                    else:
                        logger.error(f"Vector store client not available for Storage download")
                        raise ValueError("Cannot download from Storage: vector_store not initialized")

                except Exception as download_error:
                    logger.error(f"Failed to download image from Storage {image_path}: {download_error}")
                    raise

            # ローカルファイルの場合
            else:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def _limit_chat_history(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        会話履歴を最大メッセージ数に制限

        Args:
            chat_history: 会話履歴

        Returns:
            list: 制限後の会話履歴
        """
        if not chat_history or len(chat_history) <= self.max_history_messages:
            return chat_history

        # 最新のmax_history_messages件のみ保持
        limited = chat_history[-self.max_history_messages:]
        logger.info(f"Chat history limited: {len(chat_history)} -> {len(limited)} messages")
        return limited

    def _convert_history_to_messages(self, chat_history: List[Dict[str, str]]) -> List:
        """
        辞書形式の会話履歴をLangChainメッセージに変換

        Args:
            chat_history: 会話履歴

        Returns:
            list: LangChainメッセージのリスト
        """
        messages = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")

            # 画像を含むメッセージの場合、テキストのみ抽出
            if not self.include_images_in_history and isinstance(content, list):
                # contentがリストの場合、textパートのみ抽出
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                content = " ".join(text_parts)

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    def _rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerankingを使用して検索結果を再順位付け

        Args:
            query: 検索クエリ
            results: 検索結果のリスト
            top_k: 返却する結果数

        Returns:
            list: Reranking後の検索結果
        """
        if not self.reranker or not results:
            return results[:top_k]

        try:
            # 検索結果からテキストを抽出
            documents = [result.get("content", "") for result in results]

            # Rerankingを実行
            logger.debug(f"Reranking {len(documents)} documents with query: {query[:50]}...")
            reranked_indices_scores = self.reranker.rerank(query, documents, top_k=top_k)

            # インデックスに基づいて結果を並び替え
            reranked_results = []
            for idx, score in reranked_indices_scores:
                result = results[idx].copy()
                result["rerank_score"] = score  # Rerankingスコアを追加
                reranked_results.append(result)

            logger.info(
                f"Reranking completed: {len(results)} -> {len(reranked_results)} results",
                extra={"top_rerank_scores": [s for _, s in reranked_indices_scores[:3]]}
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            logger.warning("Falling back to original results")
            return results[:top_k]

    def query(self, question: str, category: Optional[str] = None, model_type: str = "openai", chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None) -> Dict[str, Any]:
        """
        質問に対して回答を生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            model_type: 使用するモデル ("openai" or "gemini")
            chat_history: 会話履歴（Noneの場合は会話履歴なし）
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Returns:
            dict: 回答と関連情報
        """
        try:
            # 会話履歴の制限
            if chat_history:
                chat_history = self._limit_chat_history(chat_history)
                logger.info(f"Using {len(chat_history)} messages from chat history")

            # 1. 質問のエンベディング取得
            query_embedding = self.embedder.embed_query(question)

            # 2. ベクトル検索（Reranking有効時はより多く取得）
            top_k_text = self.search_config.get("top_k_text", 5)
            if self.reranker:
                top_k_initial = self.reranking_config.get("top_k_initial", 10)
                logger.debug(f"Reranking enabled: fetching {top_k_initial} results for reranking")
            else:
                top_k_initial = top_k_text

            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                category=category,
                top_k=top_k_initial,
                search_type="both",
                query_text=question  # BM25ハイブリッド検索用
            )

            # 3. Reranking（有効な場合）
            if self.reranker and search_results.get("text"):
                top_k_final = self.reranking_config.get("top_k_final", 5)
                search_results["text"] = self._rerank_results(
                    query=question,
                    results=search_results["text"],
                    top_k=top_k_final
                )

            # 4. コンテキストの構築
            context_parts = []
            image_data_list = []

            # テキストコンテキスト
            for result in search_results.get("text", []):
                doc = result.get("content", "")
                source_file = result.get("source_file", "")
                page_number = result.get("page_number", "")
                context_parts.append(f"[{source_file} - Page {page_number}]\n{doc}")

            # 画像コンテキスト
            for result in search_results.get("images", []):
                image_path = result.get("path", "")
                # ローカルファイルまたはSupabase Storageパスの両方を許可
                if image_path:
                    image_data_list.append({
                        "path": image_path,
                        "description": result.get("description", ""),
                        "source": result.get("source_file", ""),
                        "page": result.get("page_number", ""),
                    })
                    logger.debug(f"Added image to image_data_list: {image_path}")

            context_text = "\n\n".join(context_parts)
            logger.info(f"📸 image_data_list contains {len(image_data_list)} images")

            # 5. LLMで回答生成
            if model_type == "openai":
                answer = self._generate_answer_openai(question, context_text, image_data_list, chat_history, uploaded_images)
            else:
                answer = self._generate_answer_gemini(question, context_text, image_data_list, chat_history, uploaded_images)

            return {
                "answer": answer,
                "sources": search_results,
                "context": context_text,
                "images": image_data_list,
            }

        except Exception as e:
            logger.error(f"Error in query: {e}")
            raise

    def _generate_answer_openai(self, question: str, context: str, image_data_list: List[Dict], chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None) -> str:
        """
        OpenAIで回答を生成

        Args:
            question: 質問
            context: コンテキスト
            image_data_list: 画像データのリスト（ベクトル検索結果）
            chat_history: 会話履歴
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Returns:
            str: 回答
        """
        # プロンプト構築
        prompt_text = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # メッセージ構築
        messages = [SystemMessage(content="あなたは資料を理解し、正確に回答する専門家アシスタントです。")]

        # 会話履歴を追加
        if chat_history:
            history_messages = self._convert_history_to_messages(chat_history)
            messages.extend(history_messages)

        # 画像を含むコンテンツの構築
        content_parts = [{"type": "text", "text": prompt_text}]

        # アップロードされた画像を追加（役割を明示）
        if uploaded_images:
            content_parts.append({
                "type": "text",
                "text": f"\n\n━━━ 以下、ユーザーが質問のために添付した画像（{len(uploaded_images[:5])}枚）━━━"
            })
            for uploaded_img in uploaded_images[:5]:  # 最大5枚
                # BytesIOオブジェクトをbase64エンコード
                uploaded_img.seek(0)  # ファイルポインタを先頭に戻す
                img_bytes = uploaded_img.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

        # ベクトル検索で取得した画像を追加（役割を明示）
        remaining_slots = 5 - len(uploaded_images) if uploaded_images else 5
        if image_data_list and remaining_slots > 0:
            content_parts.append({
                "type": "text",
                "text": "\n\n━━━ 以下、参考コンテキストとして検索された資料の画像 ━━━"
            })
            for img_data in image_data_list[:remaining_slots]:
                base64_image = self._encode_image_to_base64(img_data["path"])
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

        messages.append(HumanMessage(content=content_parts))

        # LLM呼び出し
        response = self.openai_llm.invoke(messages)
        return response.content

    def _generate_answer_gemini(self, question: str, context: str, image_data_list: List[Dict], chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None) -> str:
        """
        Geminiで回答を生成 (Native SDK使用)

        Args:
            question: 質問
            context: コンテキスト
            image_data_list: 画像データのリスト（ベクトル検索結果）
            chat_history: 会話履歴
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Returns:
            str: 回答
        """
        if not self.gemini_model:
            raise ValueError("Gemini model is not initialized. Please set GEMINI_API_KEY.")

        # プロンプト構築
        prompt_text = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 会話履歴を追加（簡易版）
        if chat_history:
            history_text = "\n\n【これまでの会話履歴】\n"
            for msg in chat_history:
                role_label = "ユーザー" if msg["role"] == "user" else "アシスタント"
                content = msg.get("content", "")
                # 画像を含む場合はテキストのみ抽出
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    content = " ".join(text_parts)
                history_text += f"{role_label}: {content}\n\n"
            prompt_text = history_text + "\n" + prompt_text

        # プロンプトテキストに画像の説明を追加
        if uploaded_images:
            prompt_text += f"\n\n━━━ 以下、ユーザーが質問のために添付した画像（{len(uploaded_images[:5])}枚）━━━"

        remaining_slots = 5 - len(uploaded_images) if uploaded_images else 5
        if image_data_list and remaining_slots > 0:
            prompt_text += "\n\n━━━ 以下、参考コンテキストとして検索された資料の画像 ━━━"

        # コンテンツの構築（テキストを先に、画像を後に）
        content_parts = [prompt_text]

        # アップロードされた画像を追加
        from PIL import Image
        if uploaded_images:
            for uploaded_img in uploaded_images[:5]:  # 最大5枚
                # BytesIOオブジェクトをPIL Imageに変換
                uploaded_img.seek(0)  # ファイルポインタを先頭に戻す
                image = Image.open(uploaded_img)
                content_parts.append(image)

        # ベクトル検索で取得した画像を追加
        logger.info(f"📸 Gemini: Processing {len(image_data_list)} images (remaining_slots={remaining_slots})")
        if image_data_list and remaining_slots > 0:
            for img_data in image_data_list[:remaining_slots]:
                img_path_str = img_data["path"]
                img_path = Path(img_path_str)
                logger.debug(f"📸 Gemini: Processing image {img_path_str}")

                # ローカルファイルが存在する場合
                if img_path.exists():
                    logger.info(f"📸 Gemini: Loading local image: {img_path_str}")
                    image = Image.open(img_path)
                    content_parts.append(image)
                # Supabase Storage URLの場合
                elif '/' in img_path_str:
                    try:
                        # Storageから画像をダウンロード
                        logger.info(f"📸 Gemini: Downloading image from Storage: {img_path_str}")
                        storage_bucket = self.config.get('vector_store', {}).get('supabase', {}).get('storage_bucket', 'pdf-images')

                        if hasattr(self, 'vector_store') and hasattr(self.vector_store, 'client'):
                            image_bytes = self.vector_store.client.storage.from_(storage_bucket).download(img_path_str)
                            from io import BytesIO
                            image = Image.open(BytesIO(image_bytes))
                            content_parts.append(image)
                            logger.info(f"📸 Gemini: Successfully added image from Storage")
                        else:
                            logger.warning(f"Cannot download image from Storage: vector_store not initialized")
                    except Exception as e:
                        logger.error(f"Failed to download/open image from Storage {img_path_str}: {e}", exc_info=True)
                else:
                    logger.warning(f"Image path does not exist and is not a Storage URL: {img_path_str}")

        # Gemini API呼び出し
        response = self.gemini_model.generate_content(content_parts)
        return response.text

    def query_stream(self, question: str, category: Optional[str] = None, model_type: str = "openai", chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None):
        """
        質問に対してストリーミングで回答を生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            model_type: 使用するモデル ("openai" or "gemini")
            chat_history: 会話履歴（Noneの場合は会話履歴なし）
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Yields:
            dict: 回答の断片と関連情報
        """
        try:
            # 会話履歴の制限
            if chat_history:
                chat_history = self._limit_chat_history(chat_history)
                logger.info(f"Using {len(chat_history)} messages from chat history (streaming)")

            # 1. 質問のエンベディング取得
            query_embedding = self.embedder.embed_query(question)

            # 2. ベクトル検索（Reranking有効時はより多く取得）
            top_k_text = self.search_config.get("top_k_text", 5)
            if self.reranker:
                top_k_initial = self.reranking_config.get("top_k_initial", 10)
                logger.debug(f"Reranking enabled (streaming): fetching {top_k_initial} results for reranking")
            else:
                top_k_initial = top_k_text

            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                category=category,
                top_k=top_k_initial,
                search_type="both",
                query_text=question  # BM25ハイブリッド検索用
            )

            # 3. Reranking（有効な場合）
            if self.reranker and search_results.get("text"):
                top_k_final = self.reranking_config.get("top_k_final", 5)
                search_results["text"] = self._rerank_results(
                    query=question,
                    results=search_results["text"],
                    top_k=top_k_final
                )

            # 4. コンテキストの構築
            context_parts = []
            image_data_list = []

            # テキストコンテキスト
            for result in search_results.get("text", []):
                doc = result.get("content", "")
                source_file = result.get("source_file", "")
                page_number = result.get("page_number", "")
                context_parts.append(f"[{source_file} - Page {page_number}]\n{doc}")

            # 画像コンテキスト
            for result in search_results.get("images", []):
                image_path = result.get("path", "")
                # ローカルファイルまたはSupabase Storageパスの両方を許可
                if image_path:
                    image_data_list.append({
                        "path": image_path,
                        "description": result.get("description", ""),
                        "source": result.get("source_file", ""),
                        "page": result.get("page_number", ""),
                    })
                    logger.debug(f"Added image to image_data_list: {image_path}")

            context_text = "\n\n".join(context_parts)
            logger.info(f"📸 image_data_list contains {len(image_data_list)} images")

            # 最初にコンテキスト情報を返す
            yield {
                "type": "context",
                "sources": search_results,
                "context": context_text,
                "images": image_data_list,
            }

            # 5. LLMでストリーミング回答生成
            if model_type == "openai":
                yield from self._stream_answer_openai(question, context_text, image_data_list, chat_history, uploaded_images)
            else:
                yield from self._stream_answer_gemini(question, context_text, image_data_list, chat_history, uploaded_images)

        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            raise

    def _stream_answer_openai(self, question: str, context: str, image_data_list: List[Dict], chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None):
        """
        OpenAIでストリーミング回答を生成

        Args:
            question: 質問
            context: コンテキスト
            image_data_list: 画像データのリスト（ベクトル検索結果）
            chat_history: 会話履歴
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Yields:
            dict: 回答の断片
        """
        # プロンプト構築
        prompt_text = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # メッセージ構築
        messages = [SystemMessage(content="あなたは資料を理解し、正確に回答する専門家アシスタントです。")]

        # 会話履歴を追加
        if chat_history:
            history_messages = self._convert_history_to_messages(chat_history)
            messages.extend(history_messages)

        # 画像を含むコンテンツの構築
        content_parts = [{"type": "text", "text": prompt_text}]

        # アップロードされた画像を追加（役割を明示）
        if uploaded_images:
            content_parts.append({
                "type": "text",
                "text": f"\n\n━━━ 以下、ユーザーが質問のために添付した画像（{len(uploaded_images[:5])}枚）━━━"
            })
            for uploaded_img in uploaded_images[:5]:  # 最大5枚
                # BytesIOオブジェクトをbase64エンコード
                uploaded_img.seek(0)  # ファイルポインタを先頭に戻す
                img_bytes = uploaded_img.read()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

        # ベクトル検索で取得した画像を追加（役割を明示）
        remaining_slots = 5 - len(uploaded_images) if uploaded_images else 5
        logger.info(f"📸 OpenAI: Processing {len(image_data_list)} images (remaining_slots={remaining_slots})")
        if image_data_list and remaining_slots > 0:
            content_parts.append({
                "type": "text",
                "text": "\n\n━━━ 以下、参考コンテキストとして検索された資料の画像 ━━━"
            })
            for img_data in image_data_list[:remaining_slots]:
                img_path = img_data["path"]
                logger.info(f"📸 OpenAI: Encoding image: {img_path}")
                try:
                    base64_image = self._encode_image_to_base64(img_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                    logger.info(f"📸 OpenAI: Successfully added image")
                except Exception as e:
                    logger.error(f"📸 OpenAI: Failed to encode image {img_path}: {e}", exc_info=True)

        messages.append(HumanMessage(content=content_parts))

        # LLMストリーミング呼び出し
        logger.debug("Starting OpenAI streaming...")
        for chunk in self.openai_llm.stream(messages):
            if chunk.content:
                yield {
                    "type": "chunk",
                    "content": chunk.content,
                }
        logger.debug("OpenAI streaming completed")

    def _stream_answer_gemini(self, question: str, context: str, image_data_list: List[Dict], chat_history: Optional[List[Dict[str, str]]] = None, uploaded_images: Optional[List] = None):
        """
        Geminiでストリーミング回答を生成 (Native SDK使用)

        Args:
            question: 質問
            context: コンテキスト
            image_data_list: 画像データのリスト（ベクトル検索結果）
            chat_history: 会話履歴
            uploaded_images: ユーザーがアップロードした画像のリスト（BytesIOオブジェクト）

        Yields:
            dict: 回答の断片
        """
        if not self.gemini_model:
            raise ValueError("Gemini model is not initialized. Please set GEMINI_API_KEY.")

        # プロンプト構築
        prompt_text = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 会話履歴を追加（簡易版）
        if chat_history:
            history_text = "\n\n【これまでの会話履歴】\n"
            for msg in chat_history:
                role_label = "ユーザー" if msg["role"] == "user" else "アシスタント"
                content = msg.get("content", "")
                # 画像を含む場合はテキストのみ抽出
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    content = " ".join(text_parts)
                history_text += f"{role_label}: {content}\n\n"
            prompt_text = history_text + "\n" + prompt_text

        # プロンプトテキストに画像の説明を追加
        if uploaded_images:
            prompt_text += f"\n\n━━━ 以下、ユーザーが質問のために添付した画像（{len(uploaded_images[:5])}枚）━━━"

        remaining_slots = 5 - len(uploaded_images) if uploaded_images else 5
        if image_data_list and remaining_slots > 0:
            prompt_text += "\n\n━━━ 以下、参考コンテキストとして検索された資料の画像 ━━━"

        # コンテンツの構築（テキストを先に、画像を後に）
        content_parts = [prompt_text]

        # アップロードされた画像を追加
        from PIL import Image
        if uploaded_images:
            for uploaded_img in uploaded_images[:5]:  # 最大5枚
                # BytesIOオブジェクトをPIL Imageに変換
                uploaded_img.seek(0)  # ファイルポインタを先頭に戻す
                image = Image.open(uploaded_img)
                content_parts.append(image)

        # ベクトル検索で取得した画像を追加
        logger.info(f"📸 Gemini (streaming): Processing {len(image_data_list)} images (remaining_slots={remaining_slots})")
        if image_data_list and remaining_slots > 0:
            for img_data in image_data_list[:remaining_slots]:
                img_path_str = img_data["path"]
                img_path = Path(img_path_str)
                logger.debug(f"📸 Gemini (streaming): Processing image {img_path_str}")

                # ローカルファイルが存在する場合
                if img_path.exists():
                    logger.info(f"📸 Gemini (streaming): Loading local image: {img_path_str}")
                    image = Image.open(img_path)
                    content_parts.append(image)
                # Supabase Storage URLの場合
                elif '/' in img_path_str:
                    try:
                        # Storageから画像をダウンロード
                        logger.info(f"📸 Gemini (streaming): Downloading image from Storage: {img_path_str}")
                        storage_bucket = self.config.get('vector_store', {}).get('supabase', {}).get('storage_bucket', 'pdf-images')

                        if hasattr(self, 'vector_store') and hasattr(self.vector_store, 'client'):
                            image_bytes = self.vector_store.client.storage.from_(storage_bucket).download(img_path_str)
                            from io import BytesIO
                            image = Image.open(BytesIO(image_bytes))
                            content_parts.append(image)
                            logger.info(f"📸 Gemini (streaming): Successfully added image from Storage")
                        else:
                            logger.warning(f"Cannot download image from Storage: vector_store not initialized")
                    except Exception as e:
                        logger.error(f"Failed to download/open image from Storage {img_path_str}: {e}", exc_info=True)
                else:
                    logger.warning(f"Image path does not exist and is not a Storage URL: {img_path_str}")

        # Gemini APIストリーミング呼び出し
        logger.debug("Starting Gemini streaming...")
        response = self.gemini_model.generate_content(content_parts, stream=True)
        for chunk in response:
            if chunk.text:
                yield {
                    "type": "chunk",
                    "content": chunk.text,
                }
        logger.debug("Gemini streaming completed")

    def get_top_reference_pages(
        self,
        search_results: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        検索結果から関連度の高いトップNページを抽出

        Args:
            search_results: ベクトル検索結果（rerankingスコア含む）
            top_n: 抽出するページ数（デフォルト: 5）

        Returns:
            list: トップNページの情報
                [{
                    "source_file": str,
                    "page_number": int,
                    "score": float,  # rerankスコア（なければNone）
                    "content_preview": str,  # 内容のプレビュー（最初の100文字）
                    "file_extension": str,  # ファイル拡張子（例: ".xlsx", ".pdf"）
                }]
        """
        top_pages = []
        seen_pages = set()  # (source_file, page_number) の重複チェック用

        # テキスト検索結果から抽出
        for result in search_results.get("text", []):
            source_file = result.get("source_file", "")
            page_number = result.get("page_number", 0)

            # ページ識別子
            page_id = (source_file, page_number)

            # 重複チェック
            if page_id in seen_pages:
                continue

            # ファイル拡張子を取得
            file_extension = Path(source_file).suffix.lower()

            # ページ情報を追加
            top_pages.append({
                "source_file": source_file,
                "page_number": page_number,
                "score": result.get("rerank_score"),  # rerankingスコア（なければNone）
                "content_preview": result.get("content", "")[:100],  # 最初の100文字
                "file_extension": file_extension,  # ファイル拡張子
            })

            seen_pages.add(page_id)

            # 目標数に達したら終了
            if len(top_pages) >= top_n:
                break

        logger.info(f"Extracted {len(top_pages)} unique pages from search results")
        return top_pages
