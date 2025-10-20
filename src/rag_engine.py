"""
RAG (Retrieval-Augmented Generation) engine
"""

import logging
import os
import base64
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
from PIL import Image
from .prompt_templates import get_rag_prompt

logger = logging.getLogger(__name__)

# Langfuse統合
try:
    from langfuse.openai import OpenAI
    from langfuse import observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    OpenAI = OpenAIClient
    LANGFUSE_AVAILABLE = False
    # logger.warning()はこの時点ではまだloggerが設定されていない可能性があるため削除


class RAGEngine:
    """RAGエンジン - 検索と回答生成を統合（OpenAI & Gemini対応）"""

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

        # Langfuse有効化チェック
        langfuse_available = LANGFUSE_AVAILABLE
        config_enabled = self.langfuse_config.get("enabled", True)
        has_public_key = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        has_secret_key = bool(os.getenv("LANGFUSE_SECRET_KEY"))

        self.langfuse_enabled = langfuse_available and config_enabled and has_public_key and has_secret_key

        if self.langfuse_enabled:
            logger.info("Langfuse tracing enabled")
        else:
            logger.info(
                f"Langfuse tracing disabled (available={langfuse_available}, config={config_enabled}, public_key={has_public_key}, secret_key={has_secret_key})"
            )

        # OpenAI初期化（Langfuseラッパー使用）
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_model = self.openai_config.get("model_chat", "gpt-5")

        # Gemini初期化
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model_name = self.gemini_config.get("model_chat", "gemini-2.5-flash")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        画像をbase64エンコード

        Args:
            image_path: 画像ファイルのパス

        Returns:
            str: base64エンコードされた画像データ
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def query(self, question: str, category: Optional[str] = None, model_type: str = "openai") -> Dict[str, Any]:
        """
        質問に対して回答を生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            model_type: 使用するモデル ("openai" or "gemini")

        Returns:
            dict: 回答と参照元情報
        """
        logger.info(f"Processing query: {question} (category: {category}, model: {model_type})")

        # Note: Langfuse tracing is automatically handled by the OpenAI wrapper (langfuse.openai.OpenAI)
        # when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables are set.

        try:
            # 1. 質問をエンベディング化
            query_embedding = self.embedder.embed_text(question)

            # 2. ベクトル検索
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                category=category,
                top_k=self.search_config.get("top_k_images", 10),
                search_type="both",
            )

            # 3. コンテキストを構築
            context = self._build_context(search_results)

            # 4. プロンプトを生成
            prompt = get_rag_prompt(question, context)

            # 5. 検索結果から画像パスを抽出
            image_paths = []
            for result in search_results.get("images", []):
                metadata = result.get("metadata", {})
                image_path = metadata.get("image_path")
                if image_path and Path(image_path).exists():
                    image_paths.append(image_path)

            # 6. モデルに応じて回答生成
            if model_type == "gemini":
                answer = self._generate_answer_gemini(prompt, image_paths)
            else:  # デフォルトはOpenAI
                answer = self._generate_answer_openai(prompt, image_paths)

            # 7. 結果をフォーマット
            result = {
                "answer": answer,
                "sources": self._extract_sources(search_results),
                "search_results": search_results,
            }

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def query_stream(self, question: str, category: Optional[str] = None, model_type: str = "openai"):
        """
        質問に対して回答をストリーミング生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）
            model_type: 使用するモデル ("openai" or "gemini")

        Yields:
            dict: ストリーミングチャンクまたは最終結果
        """
        logger.info(f"Processing streaming query: {question} (category: {category}, model: {model_type})")

        # Note: Langfuse tracing is handled by @observe decorators on _generate_answer_gemini and _stream_gemini

        try:
            # 1. 質問をエンベディング化
            query_embedding = self.embedder.embed_text(question)

            # 2. ベクトル検索
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                category=category,
                top_k=self.search_config.get("top_k_images", 10),  # 画像も含むためtop_k_imagesを使用
                search_type="both",
            )

            # 3. コンテキストを構築
            context = self._build_context(search_results)

            # 4. プロンプトを生成
            prompt = get_rag_prompt(question, context)

            # 5. 検索結果から画像パスを抽出
            image_paths = []
            for result in search_results.get("images", []):
                metadata = result.get("metadata", {})
                image_path = metadata.get("image_path")
                if image_path and Path(image_path).exists():
                    image_paths.append(image_path)

            # 6. モデルに応じてストリーミング回答生成
            if model_type == "gemini":
                # Geminiストリーミング
                yield from self._stream_gemini(prompt, image_paths, search_results)
            else:
                # OpenAI用マルチモーダルコンテンツを構築（テキスト + 画像）
                content = [{"type": "text", "text": prompt}]

                # 画像を追加
                image_count = 0
                for image_path in image_paths:
                    try:
                        logger.debug(f"Adding image to streaming prompt: {image_path}")
                        image_base64 = self._encode_image_to_base64(image_path)
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        )
                        image_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to encode image {image_path}: {e}")

                # OpenAIストリーミング
                logger.info(f"Sending streaming prompt to OpenAI (text: {len(prompt)} chars, images: {image_count})")

                stream = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "あなたは技術文書を理解する専門家アシスタントです。"},
                        {"role": "user", "content": content},
                    ],
                    max_completion_tokens=self.openai_config.get("max_tokens", 16000),
                    stream=True,
                )

                # ストリーミングチャンクをyield
                full_answer = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_answer += chunk_content
                        yield {"type": "chunk", "content": chunk_content}

                # 最終結果をyield
                yield {
                    "type": "final",
                    "answer": full_answer,
                    "sources": self._extract_sources(search_results),
                    "search_results": search_results,
                }

            logger.info("Streaming query processed successfully")

        except Exception as e:
            logger.error(f"Error processing streaming query: {e}")
            raise

    def _generate_answer_openai(self, prompt: str, image_paths: List[str]) -> str:
        """
        OpenAI APIで回答を生成

        Args:
            prompt: プロンプトテキスト
            image_paths: 画像パスのリスト

        Returns:
            str: 生成された回答
        """
        # マルチモーダルコンテンツを構築
        content = [{"type": "text", "text": prompt}]

        # 画像を追加
        for image_path in image_paths:
            if Path(image_path).exists():
                try:
                    image_base64 = self._encode_image_to_base64(image_path)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
                except Exception as e:
                    logger.warning(f"Failed to encode image {image_path}: {e}")

        logger.info(
            f"Sending prompt to OpenAI (text: {len(prompt)} chars, images: {len([c for c in content if c['type'] == 'image_url'])})"
        )

        # OpenAI API呼び出し
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "あなたは技術文書を理解する専門家アシスタントです。"},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=self.openai_config.get("max_tokens", 16000),
        )

        logger.info(
            f"OpenAI response received. Model: {response.model}, Finish reason: {response.choices[0].finish_reason}"
        )

        answer = response.choices[0].message.content
        logger.info(f"Generated answer length: {len(answer) if answer else 0} characters")

        return answer

    @observe(name="gemini_generation", as_type="generation")
    def _generate_answer_gemini(self, prompt: str, image_paths: List[str]) -> str:
        """
        Gemini APIで回答を生成

        Args:
            prompt: プロンプトテキスト
            image_paths: 画像パスのリスト

        Returns:
            str: 生成された回答
        """
        # マルチモーダルコンテンツを構築
        content_parts = [prompt]

        # 画像を追加
        for image_path in image_paths:
            if Path(image_path).exists():
                try:
                    img = Image.open(image_path)
                    content_parts.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")

        logger.info(f"Sending prompt to Gemini (text: {len(prompt)} chars, images: {len(content_parts) - 1})")

        # Gemini API呼び出し
        generation_config = {
            "max_output_tokens": self.gemini_config.get("max_tokens", 8192),
            "temperature": self.gemini_config.get("temperature", 1.0),
        }
        model = genai.GenerativeModel(
            model_name=self.gemini_model_name,
            system_instruction="あなたは技術文書を理解する専門家アシスタントです。コンテキスト情報に基づいて正確に回答してください。",
        )
        response = model.generate_content(content_parts, generation_config=generation_config)

        answer = response.text
        logger.info(f"Gemini response received. Generated answer length: {len(answer) if answer else 0} characters")

        return answer

    @observe(name="gemini_streaming", as_type="generation")
    def _stream_gemini(
        self, prompt: str, image_paths: List[str], search_results: Dict[str, List[Dict[str, Any]]]
    ):
        """
        Gemini APIでストリーミング回答を生成（@observeデコレーターでLangfuseトレーシング）

        Args:
            prompt: プロンプトテキスト
            image_paths: 画像パスのリスト
            search_results: ベクトル検索結果

        Yields:
            dict: ストリーミングチャンクまたは最終結果
        """
        # マルチモーダルコンテンツを構築
        content_parts = [prompt]

        # 画像を追加
        for image_path in image_paths:
            if Path(image_path).exists():
                try:
                    img = Image.open(image_path)
                    content_parts.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")

        logger.info(f"Sending streaming prompt to Gemini (text: {len(prompt)} chars, images: {len(content_parts) - 1})")

        # Gemini API呼び出し（ストリーミング）
        generation_config = {
            "max_output_tokens": self.gemini_config.get("max_tokens", 8192),
            "temperature": self.gemini_config.get("temperature", 1.0),
        }
        model = genai.GenerativeModel(
            model_name=self.gemini_model_name,
            system_instruction="あなたは業務マニュアルを理解する専門家アシスタントです。コンテキスト情報に基づいて正確に回答してください。",
        )
        response = model.generate_content(content_parts, stream=True, generation_config=generation_config)

        # ストリーミングチャンクをyield
        full_answer = ""
        for chunk in response:
            if chunk.text:
                full_answer += chunk.text
                yield {"type": "chunk", "content": chunk.text}

        logger.info(f"Gemini streaming response completed. Generated answer length: {len(full_answer)} characters")

        # Langfuse Generation トレース（ストリーミング完了後）
        if trace:
            try:
                trace.generation(
                    name="gemini_streaming_completion",
                    model=self.gemini_model_name,
                    input=prompt,
                    output=full_answer,
                    metadata={
                        "image_count": len(content_parts) - 1,
                        "image_paths": image_paths if len(image_paths) <= 5 else image_paths[:5] + ["..."],
                        "provider": "google",
                        "temperature": generation_config["temperature"],
                        "max_tokens": generation_config["max_output_tokens"],
                        "streaming": True,
                    },
                )
                logger.info(f"Langfuse trace generation recorded for Gemini streaming")
            except Exception as e:
                logger.error(f"Failed to record Langfuse trace: {e}")

        # 最終結果をyield
        yield {
            "type": "final",
            "answer": full_answer,
            "sources": self._extract_sources(search_results),
            "search_results": search_results,
        }

    def _build_context(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        検索結果からコンテキストを構築

        Args:
            search_results: ベクトル検索結果

        Returns:
            str: コンテキスト文字列
        """
        context_parts = []

        # テキストコンテンツ
        for i, result in enumerate(search_results.get("text", []), 1):
            metadata = result.get("metadata", {})
            doc = result.get("document", "")
            source = f"{metadata.get('source_file', 'Unknown')} (p.{metadata.get('page_number', '?')})"

            context_parts.append(f"[テキスト {i}] (出典: {source})\n{doc}\n")

        # 画像コンテンツ
        for i, result in enumerate(search_results.get("images", []), 1):
            metadata = result.get("metadata", {})
            doc = result.get("document", "")
            source = f"{metadata.get('source_file', 'Unknown')} (p.{metadata.get('page_number', '?')})"
            content_type = metadata.get("content_type", "image")

            context_parts.append(f"[{content_type.upper()} {i}] (出典: {source})\n{doc}\n")

        return "\n---\n".join(context_parts) if context_parts else "該当する情報が見つかりませんでした。"

    def _extract_sources(self, search_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        検索結果から参照元情報を抽出

        Args:
            search_results: ベクトル検索結果

        Returns:
            list: 参照元情報のリスト
        """
        sources = []

        # テキストソース
        for result in search_results.get("text", []):
            metadata = result.get("metadata", {})
            sources.append(
                {
                    "type": "text",
                    "file": metadata.get("source_file", ""),
                    "page": metadata.get("page_number", 0),
                    "category": metadata.get("category", ""),
                }
            )

        # 画像ソース
        for result in search_results.get("images", []):
            metadata = result.get("metadata", {})
            sources.append(
                {
                    "type": metadata.get("content_type", "image"),
                    "file": metadata.get("source_file", ""),
                    "page": metadata.get("page_number", 0),
                    "category": metadata.get("category", ""),
                    "image_path": metadata.get("image_path", ""),
                }
            )

        return sources
