"""
RAG (Retrieval-Augmented Generation) engine
"""

import logging
import os
import base64
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI
from .prompt_templates import get_rag_prompt


logger = logging.getLogger(__name__)


class RAGEngine:
    """RAGエンジン - 検索と回答生成を統合"""

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
        self.search_config = config.get("search", {})

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self.openai_config.get("model_chat", "gpt-5")

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

    def query(self, question: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        質問に対して回答を生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）

        Returns:
            dict: 回答と参照元情報
        """
        logger.info(f"Processing query: {question} (category: {category})")

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

            # 5. マルチモーダルコンテンツを構築（テキスト + 画像）
            content = [{"type": "text", "text": prompt}]

            # 検索結果から画像を追加
            image_count = 0
            logger.debug(f"Number of image results: {len(search_results.get('images', []))}")
            for idx, result in enumerate(search_results.get("images", [])):
                metadata = result.get("metadata", {})
                image_path = metadata.get("image_path")
                logger.debug(f"Image {idx}: path={image_path}, exists={Path(image_path).exists() if image_path else False}")

                if image_path and Path(image_path).exists():
                    try:
                        logger.debug(f"Adding image to prompt: {image_path}")
                        image_base64 = self._encode_image_to_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                        image_count += 1
                        logger.debug(f"Successfully added image {idx} (total: {image_count})")
                    except Exception as e:
                        logger.warning(f"Failed to encode image {image_path}: {e}")
                elif image_path:
                    logger.warning(f"Image path does not exist: {image_path}")
                else:
                    logger.warning(f"Image {idx} has no image_path in metadata")

            # 6. LLMで回答生成
            logger.info(f"Sending prompt to LLM (text: {len(prompt)} chars, images: {image_count})")
            logger.debug(f"Prompt preview: {prompt[:500]}...")
            logger.debug(f"Content structure: {len(content)} parts - {[c['type'] for c in content]}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは技術文書を理解する専門家アシスタントです。"},
                    {"role": "user", "content": content},
                ],
                max_completion_tokens=self.openai_config.get("max_tokens", 16000),
            )

            logger.info(f"LLM response received. Model used: {response.model}")
            logger.info(f"Finish reason: {response.choices[0].finish_reason}")

            answer = response.choices[0].message.content

            # レスポンス全体をログに出力（デバッグ用）
            if not answer or len(answer) == 0:
                logger.error(f"Empty answer! Response object: {response}")
                logger.error(f"Message content type: {type(answer)}, value: {repr(answer)}")

            # デバッグ: 回答内容を確認
            logger.info(f"Generated answer length: {len(answer) if answer else 0} characters")
            if answer:
                logger.info(f"Answer preview: {answer[:200]}...")
            else:
                logger.warning("Answer is empty!")

            # 6. 結果をフォーマット
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

    def query_stream(self, question: str, category: Optional[str] = None):
        """
        質問に対して回答をストリーミング生成

        Args:
            question: ユーザーの質問
            category: 検索対象カテゴリー（Noneの場合は全カテゴリー）

        Yields:
            dict: ストリーミングチャンクまたは最終結果
        """
        logger.info(f"Processing streaming query: {question} (category: {category})")

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

            # 5. マルチモーダルコンテンツを構築（テキスト + 画像）
            content = [{"type": "text", "text": prompt}]

            # 検索結果から画像を追加
            image_count = 0
            for result in search_results.get("images", []):
                metadata = result.get("metadata", {})
                image_path = metadata.get("image_path")

                if image_path and Path(image_path).exists():
                    try:
                        logger.debug(f"Adding image to streaming prompt: {image_path}")
                        image_base64 = self._encode_image_to_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                        image_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to encode image {image_path}: {e}")

            logger.info(f"Sending streaming prompt to LLM (text: {len(prompt)} chars, images: {image_count})")

            # 6. LLMでストリーミング回答生成
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは技術文書を理解する専門家アシスタントです。"},
                    {"role": "user", "content": content},
                ],
                max_completion_tokens=self.openai_config.get("max_tokens", 16000),
                stream=True,
            )

            # 6. ストリーミングチャンクをyield
            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield {"type": "chunk", "content": content}

            # 7. 最終結果をyield
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
