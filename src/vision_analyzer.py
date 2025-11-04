"""
Vision AI module for analyzing tables and graphs using GPT-5
"""

import logging
import os
import base64
import hashlib
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from PIL import Image

# Langfuse統合
try:
    from langfuse import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # ダミーデコレーター
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else observe()(args[0])
    LANGFUSE_AVAILABLE = False


logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Vision AIを使用して表・グラフを解析するクラス（OpenAI GPT-5対応）"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: Vision設定
        """
        self.config = config
        self.openai_config = config.get("openai", {})
        self.vision_config = config.get("vision", {})
        self.rag_config = config.get("rag", {})
        self.cache_config = config.get("cache", {}).get("vision", {})

        # キャッシュ設定
        self.cache_dir = None
        if self.rag_config.get("enable_vision_cache", False) or self.cache_config.get("enabled", False):
            try:
                cache_dir_path = self.cache_config.get("directory", "./cache/vision_analysis")
                self.cache_dir = Path(cache_dir_path)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Vision cache enabled: {cache_dir_path}")
            except Exception as e:
                logger.error(f"Failed to initialize vision cache: {e}")
                logger.warning("Continuing without vision caching")

        # OpenAI APIキーの確認と設定
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_key_valid = bool(self.api_key)

        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set - vision analysis will be disabled")
            logger.warning("To enable vision analysis, please set OPENAI_API_KEY in your .env file")
            self.model_name = self.openai_config.get("model_vision", "gpt-5")
            self.client = None
        else:
            # OpenAI クライアント初期化
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.model_name = self.openai_config.get("model_vision", "gpt-5")
                logger.info(f"VisionAnalyzer initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.api_key_valid = False
                self.client = None

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

    def _get_image_hash(self, image_path: str) -> str:
        """
        画像ファイルのSHA256ハッシュを計算

        Args:
            image_path: 画像ファイルのパス

        Returns:
            str: SHA256ハッシュ値
        """
        try:
            with open(image_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            raise

    @observe(name="vision_analysis")
    def analyze_image(self, image_path: str, content_type: str = "table") -> Dict[str, Any]:
        """
        画像（表・グラフ）を解析（OpenAI Vision API使用）

        Args:
            image_path: 画像ファイルのパス
            content_type: コンテンツタイプ ('table' or 'graph')

        Returns:
            dict: 解析結果

        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合
            ValueError: API応答が不正な場合、またはAPIキーが未設定の場合
            Exception: その他のエラー
        """
        logger.info(f"Analyzing {content_type} image with GPT-5: {image_path}")

        # Langfuseコンテキストのインポート（詳細トレース用）
        if LANGFUSE_AVAILABLE:
            try:
                from langfuse.decorators import langfuse_context
            except ImportError:
                langfuse_context = None
        else:
            langfuse_context = None

        # キャッシュチェック（ファイル存在確認の前に）
        if self.cache_dir:
            try:
                # 画像ファイルが存在する場合のみキャッシュチェック
                if Path(image_path).exists():
                    image_hash = self._get_image_hash(image_path)
                    cache_key = f"{image_hash}_{content_type}"
                    cache_file = self.cache_dir / f"{cache_key}.json"

                    if cache_file.exists():
                        try:
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cached_data = json.load(f)
                            logger.info(f"Vision cache hit: {cache_key[:16]}... (skipping API call)")
                            return cached_data['result']
                        except Exception as e:
                            logger.warning(f"Failed to load vision cache: {e}")
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")

        # APIキーチェック
        if not self.api_key_valid or self.client is None:
            error_msg = "OPENAI_API_KEY is not set or invalid. Vision analysis is disabled. Please set OPENAI_API_KEY in your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # 画像ファイルの存在確認
            if not Path(image_path).exists():
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # プロンプト選択
            if content_type in ["table", "full_page"]:
                # 表およびページ全体は平文テキスト形式で抽出
                prompt = self.vision_config.get("analysis_prompt_table", "")
            elif content_type == "graph":
                # グラフはJSON形式で構造化データを抽出
                prompt = self.vision_config.get("analysis_prompt_graph", "")
            else:
                # デフォルトは平文テキスト形式
                prompt = self.vision_config.get("analysis_prompt_table", "")

            if not prompt:
                error_msg = f"No prompt configured for content_type: {content_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # プロンプトの最初の100文字をログ出力（デバッグ用）
            logger.warning(f"[DEBUG] Using prompt (first 100 chars): {prompt[:100]}...")
            logger.warning(f"[DEBUG] Content type: {content_type}")

            # Langfuseにプロンプト詳細を記録
            if langfuse_context:
                try:
                    langfuse_context.update_current_trace(
                        metadata={
                            "content_type": content_type,
                            "image_path": image_path,
                            "model": self.model_name,
                            "prompt_length": len(prompt)
                        }
                    )
                    langfuse_context.update_current_observation(
                        input=prompt,  # プロンプト全文をLangfuseに記録
                        metadata={
                            "image_path": image_path,
                            "content_type": content_type,
                            "prompt_first_100_chars": prompt[:100]
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Langfuse context: {e}")

            # 画像をbase64エンコード
            logger.debug(f"Encoding image: {image_path}")
            base64_image = self._encode_image_to_base64(image_path)

            # 画像の拡張子を取得してMIMEタイプを決定
            image_ext = Path(image_path).suffix.lower()
            mime_type = "image/png" if image_ext == ".png" else "image/jpeg"

            # OpenAI API呼び出し
            try:
                logger.debug(f"Calling OpenAI with model: {self.model_name}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4096
                )

                if not response.choices or not response.choices[0].message.content:
                    error_msg = f"Empty response from OpenAI API for image: {image_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                raw_response = response.choices[0].message.content
                logger.debug(f"OpenAI API response length: {len(raw_response)} characters")

                # Langfuseに生応答を記録
                has_json_block = "```json" in raw_response.lower()
                starts_with_brace = raw_response.strip().startswith("{")

                if langfuse_context:
                    try:
                        langfuse_context.update_current_observation(
                            output=raw_response,  # OpenAIの生応答
                            metadata={
                                "response_length": len(raw_response),
                                "has_json_block": has_json_block,
                                "starts_with_brace": starts_with_brace,
                                "response_first_200_chars": raw_response[:200]
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update Langfuse with response: {e}")

                analysis_result = raw_response

            except Exception as e:
                error_msg = f"OpenAI API call failed for {image_path}: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                raise

            # 平文テキストプロンプトを使用した場合、モデルが勝手にJSON形式で返すことがあるので、
            # JSONブロックを検出して警告し、平文に変換する
            json_removed = False
            if content_type in ["table", "full_page"]:
                if "```json" in analysis_result.lower() or analysis_result.strip().startswith("{"):
                    logger.warning(f"[ISSUE] OpenAI returned JSON format despite plain text prompt for {content_type}")
                    logger.warning(f"[ISSUE] Original response (first 200 chars): {analysis_result[:200]}")

                    # JSONブロックを除去して平文に変換
                    # パターン1: ```json ... ``` ブロック
                    import re
                    cleaned = re.sub(r'```json.*?```', '', analysis_result, flags=re.DOTALL | re.IGNORECASE)
                    # パターン2: ```...``` ブロック（言語指定なし）
                    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
                    # 前後の説明文だけを残す
                    cleaned = cleaned.strip()

                    if cleaned and len(cleaned) > 50:
                        analysis_result = cleaned
                        json_removed = True
                        logger.warning(f"[FIX] Converted to plain text ({len(analysis_result)} chars)")
                    else:
                        logger.warning(f"[FIX] Could not extract plain text, keeping original")

            # Langfuseに後処理結果を記録
            if langfuse_context and json_removed:
                try:
                    langfuse_context.update_current_trace(
                        metadata={
                            "json_removed": True,
                            "original_length": len(raw_response),
                            "cleaned_length": len(analysis_result),
                            "final_result_first_200_chars": analysis_result[:200]
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Langfuse with cleanup info: {e}")

            result = {
                "content_type": content_type,
                "description": analysis_result,
                "image_path": image_path,
                "model": self.model_name,
            }

            # キャッシュに保存
            if self.cache_dir:
                try:
                    image_hash = self._get_image_hash(image_path)
                    cache_key = f"{image_hash}_{content_type}"
                    cache_file = self.cache_dir / f"{cache_key}.json"

                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'result': result,
                            'timestamp': datetime.now().isoformat(),
                            'content_type': content_type,
                            'image_hash': image_hash
                        }, f, ensure_ascii=False, indent=2)
                    logger.debug(f"Cached vision result: {cache_key[:16]}...")
                except Exception as e:
                    logger.warning(f"Failed to save vision cache: {e}")

            logger.info(f"Successfully analyzed {content_type} with GPT-5 (result: {len(analysis_result)} chars)")
            return result

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {type(e).__name__}: {str(e)}")
            raise

    def analyze_table(self, image_path: str) -> Dict[str, Any]:
        """
        表を解析

        Args:
            image_path: 表画像のパス

        Returns:
            dict: 解析結果
        """
        return self.analyze_image(image_path, content_type="table")

    def analyze_graph(self, image_path: str) -> Dict[str, Any]:
        """
        グラフを解析

        Args:
            image_path: グラフ画像のパス

        Returns:
            dict: 解析結果
        """
        return self.analyze_image(image_path, content_type="graph")
