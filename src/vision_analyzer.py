"""
Vision AI module for analyzing tables and graphs using Gemini-2.5-flash
"""

import logging
import os
from typing import Dict, Any
from pathlib import Path
import google.generativeai as genai
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
    """Vision AIを使用して表・グラフを解析するクラス（Gemini対応）"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: Vision設定
        """
        self.config = config
        self.gemini_config = config.get("gemini", {})
        self.vision_config = config.get("vision", {})

        # Gemini APIキーの確認と設定
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_key_valid = bool(self.api_key)

        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable is not set - vision analysis will be disabled")
            logger.warning("To enable vision analysis, please set GEMINI_API_KEY in your .env file")
            self.model_name = self.gemini_config.get("model_vision", "gemini-2.5-flash")
            self.model = None
        else:
            # Native Gemini SDK初期化
            try:
                genai.configure(api_key=self.api_key)
                self.model_name = self.gemini_config.get("model_vision", "gemini-2.5-flash")
                self.model = genai.GenerativeModel(model_name=self.model_name)
                logger.info(f"VisionAnalyzer initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.api_key_valid = False
                self.model = None

    @observe(name="vision_analysis")
    def analyze_image(self, image_path: str, content_type: str = "table") -> Dict[str, Any]:
        """
        画像（表・グラフ）を解析（Native Gemini SDK使用）

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
        logger.info(f"Analyzing {content_type} image with Gemini: {image_path}")

        # APIキーチェック
        if not self.api_key_valid or self.model is None:
            error_msg = "GEMINI_API_KEY is not set or invalid. Vision analysis is disabled. Please set GEMINI_API_KEY in your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # 画像ファイルの存在確認
            if not Path(image_path).exists():
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # プロンプト選択
            if content_type == "table":
                prompt = self.vision_config.get("analysis_prompt_table", "")
            else:
                prompt = self.vision_config.get("analysis_prompt_graph", "")

            if not prompt:
                error_msg = f"No prompt configured for content_type: {content_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 画像を開く
            image = Image.open(image_path)
            logger.debug(f"Image loaded: {image_path}")

            # Gemini API呼び出し
            try:
                logger.debug(f"Calling Gemini with model: {self.model_name}")
                response = self.model.generate_content([prompt, image])

                if not response or not response.text:
                    error_msg = f"Empty response from Gemini API for image: {image_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                analysis_result = response.text
                logger.debug(f"Gemini API response length: {len(analysis_result)} characters")

            except Exception as e:
                error_msg = f"Gemini API call failed for {image_path}: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                raise

            result = {
                "content_type": content_type,
                "description": analysis_result,
                "image_path": image_path,
                "model": self.model_name,
            }

            logger.info(f"Successfully analyzed {content_type} with Gemini (result: {len(analysis_result)} chars)")
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
