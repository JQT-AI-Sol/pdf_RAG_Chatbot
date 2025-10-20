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

        # Gemini API初期化
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = self.gemini_config.get("model_vision", "gemini-2.5-flash")

    @observe(name="vision_analysis")
    def analyze_image(self, image_path: str, content_type: str = "table") -> Dict[str, Any]:
        """
        画像（表・グラフ）を解析（Gemini使用）

        Args:
            image_path: 画像ファイルのパス
            content_type: コンテンツタイプ ('table' or 'graph')

        Returns:
            dict: 解析結果
        """
        logger.info(f"Analyzing {content_type} image with Gemini: {image_path}")

        try:
            # 画像を読み込み
            image = Image.open(image_path)

            # プロンプト選択
            if content_type == "table":
                prompt = self.vision_config.get("analysis_prompt_table", "")
            else:
                prompt = self.vision_config.get("analysis_prompt_graph", "")

            # Gemini API呼び出し
            model = genai.GenerativeModel(self.model)
            response = model.generate_content([prompt, image])

            analysis_result = response.text

            result = {
                "content_type": content_type,
                "description": analysis_result,
                "image_path": image_path,
                "model": self.model,
            }

            logger.info(f"Successfully analyzed {content_type} with Gemini")
            return result

        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
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
