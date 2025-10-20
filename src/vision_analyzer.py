"""
Vision AI module for analyzing tables and graphs using GPT-4o/GPT-5
"""

import logging
import os
import base64
from typing import Dict, Any
from openai import OpenAI
from pathlib import Path


logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Vision AIを使用して表・グラフを解析するクラス"""

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: Vision設定
        """
        self.config = config
        self.openai_config = config.get("openai", {})
        self.vision_config = config.get("vision", {})
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self.openai_config.get("model_chat", "gpt-5")

    def analyze_image(self, image_path: str, content_type: str = "table") -> Dict[str, Any]:
        """
        画像（表・グラフ）を解析

        Args:
            image_path: 画像ファイルのパス
            content_type: コンテンツタイプ ('table' or 'graph')

        Returns:
            dict: 解析結果
        """
        logger.info(f"Analyzing {content_type} image: {image_path}")

        try:
            # 画像をbase64エンコード
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # プロンプト選択
            if content_type == "table":
                prompt = self.vision_config.get("analysis_prompt_table", "")
            else:
                prompt = self.vision_config.get("analysis_prompt_graph", "")

            # Vision API呼び出し
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                        ],
                    }
                ],
                max_completion_tokens=self.openai_config.get("max_tokens", 1500),
            )

            analysis_result = response.choices[0].message.content

            result = {
                "content_type": content_type,
                "description": analysis_result,
                "image_path": image_path,
                "model": self.model,
            }

            logger.info(f"Successfully analyzed {content_type}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
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
