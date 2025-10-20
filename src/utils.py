"""
Utility functions for the PDF RAG system
"""

import logging
import yaml
import base64
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    ログ設定をセットアップ

    Args:
        config: ログ設定を含む設定辞書

    Returns:
        logger: 設定済みロガー
    """
    log_config = config.get('logging', {})

    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', './logs/app.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    設定ファイルを読み込み

    Args:
        config_path: config.yamlファイルのパス

    Returns:
        config: 設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_environment():
    """環境変数を読み込み"""
    load_dotenv()


def ensure_directories():
    """必要なディレクトリを作成"""
    directories = [
        "data/uploaded_pdfs",
        "data/extracted_images",
        "data/chroma_db",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def encode_pdf_to_base64(pdf_path: str) -> str:
    """
    PDFファイルをbase64エンコード

    Args:
        pdf_path: PDFファイルのパス

    Returns:
        str: base64エンコードされた文字列
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding PDF to base64: {e}")
        raise
