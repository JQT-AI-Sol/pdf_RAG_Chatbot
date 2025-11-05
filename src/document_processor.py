"""
Unified document processor - PDF、Word、Excelを統一的に処理
"""

import logging
from pathlib import Path
from typing import Dict, Any

from src.pdf_processor import PDFProcessor
from src.word_processor import WordProcessor
from src.excel_processor import ExcelProcessor


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    複数のドキュメント形式を統一的に処理するクラス

    サポートするフォーマット:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - Excel (.xlsx, .xls)
    """

    # サポートする拡張子
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "word",
        ".doc": "word",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: 処理設定
        """
        self.config = config

        # 各プロセッサを初期化
        self.pdf_processor = PDFProcessor(config)
        self.word_processor = WordProcessor(config)
        self.excel_processor = ExcelProcessor(config)

        logger.info("DocumentProcessor initialized with support for: PDF, Word, Excel")

    def is_supported(self, file_path: str) -> bool:
        """
        ファイル形式がサポートされているか確認

        Args:
            file_path: ファイルパス

        Returns:
            bool: サポートされている場合True
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def get_file_type(self, file_path: str) -> str:
        """
        ファイルタイプを取得

        Args:
            file_path: ファイルパス

        Returns:
            str: ファイルタイプ ("pdf", "word", "excel") またはNone
        """
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext)

    def process_document(self, file_path: str, category: str) -> Dict[str, Any]:
        """
        ドキュメントを処理（形式を自動判定）

        Args:
            file_path: ドキュメントファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果（テキストチャンク、画像情報など）

        Raises:
            ValueError: サポートされていないファイル形式の場合
            FileNotFoundError: ファイルが存在しない場合
        """
        # ファイルの存在確認
        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # ファイルタイプを判定
        file_type = self.get_file_type(file_path)

        if not file_type:
            ext = Path(file_path).suffix.lower()
            supported = ", ".join(self.SUPPORTED_EXTENSIONS.keys())
            error_msg = f"Unsupported file format: {ext}. Supported formats: {supported}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Processing document: {file_path} (type: {file_type})")

        # 適切なプロセッサで処理
        try:
            if file_type == "pdf":
                result = self.pdf_processor.process_pdf(file_path, category)
            elif file_type == "word":
                result = self.word_processor.process_word(file_path, category)
            elif file_type == "excel":
                result = self.excel_processor.process_excel(file_path, category)
            else:
                # 理論上ここには到達しないはず
                raise ValueError(f"Unknown file type: {file_type}")

            # メタデータにファイルタイプを追加
            result["metadata"]["file_type"] = file_type

            logger.info(f"Successfully processed {file_type} document: {len(result['text_chunks'])} chunks, {len(result['images'])} images")

            return result

        except Exception as e:
            logger.error(f"Error processing {file_type} document {file_path}: {e}")
            raise

    def process_pdf(self, pdf_path: str, category: str) -> Dict[str, Any]:
        """
        PDFを処理（後方互換性のため）

        Args:
            pdf_path: PDFファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果
        """
        return self.pdf_processor.process_pdf(pdf_path, category)

    def process_word(self, word_path: str, category: str) -> Dict[str, Any]:
        """
        Wordドキュメントを処理

        Args:
            word_path: Wordファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果
        """
        return self.word_processor.process_word(word_path, category)

    def process_excel(self, excel_path: str, category: str) -> Dict[str, Any]:
        """
        Excelドキュメントを処理

        Args:
            excel_path: Excelファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果
        """
        return self.excel_processor.process_excel(excel_path, category)
