"""
PDF management module - PDF削除を統合的に管理
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class PDFManager:
    """PDFファイルとその関連データを管理するクラス"""

    def __init__(self, vector_store: VectorStore, config: dict):
        """
        初期化

        Args:
            vector_store: VectorStoreインスタンス
            config: 設定
        """
        self.vector_store = vector_store
        self.config = config

    def get_registered_pdfs(self):
        """
        登録済みPDFのリストを取得

        Returns:
            list: PDFファイルごとの情報
        """
        return self.vector_store.get_registered_pdfs()

    def delete_pdf(self, source_file: str) -> Dict[str, Any]:
        """
        PDFとその関連データを全て削除

        Args:
            source_file: 削除対象のPDFファイル名

        Returns:
            dict: 削除結果
                {
                    'success': bool,
                    'message': str,
                    'vector_deleted': dict,
                    'pdf_file_deleted': bool,
                    'images_deleted': bool,
                    'category_deleted': bool
                }
        """
        result = {
            'success': False,
            'message': '',
            'vector_deleted': {'text_deleted': 0, 'image_deleted': 0},
            'pdf_file_deleted': False,
            'images_deleted': False,
            'category_deleted': False
        }

        try:
            # PDFの情報を取得（削除前にカテゴリーを確認）
            pdf_list = self.vector_store.get_registered_pdfs()
            target_pdf = None
            for pdf in pdf_list:
                if pdf['source_file'] == source_file:
                    target_pdf = pdf
                    break

            if not target_pdf:
                result['message'] = f"PDF not found: {source_file}"
                return result

            category = target_pdf.get('category', '')

            # 1. VectorStoreからベクトルデータを削除
            logger.info(f"Deleting vector data for {source_file}")
            deleted_counts = self.vector_store.delete_by_source_file(source_file)
            result['vector_deleted'] = deleted_counts

            # 2. PDFファイルを削除（data/uploaded_pdfs/ と static/pdfs/ の両方）
            pdf_path = Path("data/uploaded_pdfs") / source_file
            static_pdf_path = Path("static/pdfs") / source_file

            files_deleted = False
            if pdf_path.exists():
                pdf_path.unlink()
                files_deleted = True
                logger.info(f"Deleted PDF file: {pdf_path}")
            else:
                logger.warning(f"PDF file not found: {pdf_path}")

            if static_pdf_path.exists():
                static_pdf_path.unlink()
                files_deleted = True
                logger.info(f"Deleted static PDF file: {static_pdf_path}")
            else:
                logger.warning(f"Static PDF file not found: {static_pdf_path}")

            result['pdf_file_deleted'] = files_deleted

            # 3. 抽出された画像ディレクトリを削除
            pdf_stem = Path(source_file).stem
            images_dir = Path(f"data/extracted_images/{pdf_stem}")
            if images_dir.exists():
                shutil.rmtree(images_dir)
                result['images_deleted'] = True
                logger.info(f"Deleted images directory: {images_dir}")
            else:
                logger.warning(f"Images directory not found: {images_dir}")

            # 4. カテゴリー内のPDFをチェック
            if category:
                remaining_pdfs = self.vector_store.get_registered_pdfs()
                category_has_pdfs = any(pdf['category'] == category for pdf in remaining_pdfs)

                # カテゴリー内にPDFがなくなった場合の記録（カテゴリーはSupabaseから動的に取得されるため削除不要）
                if not category_has_pdfs:
                    result['category_deleted'] = True
                    logger.info(f"No more PDFs in category: {category}")

            result['success'] = True
            result['message'] = f"Successfully deleted {source_file} and all related data"
            logger.info(result['message'])

        except Exception as e:
            result['message'] = f"Error deleting PDF: {str(e)}"
            logger.error(result['message'], exc_info=True)

        return result

    def get_pdf_info(self, source_file: str) -> Optional[Dict[str, Any]]:
        """
        特定のPDFの情報を取得

        Args:
            source_file: PDFファイル名

        Returns:
            dict: PDF情報（見つからない場合はNone）
        """
        pdf_list = self.get_registered_pdfs()
        for pdf in pdf_list:
            if pdf['source_file'] == source_file:
                return pdf
        return None

    def get_pdfs_by_category(self, category: str):
        """
        特定のカテゴリーに属するPDFのリストを取得

        Args:
            category: カテゴリー名

        Returns:
            list: PDFファイルのリスト
        """
        pdf_list = self.get_registered_pdfs()
        return [pdf for pdf in pdf_list if pdf.get('category') == category]
