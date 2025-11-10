"""
Unified document processor - PDF、Word、Excel、PowerPoint、テキストを統一的に処理
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pdfplumber
from pptx import Presentation
from docx import Document

from src.pdf_processor import PDFProcessor
from src.word_processor import WordProcessor
from src.excel_processor import ExcelProcessor
from src.pptx_processor import PowerPointProcessor
from src.txt_processor import TextFileProcessor


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    複数のドキュメント形式を統一的に処理するクラス

    サポートするフォーマット:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - Excel (.xlsx, .xls)
    - PowerPoint (.pptx, .ppt)
    - Text (.txt)
    """

    # サポートする拡張子
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "word",
        ".doc": "word",
        ".xlsx": "excel",
        ".xls": "excel",
        ".pptx": "powerpoint",
        ".ppt": "powerpoint",
        ".txt": "text",
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
        self.powerpoint_processor = PowerPointProcessor(config)
        self.text_processor = TextFileProcessor(config)

        logger.info("DocumentProcessor initialized with support for: PDF, Word, Excel, PowerPoint, Text")

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
            elif file_type == "powerpoint":
                result = self.powerpoint_processor.process_powerpoint(file_path, category)
            elif file_type == "text":
                result = self.text_processor.process_text_file(file_path, category)
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

    def process_powerpoint(self, pptx_path: str, category: str) -> Dict[str, Any]:
        """
        PowerPointドキュメントを処理

        Args:
            pptx_path: PowerPointファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果
        """
        return self.powerpoint_processor.process_powerpoint(pptx_path, category)

    def process_text_file(self, txt_path: str, category: str) -> Dict[str, Any]:
        """
        テキストファイルを処理

        Args:
            txt_path: テキストファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果
        """
        return self.text_processor.process_text_file(txt_path, category)


def _get_office_page_count(office_path: Path) -> int:
    """
    Officeファイルのページ/スライド/シート数を取得

    Args:
        office_path: Officeファイルのパス

    Returns:
        int: ページ数（PowerPoint: スライド数、Word: ページ数概算、Excel: シート数）
    """
    suffix = office_path.suffix.lower()

    try:
        if suffix in ['.pptx', '.ppt']:
            # PowerPoint: スライド数
            prs = Presentation(office_path)
            return len(prs.slides)

        elif suffix in ['.docx', '.doc']:
            # Word: セクション数で概算（正確なページ数は取得困難）
            doc = Document(office_path)
            # Wordの正確なページ数は取得が難しいので、段落数から推定
            # 1ページあたり約30段落と仮定
            paragraphs = len(doc.paragraphs)
            estimated_pages = max(1, paragraphs // 30)
            logger.info(f"   Word file: {paragraphs} paragraphs, estimated ~{estimated_pages} pages")
            return estimated_pages

        elif suffix in ['.xlsx', '.xls']:
            # Excel: シート数
            import openpyxl
            wb = openpyxl.load_workbook(office_path, read_only=True)
            return len(wb.sheetnames)

        else:
            logger.warning(f"Unknown Office file type: {suffix}")
            return 0

    except Exception as e:
        logger.warning(f"Failed to get page count for {office_path}: {e}")
        return 0


def convert_office_to_pdf(
    office_path: str,
    output_dir: Optional[str] = None,
    timeout: int = 60
) -> Optional[Path]:
    """
    LibreOfficeを使用してOfficeファイル（Word/Excel/PowerPoint）をPDFに変換

    Args:
        office_path: 変換元のOfficeファイルパス
        output_dir: 出力ディレクトリ（Noneの場合はdata/converted_pdfs）
        timeout: 変換タイムアウト（秒）

    Returns:
        Path: 変換後のPDFファイルパス、失敗した場合はNone

    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合
        subprocess.TimeoutExpired: 変換がタイムアウトした場合
    """
    office_path = Path(office_path)

    # ファイルの存在確認
    if not office_path.exists():
        error_msg = f"Office file not found: {office_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = Path("data/converted_pdfs")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力PDFファイル名
    pdf_filename = office_path.stem + ".pdf"
    output_path = output_dir / pdf_filename

    logger.info(f"🔄 Converting Office file to PDF: {office_path.name}")
    logger.info(f"   Output: {output_path}")

    try:
        # LibreOffice headlessモードでPDF変換
        # Windows/Linux両対応
        libreoffice_cmd = 'soffice' if os.name == 'nt' else 'libreoffice'

        # ファイルタイプに応じて適切なPDFエクスポートフィルタを選択
        suffix = office_path.suffix.lower()
        if suffix in ['.pptx', '.ppt']:
            # PowerPoint: Impressフィルタでテキストレイヤーを保持
            pdf_filter = 'pdf:impress_pdf_Export'
            logger.info(f"Using Impress PDF Export filter for PowerPoint")
        elif suffix in ['.docx', '.doc']:
            # Word: Writerフィルタでテキストレイヤーを保持
            pdf_filter = 'pdf:writer_pdf_Export'
            logger.info(f"Using Writer PDF Export filter for Word")
        elif suffix in ['.xlsx', '.xls']:
            # Excel: Calcフィルタ
            pdf_filter = 'pdf:calc_pdf_Export'
            logger.info(f"Using Calc PDF Export filter for Excel")
        else:
            # デフォルト
            pdf_filter = 'pdf'
            logger.warning(f"Using default PDF filter for {suffix}")

        result = subprocess.run(
            [
                libreoffice_cmd,
                '--headless',
                '--norestore',  # 以前のセッションを復元しない
                '--invisible',   # UIを完全に非表示
                '--nologo',      # スプラッシュ画面を表示しない
                '--convert-to', pdf_filter,
                '--outdir', str(output_dir),
                str(office_path)
            ],
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True
        )

        # 変換成功確認
        if output_path.exists():
            # 元のファイルと変換後のPDFのページ数を比較
            try:
                original_pages = _get_office_page_count(office_path)
                with pdfplumber.open(output_path) as pdf:
                    pdf_pages = len(pdf.pages)

                    logger.info(f"✅ PDF conversion successful: {output_path}")
                    logger.info(f"   Original pages: {original_pages}, PDF pages: {pdf_pages}")

                    if original_pages != pdf_pages:
                        logger.warning(f"⚠️ Page count mismatch! Original: {original_pages}, PDF: {pdf_pages}")
                        logger.warning(f"   Some pages may have been skipped during conversion")
                        logger.warning(f"   This may cause missing content in search results")

                    # テキストレイヤーの検証（最初の数ページをチェック）
                    pages_to_check = min(3, pdf_pages)  # 最初の3ページまでチェック
                    text_layer_ok = False

                    for i in range(pages_to_check):
                        page = pdf.pages[i]
                        text = page.extract_text()
                        words = page.extract_words()

                        if text and len(text.strip()) > 50 and len(words) > 5:
                            text_layer_ok = True
                            logger.info(f"✅ Text layer verified on page {i+1}: {len(text)} chars, {len(words)} words")
                            break

                    if not text_layer_ok:
                        logger.warning(f"⚠️⚠️⚠️ TEXT LAYER NOT DETECTED IN CONVERTED PDF! ⚠️⚠️⚠️")
                        logger.warning(f"   The PDF may have text as images, which will prevent highlighting.")
                        logger.warning(f"   This can happen with:")
                        logger.warning(f"   - Complex fonts or special characters")
                        logger.warning(f"   - Embedded images with text")
                        logger.warning(f"   - LibreOffice conversion limitations")
                        logger.warning(f"   Highlighting will use database chunks but may not work properly.")

            except Exception as e:
                logger.warning(f"⚠️ Failed to verify page count: {e}")

            return output_path
        else:
            logger.error(f"❌ PDF file not created: {output_path}")
            logger.error(f"   stdout: {result.stdout}")
            logger.error(f"   stderr: {result.stderr}")
            return None

    except FileNotFoundError:
        logger.error("❌ LibreOffice not found. Please install LibreOffice:")
        logger.error("   - Ubuntu/Debian: apt-get install libreoffice")
        logger.error("   - macOS: brew install libreoffice")
        logger.error("   - Windows: Download from https://www.libreoffice.org/")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"❌ PDF conversion timeout ({timeout}s): {office_path}")
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ PDF conversion failed: {office_path}")
        logger.error(f"   Error: {e}")
        logger.error(f"   stdout: {e.stdout}")
        logger.error(f"   stderr: {e.stderr}")
        return None

    except Exception as e:
        logger.error(f"❌ Unexpected error during PDF conversion: {e}")
        return None
