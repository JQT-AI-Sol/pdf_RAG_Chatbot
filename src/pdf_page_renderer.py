"""
PDF Page Rendering Module

PDFの特定ページを画像に変換してStreamlit UIで表示する機能を提供
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import streamlit as st
from PIL import Image, ImageDraw
import pdfplumber

logger = logging.getLogger(__name__)

# pdf2imageの動作確認（popplerが必要）
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    logger.info("=" * 60)
    logger.info("✅ PDF2IMAGE_AVAILABLE = True")
    logger.info("✅ pdf2image is available - PDF page rendering ENABLED")
    logger.info("✅ poppler-utils found - Highlights will work")
    logger.info("=" * 60)
except Exception as e:
    logger.warning("=" * 60)
    logger.warning("❌ PDF2IMAGE_AVAILABLE = False")
    logger.warning(f"❌ pdf2image not available: {e}")
    logger.warning("❌ PDF page preview will be DISABLED")
    logger.warning("💡 Check packages.txt contains: poppler-utils")
    logger.warning("=" * 60)

# 画像生成の設定
DEFAULT_DPI = 150  # 標準品質
DEFAULT_WIDTH = 1000  # ピクセル幅


def get_pdf_path(source_file: str, vector_store) -> Optional[Path]:
    """
    PDFファイルのローカルパスを取得（必要に応じてSupabase Storageからダウンロード）

    Args:
        source_file: PDFファイル名
        vector_store: VectorStoreインスタンス（Supabase Storage連携用）

    Returns:
        Path: PDFのローカルパス、取得失敗時はNone
    """
    # まずローカルストレージをチェック
    local_pdf_path = Path("data/uploaded_pdfs") / source_file

    if local_pdf_path.exists():
        logger.info(f"Using local PDF: {local_pdf_path}")
        return local_pdf_path

    # Supabase Storageから一時ディレクトリにダウンロード
    if vector_store and vector_store.provider == 'supabase':
        try:
            temp_dir = Path(tempfile.gettempdir()) / "rag_pdf_cache"
            temp_dir.mkdir(exist_ok=True)
            temp_pdf_path = temp_dir / source_file

            # 既にキャッシュされているかチェック
            if temp_pdf_path.exists():
                logger.info(f"Using cached PDF: {temp_pdf_path}")
                return temp_pdf_path

            # Supabase Storageからダウンロード
            success = vector_store.download_pdf_from_storage(source_file, str(temp_pdf_path))

            if success and temp_pdf_path.exists():
                logger.info(f"Downloaded PDF from Supabase Storage: {temp_pdf_path}")
                return temp_pdf_path
            else:
                logger.error(f"Failed to download PDF from Supabase Storage: {source_file}")
                return None

        except Exception as e:
            logger.error(f"Error accessing PDF from Supabase Storage: {e}")
            return None

    logger.error(f"PDF not found: {source_file}")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def extract_page_as_image(
    source_file: str,
    page_number: int,
    _vector_store,  # Streamlitキャッシュ用にアンダースコア付き
    dpi: int = DEFAULT_DPI,
    target_width: int = DEFAULT_WIDTH
) -> Optional[Image.Image]:
    """
    PDFの特定ページを画像に変換

    Args:
        source_file: PDFファイル名
        page_number: ページ番号（1始まり）
        _vector_store: VectorStoreインスタンス（キャッシュキーから除外）
        dpi: 解像度（デフォルト: 150）
        target_width: 画像幅（ピクセル、デフォルト: 1000）

    Returns:
        PIL.Image: 変換された画像、失敗時はNone
    """
    if not PDF2IMAGE_AVAILABLE:
        logger.warning("PDF page rendering is disabled (poppler not installed)")
        return None

    try:
        # PDFのローカルパスを取得
        pdf_path = get_pdf_path(source_file, _vector_store)
        if not pdf_path:
            logger.error(f"Failed to get PDF path: {source_file}")
            return None

        # PDFページを画像に変換（指定ページのみ）
        # page_numberは1始まりだが、first_pageとlast_pageも1始まりで指定
        logger.info(f"Converting page {page_number} of {source_file} to image (DPI: {dpi})")

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            fmt='png'
        )

        if not images or len(images) == 0:
            logger.error(f"No image generated for page {page_number} of {source_file}")
            return None

        image = images[0]

        # 画像サイズを調整（横幅を指定幅に合わせて縦横比維持）
        if image.width > target_width:
            aspect_ratio = image.height / image.width
            new_height = int(target_width * aspect_ratio)
            image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {target_width}x{new_height}")

        logger.info(f"Successfully converted page {page_number} of {source_file}")
        return image

    except Exception as e:
        logger.error(f"Error converting PDF page to image: {e}", exc_info=True)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def extract_multiple_pages(
    source_file: str,
    page_numbers: list[int],
    _vector_store,
    dpi: int = DEFAULT_DPI,
    target_width: int = DEFAULT_WIDTH
) -> dict[int, Optional[Image.Image]]:
    """
    複数ページを一括で画像に変換（効率化）

    Args:
        source_file: PDFファイル名
        page_numbers: ページ番号のリスト（1始まり）
        _vector_store: VectorStoreインスタンス
        dpi: 解像度
        target_width: 画像幅

    Returns:
        dict: {page_number: Image.Image}の辞書
    """
    if not PDF2IMAGE_AVAILABLE:
        logger.warning("PDF page rendering is disabled (poppler not installed)")
        return {page: None for page in page_numbers}

    results = {}

    if not page_numbers:
        return results

    try:
        # PDFのローカルパスを取得
        pdf_path = get_pdf_path(source_file, _vector_store)
        if not pdf_path:
            logger.error(f"Failed to get PDF path: {source_file}")
            return {page: None for page in page_numbers}

        # ページ番号でソート（効率的な抽出のため）
        sorted_pages = sorted(page_numbers)

        logger.info(f"Converting {len(sorted_pages)} pages of {source_file} to images")

        # 各ページを個別に変換（メモリ効率のため）
        for page_num in sorted_pages:
            try:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                    fmt='png'
                )

                if images and len(images) > 0:
                    image = images[0]

                    # リサイズ
                    if image.width > target_width:
                        aspect_ratio = image.height / image.width
                        new_height = int(target_width * aspect_ratio)
                        image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)

                    results[page_num] = image
                    logger.debug(f"Converted page {page_num}")
                else:
                    results[page_num] = None
                    logger.warning(f"No image for page {page_num}")

            except Exception as e:
                logger.error(f"Error converting page {page_num}: {e}")
                results[page_num] = None

        logger.info(f"Successfully converted {len([v for v in results.values() if v is not None])}/{len(sorted_pages)} pages")
        return results

    except Exception as e:
        logger.error(f"Error in batch page conversion: {e}", exc_info=True)
        return {page: None for page in page_numbers}


def tokenize_query(query: str) -> List[str]:
    """
    検索クエリをトークン化（日本語形態素解析）

    Args:
        query: 検索クエリ

    Returns:
        list: トークンのリスト
    """
    if not query or not query.strip():
        logger.warning("Empty query provided for tokenization")
        return []

    try:
        import MeCab
        mecab = MeCab.Tagger("-Owakati")
        tokens = mecab.parse(query).strip().split()
        logger.info(f"✅ MeCab tokenization successful: '{query}' -> {tokens}")
        return tokens
    except ImportError as e:
        logger.warning(f"⚠️ MeCab not available ({e}), using Japanese-aware fallback")
        # MeCabが使えない場合は日本語対応フォールバック
        tokens = _japanese_aware_tokenize(query)
        logger.info(f"Fallback tokenization: '{query}' -> {tokens}")
        return tokens
    except Exception as e:
        logger.error(f"❌ Error tokenizing query: {e}")
        tokens = _japanese_aware_tokenize(query)
        logger.info(f"Error fallback tokenization: '{query}' -> {tokens}")
        return tokens


def _japanese_aware_tokenize(text: str) -> List[str]:
    """
    日本語対応の簡易トークン化（MeCabが使えない場合のフォールバック）

    Args:
        text: トークン化するテキスト

    Returns:
        list: トークンのリスト
    """
    import re

    # 記号・句読点を削除
    text = re.sub(r'[「」『』【】、。？！・\s]+', ' ', text)

    # 英数字と日本語文字を分離
    tokens = []
    current_token = ""

    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char
            # 2-3文字ごとに区切る（日本語の場合）
            if len(current_token) >= 3 and not char.isascii():
                tokens.append(current_token)
                current_token = ""

    if current_token:
        tokens.append(current_token)

    # 短すぎるトークンを除外（1文字のひらがな・カタカナなど）
    tokens = [t for t in tokens if len(t) >= 2 or t.isalnum()]

    return tokens


def extract_keywords_llm(query: str, _rag_engine) -> List[str]:
    """
    LLMを使用してクエリから重要キーワードのみを抽出

    Args:
        query: ユーザークエリ
        _rag_engine: RAGEngineインスタンス（LLMアクセス用）

    Returns:
        list: 抽出された重要キーワードのリスト
    """
    if not query or not query.strip():
        logger.warning("Empty query provided for LLM keyword extraction")
        return []

    if not _rag_engine:
        logger.warning("RAGEngine not available for LLM keyword extraction")
        return tokenize_query(query)

    try:
        from langchain_core.messages import HumanMessage

        prompt = f"""以下の質問から、PDFページ上でハイライトすべき重要なキーワードのみを抽出してください。

**除外すべきもの:**
- 助詞（の、は、を、が、に、で、と、や、から、まで、より、へ）
- 指示語（この、その、あの、どの、どれ、いつ、どこ）
- 一般的な動詞（する、ある、いる、なる、行う、示す）
- 疑問詞単体（何、誰、いつ、どこ、なぜ、どう）
- 1-2文字の断片や活用語尾

**抽出すべきもの:**
- 名詞（特に固有名詞、専門用語、組織名、人名）
- 重要な動詞・形容詞（核心的な動作や状態）
- 数値や日付
- 複合語（例: 「因果関係」「認定否認」）

質問: {query}

重要キーワードをカンマ区切りで出力してください（説明不要、キーワードのみ）:"""

        # LLM呼び出し（temperature=0で確定的な出力）
        response = _rag_engine.openai_llm.invoke([HumanMessage(content=prompt)])
        keywords_text = response.content.strip()

        # カンマまたはスペースで分割
        keywords = []
        for k in keywords_text.replace('、', ',').split(','):
            k = k.strip()
            if k and len(k) >= 2:  # 1文字キーワードは除外
                keywords.append(k)

        logger.info(f"🤖 LLM keyword extraction: '{query}' -> {keywords}")
        return keywords if keywords else tokenize_query(query)

    except Exception as e:
        logger.error(f"❌ LLM keyword extraction failed: {e}")
        # フォールバック: 既存のトークン化
        fallback_keywords = tokenize_query(query)
        logger.info(f"Fallback to tokenization: {fallback_keywords}")
        return fallback_keywords


def find_text_positions(
    pdf_path: Path,
    page_number: int,
    search_terms: List[str],
    vision_analyzer=None,
    dpi: int = DEFAULT_DPI
) -> List[Dict[str, float]]:
    """
    PDFページ内で指定されたテキストの座標を検出

    Args:
        pdf_path: PDFファイルパス
        page_number: ページ番号（1始まり）
        search_terms: 検索するテキストのリスト
        vision_analyzer: VisionAnalyzerインスタンス（OCRフォールバック用、省略可）
        dpi: OCR用画像の解像度（デフォルト: 150）

    Returns:
        list: 座標情報のリスト
            [{
                "text": str,  # マッチしたテキスト
                "x0": float,  # 左端X座標
                "y0": float,  # 上端Y座標
                "x1": float,  # 右端X座標
                "y1": float,  # 下端Y座標
            }]
    """
    positions = []

    if not search_terms:
        return positions

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # ページ番号は1始まりだが、pdfplumberは0始まり
            page = pdf.pages[page_number - 1]

            # ページ内の全テキストを単語単位で取得
            words = page.extract_words()

            # OCRフォールバック: pdfplumberで抽出できない場合
            if len(words) == 0 and vision_analyzer and PDF2IMAGE_AVAILABLE:
                logger.warning(f"⚠️ PDF page {page_number} has no extractable text - attempting OCR")

                try:
                    # PDFページを画像化
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=dpi,
                        first_page=page_number,
                        last_page=page_number,
                        fmt='png'
                    )

                    if images and len(images) > 0:
                        # 一時ファイルとして保存
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                            images[0].save(tmp_path, 'PNG')

                        try:
                            # Vision API OCR実行
                            logger.info(f"🔍 Running OCR on page {page_number} using Vision API...")
                            ocr_result = vision_analyzer.ocr_page(tmp_path)
                            words = ocr_result.get("words", [])
                            logger.info(f"✅ OCR extracted {len(words)} words from page {page_number}")
                        finally:
                            # 一時ファイル削除
                            import os
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    else:
                        logger.error(f"❌ Failed to convert PDF page {page_number} to image for OCR")
                except Exception as ocr_error:
                    logger.error(f"❌ OCR fallback failed: {ocr_error}", exc_info=True)
                    # OCR失敗時は空のwordsのまま続行

            # 各検索語に対してマッチングを実行
            for search_term in search_terms:
                search_term_lower = search_term.lower()

                for word in words:
                    word_text = word['text'].lower()

                    # 部分一致でマッチング
                    if search_term_lower in word_text or word_text in search_term_lower:
                        positions.append({
                            "text": word['text'],
                            "x0": word['x0'],
                            "y0": word['top'],
                            "x1": word['x1'],
                            "y1": word['bottom'],
                        })

            logger.info(f"📍 Found {len(positions)} text positions for {len(search_terms)} search terms on page {page_number}")
            if len(positions) == 0 and len(search_terms) > 0:
                logger.warning(f"⚠️ No text positions found for search terms: {search_terms}")
            return positions

    except Exception as e:
        logger.error(f"❌ Error finding text positions: {e}", exc_info=True)
        return []


def highlight_text_on_image(
    image: Image.Image,
    text_positions: List[Dict[str, float]],
    page_height: float,
    dpi: int = DEFAULT_DPI,
    highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 80)  # 黄色半透明
) -> Image.Image:
    """
    画像上にテキストハイライトを描画

    Args:
        image: 元の画像
        text_positions: テキスト座標のリスト
        page_height: PDFページの高さ（ポイント単位）
        dpi: 画像のDPI
        highlight_color: ハイライトカラー (R, G, B, Alpha)

    Returns:
        PIL.Image: ハイライト付き画像
    """
    if not text_positions:
        return image

    try:
        # RGBAモードに変換（透過処理のため）
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # 透明なオーバーレイを作成
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # PDF座標から画像座標へのスケール係数
        # PDFは72 DPI、画像はDPI指定の解像度
        scale = dpi / 72.0

        # ページ高さのスケール調整
        img_height = image.height

        # 各テキスト位置にハイライト矩形を描画
        for pos in text_positions:
            # PDF座標（原点は左下）から画像座標（原点は左上）に変換
            x0 = pos['x0'] * scale
            y0 = (page_height - pos['y1']) * scale  # Y座標は反転
            x1 = pos['x1'] * scale
            y1 = (page_height - pos['y0']) * scale

            # パディングを追加（読みやすくするため）
            padding = 2
            draw.rectangle(
                [(x0 - padding, y0 - padding), (x1 + padding, y1 + padding)],
                fill=highlight_color
            )

        # オーバーレイを元画像に合成
        highlighted = Image.alpha_composite(image, overlay)

        # RGBモードに戻す（Streamlitでの表示のため）
        highlighted = highlighted.convert('RGB')

        logger.info(f"Applied {len(text_positions)} highlights to image")
        return highlighted

    except Exception as e:
        logger.error(f"Error highlighting text on image: {e}", exc_info=True)
        return image


@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def extract_page_with_highlight(
    source_file: str,
    page_number: int,
    query: str,
    _vector_store,
    _rag_engine=None,
    _vision_analyzer=None,
    use_llm_keywords: bool = True,
    cache_version: int = 4,  # v4: OCR fallback implementation (NO underscore to include in cache key!)
    dpi: int = DEFAULT_DPI,
    target_width: int = DEFAULT_WIDTH
) -> Optional[Image.Image]:
    """
    PDFページを画像に変換し、検索クエリに一致するテキストをハイライト

    Args:
        source_file: PDFファイル名
        page_number: ページ番号（1始まり）
        query: 検索クエリ（ハイライト対象）
        _vector_store: VectorStoreインスタンス
        _rag_engine: RAGEngineインスタンス（LLMキーワード抽出に使用、省略可）
        _vision_analyzer: VisionAnalyzerインスタンス（OCRフォールバック用、省略可）
        use_llm_keywords: LLMを使用したキーワード抽出を有効化（デフォルト: True）
        cache_version: キャッシュバージョン（変更時にインクリメント、通常変更不要）
        dpi: 解像度
        target_width: 画像幅

    Returns:
        PIL.Image: ハイライト付き画像、失敗時はNone
    """
    # 🔍 実行確認ログ（デバッグ用）
    logger.info(f"📸 extract_page_with_highlight() CALLED - cache_v{cache_version}, pdf2image={PDF2IMAGE_AVAILABLE}")
    logger.info(f"   → source={source_file}, page={page_number}, query_len={len(query) if query else 0}")

    if not PDF2IMAGE_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("❌ PDF page rendering is DISABLED (poppler not installed)")
        logger.warning(f"❌ Cannot render page {page_number} of {source_file}")
        logger.warning("💡 Check Streamlit Cloud logs for poppler installation errors")
        logger.warning("=" * 60)
        return None

    try:
        # PDFのローカルパスを取得
        pdf_path = get_pdf_path(source_file, _vector_store)
        if not pdf_path:
            logger.error(f"Failed to get PDF path: {source_file}")
            return None

        # クエリからキーワードを抽出（LLM or トークン化）
        logger.info(f"🔍 Highlighting query: '{query}' for {source_file} page {page_number}")

        if query:
            # LLMベースのキーワード抽出（推奨）
            if use_llm_keywords and _rag_engine is not None:
                try:
                    search_terms = extract_keywords_llm(query, _rag_engine)
                    logger.info(f"🤖 LLM-extracted keywords: {search_terms}")
                except Exception as e:
                    logger.warning(f"⚠️ LLM keyword extraction failed: {e}, falling back to tokenization")
                    search_terms = tokenize_query(query)
                    logger.info(f"🔤 Fallback tokenized keywords: {search_terms}")
            else:
                # トークン化（フォールバック）
                search_terms = tokenize_query(query)
                logger.info(f"🔤 Tokenized keywords: {search_terms}")
        else:
            search_terms = []

        # テキスト位置を検出
        text_positions = []
        page_height = 0

        if search_terms:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                page_height = page.height
                text_positions = find_text_positions(pdf_path, page_number, search_terms, _vision_analyzer, dpi)
                logger.info(f"📊 Text positions found: {len(text_positions)}")
        else:
            logger.warning("⚠️ No search terms to highlight")

        # ページを画像に変換
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            fmt='png'
        )

        if not images or len(images) == 0:
            logger.error(f"No image generated for page {page_number} of {source_file}")
            return None

        image = images[0]

        # ハイライトを適用
        if text_positions and page_height > 0:
            image = highlight_text_on_image(image, text_positions, page_height, dpi)
            logger.info(f"✅ Applied {len(text_positions)} highlights to page {page_number} of {source_file}")
        elif search_terms:
            logger.warning(f"⚠️ No highlights applied (text_positions={len(text_positions)}, page_height={page_height})")

        # 画像サイズを調整
        if image.width > target_width:
            aspect_ratio = image.height / image.width
            new_height = int(target_width * aspect_ratio)
            image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)

        return image

    except Exception as e:
        logger.error(f"Error extracting page with highlight: {e}", exc_info=True)
        return None
