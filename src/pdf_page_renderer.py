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

# PyMuPDF (fitz) のインポート - ハイライト座標の高速検索用
PYMUPDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("✅ PyMuPDF is available - Fast PDF text search enabled")
except ImportError:
    logger.warning("❌ PyMuPDF not available - Using pdfplumber fallback")

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
    Office→PDF変換されたファイルにも対応

    Args:
        source_file: PDFファイル名（またはOfficeファイル名）
        vector_store: VectorStoreインスタンス（Supabase Storage連携用）

    Returns:
        Path: PDFのローカルパス、取得失敗時はNone
    """
    # まずローカルストレージをチェック
    local_pdf_path = Path("data/uploaded_pdfs") / source_file

    if local_pdf_path.exists():
        logger.info(f"Using local PDF: {local_pdf_path}")
        return local_pdf_path

    # Office→PDF変換済みファイルをチェック
    # source_fileが.docx, .xlsx, .pptx等の場合、.pdfに変換して検索
    source_path = Path(source_file)
    office_extensions = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']

    if source_path.suffix.lower() in office_extensions:
        # 拡張子を.pdfに変更
        pdf_filename = source_path.stem + ".pdf"

        # 変換済みPDFディレクトリをチェック
        converted_pdf_path = Path("data/converted_pdfs") / pdf_filename
        if converted_pdf_path.exists():
            logger.info(f"Using converted PDF: {converted_pdf_path}")
            return converted_pdf_path

        # static/pdfsディレクトリもチェック
        static_pdf_path = Path("static/pdfs") / pdf_filename
        if static_pdf_path.exists():
            logger.info(f"Using static converted PDF: {static_pdf_path}")
            return static_pdf_path

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


def create_pdf_annotations_pymupdf(
    pdf_path: Path,
    search_terms: List[str],
    page_numbers: Optional[List[int]] = None
) -> List[Dict]:
    """
    PyMuPDFを使用してPDF内のキーワードを検索し、streamlit-pdf-viewer用のアノテーションを生成

    Args:
        pdf_path: PDFファイルのパス
        search_terms: 検索キーワードのリスト
        page_numbers: 検索対象のページ番号リスト（1始まり）。Noneの場合は全ページ

    Returns:
        List[Dict]: streamlit-pdf-viewer用のアノテーション形式
            [
                {
                    "page": 1,
                    "x": 220,
                    "y": 155,
                    "width": 65,
                    "height": 22,
                    "color": "yellow",
                    "border": "solid"
                },
                ...
            ]
    """
    import unicodedata

    logger.info(f"🔍 create_pdf_annotations_pymupdf() called")
    logger.info(f"   pdf_path={pdf_path}")
    logger.info(f"   search_terms={search_terms}")
    logger.info(f"   page_numbers={page_numbers}")
    logger.info(f"   PYMUPDF_AVAILABLE={PYMUPDF_AVAILABLE}")

    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available - cannot create annotations")
        return []

    annotations = []

    try:
        doc = fitz.open(pdf_path)
        logger.info(f"✅ PDF opened successfully: {len(doc)} pages")

        # 検索対象ページの決定
        if page_numbers is None:
            page_numbers = list(range(1, len(doc) + 1))

        for page_num in page_numbers:
            try:
                # ページ番号は1始まりだが、PyMuPDFは0始まり
                page = doc[page_num - 1]
                page_height = page.rect.height

                # ページテキストを取得（検証用）
                page_text = page.get_text()
                logger.info(f"📄 Processing page {page_num}: size={page.rect}, height={page_height}")
                logger.info(f"   Page text length: {len(page_text)} chars")
                if len(page_text) > 0:
                    logger.debug(f"   First 100 chars: {page_text[:100]}")
                else:
                    logger.warning(f"   ⚠️ Page {page_num} has NO extractable text (might be scanned)")

                for term in search_terms:
                    # キーワード長フィルタ（2文字以上のみ）
                    if len(term) < 2:
                        logger.debug(f"   Skipping term '{term}' (too short)")
                        continue

                    logger.info(f"   🔍 Searching for: '{term}' (len={len(term)})")

                    # Unicode正規化（NFC形式）
                    term_normalized = unicodedata.normalize('NFC', term)
                    if term_normalized != term:
                        logger.info(f"      Unicode normalized: '{term}' → '{term_normalized}'")

                    # テキスト検索（矩形リストを取得）
                    rects = page.search_for(term_normalized)
                    logger.info(f"      → Found {len(rects)} matches for '{term_normalized}' on page {page_num}")

                    # NFC正規化で見つからない場合、NFD形式も試す
                    if len(rects) == 0:
                        term_nfd = unicodedata.normalize('NFD', term)
                        if term_nfd != term_normalized:
                            logger.info(f"      Trying NFD normalization: '{term_nfd}'")
                            rects = page.search_for(term_nfd)
                            logger.info(f"      → NFD search found {len(rects)} matches")

                    # それでも見つからない場合、ページテキスト内に存在するか確認
                    if len(rects) == 0 and len(page_text) > 0:
                        if term in page_text or term_normalized in page_text:
                            logger.warning(f"      ⚠️ Term '{term}' exists in page text but search_for() returned 0 results!")
                            logger.warning(f"         This might be an encoding issue")
                        else:
                            logger.debug(f"      ℹ️ Term '{term}' not found in page text")

                    for rect in rects:
                        # PyMuPDF座標（左下原点）→ streamlit-pdf-viewer座標（左上原点）
                        annotations.append({
                            "page": page_num,
                            "x": float(rect.x0),
                            "y": float(page_height - rect.y1),  # Y座標を反転
                            "width": float(rect.x1 - rect.x0),
                            "height": float(rect.y1 - rect.y0),
                            "color": "yellow",
                            "border": "solid"
                        })

                page_annotations = len([a for a in annotations if a['page'] == page_num])
                logger.info(f"   📍 Created {page_annotations} annotations for page {page_num}")

            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}", exc_info=True)
                continue

        doc.close()
        logger.info(f"📊 Summary: Created {len(annotations)} annotations for {len(search_terms)} search terms")
        if len(annotations) == 0:
            logger.warning(f"   ⚠️ NO ANNOTATIONS CREATED despite {len(search_terms)} search terms!")
        return annotations

    except Exception as e:
        logger.error(f"Error creating annotations: {e}", exc_info=True)
        return []


def split_text_into_sentences(text: str) -> List[Dict]:
    """
    テキストを文単位に分割し、各文の開始・終了位置を記録

    Args:
        text: 分割対象のテキスト

    Returns:
        List[Dict]: 文のリスト
            [
                {
                    "text": "文の内容",
                    "start": 0,  # 文字オフセット（開始位置）
                    "end": 10    # 文字オフセット（終了位置）
                },
                ...
            ]
    """
    import re

    if not text or not text.strip():
        return []

    sentences = []

    # 日本語・英語対応の文区切りパターン
    # 。．.!！?？で区切る
    pattern = r'([^。．.!！?？]+[。\.!！?？]+)'

    matches = re.finditer(pattern, text)

    for match in matches:
        sentence_text = match.group(0).strip()
        if len(sentence_text) > 0:
            sentences.append({
                "text": sentence_text,
                "start": match.start(),
                "end": match.end()
            })

    # パターンにマッチしない残りのテキスト（最後の文など）
    if sentences:
        last_end = sentences[-1]["end"]
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if len(remaining) > 0:
                sentences.append({
                    "text": remaining,
                    "start": last_end,
                    "end": len(text)
                })
    elif len(text.strip()) > 0:
        # パターンにマッチしない場合、全体を1文として扱う
        sentences.append({
            "text": text.strip(),
            "start": 0,
            "end": len(text)
        })

    logger.debug(f"Split text into {len(sentences)} sentences")
    return sentences


def filter_sentences_by_embedding(
    sentences: List[Dict],
    query: str,
    rag_engine,
    threshold: float = 0.7,
    max_candidates: int = 10
) -> List[Dict]:
    """
    エンベディングの類似度で文を絞り込み

    Args:
        sentences: 候補文のリスト（split_text_into_sentences()の出力）
        query: ユーザークエリ
        rag_engine: RAGEngineインスタンス（エンベディング計算用）
        threshold: 類似度閾値（0-1）
        max_candidates: 最大候補数

    Returns:
        List[Dict]: 類似度の高い文のリスト（類似度でソート済み）
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if not sentences:
        return []

    try:
        # クエリのエンベディングを取得
        query_embedding = rag_engine.embedding_model.embed_query(query)

        # 各文のエンベディングを計算
        sentence_embeddings = []
        for sent in sentences:
            sent_embedding = rag_engine.embedding_model.embed_query(sent["text"])
            sentence_embeddings.append(sent_embedding)

        # コサイン類似度を計算
        similarities = cosine_similarity(
            [query_embedding],
            sentence_embeddings
        )[0]

        # 各文に類似度を追加
        for i, sent in enumerate(sentences):
            sent["similarity"] = float(similarities[i])

        # 類似度でフィルタリングとソート
        filtered = [s for s in sentences if s["similarity"] >= threshold]
        filtered.sort(key=lambda x: x["similarity"], reverse=True)

        # 上位max_candidates件を返す
        result = filtered[:max_candidates]

        logger.info(f"🔍 Embedding filter: {len(sentences)} sentences → {len(result)} candidates (threshold={threshold})")
        for i, sent in enumerate(result[:3]):  # 上位3件をログ出力
            logger.debug(f"   {i+1}. similarity={sent['similarity']:.3f}: {sent['text'][:50]}...")

        return result

    except Exception as e:
        logger.error(f"Error in embedding filter: {e}", exc_info=True)
        return sentences[:max_candidates]  # エラー時は先頭から返す


def refine_with_llm(
    candidate_sentences: List[Dict],
    query: str,
    rag_engine,
    max_sentences: int = 5
) -> List[Dict]:
    """
    LLMで候補文を精査し、最も関連性の高い文を選択

    Args:
        candidate_sentences: 候補文のリスト（filter_sentences_by_embedding()の出力）
        query: ユーザークエリ
        rag_engine: RAGEngineインスタンス（LLM呼び出し用）
        max_sentences: 最終選択する最大文数

    Returns:
        List[Dict]: LLMが選択した関連文のリスト
    """
    if not candidate_sentences:
        return []

    try:
        # 候補文に番号を付ける
        numbered_candidates = []
        for i, sent in enumerate(candidate_sentences):
            numbered_candidates.append(f"{i+1}. {sent['text']}")

        candidates_text = "\n".join(numbered_candidates)

        # LLMプロンプト
        prompt = f"""以下のユーザークエリに最も関連する文を、候補から最大{max_sentences}個選んでください。

【ユーザークエリ】
{query}

【候補文】
{candidates_text}

【指示】
- 上記の候補から、クエリに直接関連する文の番号のみを選んでください
- 番号はカンマ区切りで出力してください（例: 1,3,5）
- 関連する文がない場合は「なし」と出力してください
- 番号以外の説明は不要です

【出力】
"""

        # LLMを呼び出し
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        llm_response = response.choices[0].message.content.strip()
        logger.info(f"🤖 LLM refinement response: '{llm_response}'")

        # 応答から番号を抽出
        if "なし" in llm_response or "None" in llm_response:
            logger.info(f"   LLM found no relevant sentences")
            return []

        # 番号を解析（例: "1,3,5" → [1, 3, 5]）
        import re
        numbers = re.findall(r'\d+', llm_response)
        selected_indices = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < len(candidate_sentences)]

        selected_sentences = [candidate_sentences[i] for i in selected_indices]

        logger.info(f"   Selected {len(selected_sentences)} sentences from {len(candidate_sentences)} candidates")
        for sent in selected_sentences:
            logger.debug(f"      - {sent['text'][:50]}...")

        return selected_sentences

    except Exception as e:
        logger.error(f"Error in LLM refinement: {e}", exc_info=True)
        # エラー時は類似度上位を返す
        return candidate_sentences[:max_sentences]


def create_pdf_annotations_hybrid(
    pdf_path: Path,
    query: str,
    page_numbers: List[int],
    rag_engine,
    config: dict
) -> List[Dict]:
    """
    ハイブリッドアプローチでPDFアノテーションを生成

    Stage 1: エンベディングで候補文を絞り込み（高速）
    Stage 2: LLMで関連文を精査（高精度）
    Stage 3: 座標を取得してアノテーション生成

    Args:
        pdf_path: PDFファイルのパス
        query: ユーザークエリ
        page_numbers: 検索対象ページ番号リスト（1始まり）
        rag_engine: RAGEngineインスタンス
        config: 設定辞書

    Returns:
        List[Dict]: streamlit-pdf-viewer用のアノテーション形式
    """
    import pdfplumber

    logger.info(f"🎯 create_pdf_annotations_hybrid() called")
    logger.info(f"   pdf_path={pdf_path}")
    logger.info(f"   query={query}")
    logger.info(f"   page_numbers={page_numbers}")

    # 設定を取得
    hybrid_config = config.get("pdf_highlighting", {}).get("hybrid", {})
    embedding_threshold = hybrid_config.get("embedding_threshold", 0.7)
    max_candidates = hybrid_config.get("max_candidates", 10)
    max_final = hybrid_config.get("max_final", 5)
    use_llm_refinement = hybrid_config.get("use_llm_refinement", True)
    fallback_to_keyword = hybrid_config.get("fallback_to_keyword", True)

    annotations = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in page_numbers:
                try:
                    page = pdf.pages[page_num - 1]
                    page_text = page.extract_text()

                    if not page_text:
                        logger.warning(f"   Page {page_num} has no extractable text")
                        continue

                    logger.info(f"📄 Processing page {page_num} ({len(page_text)} chars)")

                    # Stage 1: 文分割
                    sentences = split_text_into_sentences(page_text)
                    logger.info(f"   Stage 1: Split into {len(sentences)} sentences")

                    if not sentences:
                        continue

                    # Stage 2: エンベディングでフィルタリング
                    candidates = filter_sentences_by_embedding(
                        sentences,
                        query,
                        rag_engine,
                        threshold=embedding_threshold,
                        max_candidates=max_candidates
                    )

                    if not candidates:
                        logger.info(f"   Stage 2: No candidates above threshold={embedding_threshold}")
                        continue

                    # Stage 3: LLMで精査（オプション）
                    if use_llm_refinement and len(candidates) > 0:
                        selected_sentences = refine_with_llm(
                            candidates,
                            query,
                            rag_engine,
                            max_sentences=max_final
                        )
                    else:
                        selected_sentences = candidates[:max_final]

                    if not selected_sentences:
                        logger.info(f"   Stage 3: No sentences selected by LLM")
                        continue

                    # Stage 4: 座標を取得してアノテーション生成
                    page_height = page.height
                    for sent in selected_sentences:
                        # 文のテキストから座標を検索
                        words = page.extract_words()
                        positions = find_text_positions_in_words(
                            sent["text"],
                            words,
                            page_num
                        )

                        # アノテーションに変換
                        for pos in positions:
                            annotations.append({
                                "page": page_num,
                                "x": float(pos["x0"]),
                                "y": float(pos["y0"]),
                                "width": float(pos["x1"] - pos["x0"]),
                                "height": float(pos["y1"] - pos["y0"]),
                                "color": "yellow",
                                "border": "solid"
                            })

                    logger.info(f"   📍 Created {len(annotations)} annotations for page {page_num}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
                    continue

        logger.info(f"📊 Hybrid annotation summary: {len(annotations)} annotations created")
        return annotations

    except Exception as e:
        logger.error(f"Error in hybrid annotation generation: {e}", exc_info=True)

        # フォールバック: キーワード方式
        if fallback_to_keyword:
            logger.warning(f"   Falling back to keyword-based highlighting")
            from src.pdf_page_renderer import extract_keywords_llm
            keywords = extract_keywords_llm(query, rag_engine)
            return create_pdf_annotations_pymupdf(pdf_path, keywords, page_numbers)
        else:
            return []


def find_text_positions_in_words(
    search_text: str,
    words: List[Dict],
    page_number: int
) -> List[Dict]:
    """
    単語リストから検索テキストの座標を取得

    Args:
        search_text: 検索するテキスト
        words: pdfplumberのextract_words()の出力
        page_number: ページ番号

    Returns:
        List[Dict]: 座標のリスト
    """
    positions = []
    search_text_lower = search_text.lower()

    # 単語をテキスト順に結合して検索
    for i, word in enumerate(words):
        word_text = word['text'].lower()

        # 部分一致で検索
        if search_text_lower in word_text or word_text in search_text_lower:
            positions.append({
                "text": word['text'],
                "x0": word['x0'],
                "y0": word['top'],
                "x1": word['x1'],
                "y1": word['bottom'],
            })

    return positions


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

            # スキャンPDF（テキスト抽出不可）の場合は空リストを返す
            if len(words) == 0:
                logger.warning(f"⚠️ PDF page {page_number} has no extractable text (scanned PDF) - no highlights will be shown")
                return []

            # 各検索語に対してマッチングを実行（改善版：単方向部分一致 + 長さフィルタ）
            for search_term in search_terms:
                search_term_lower = search_term.lower()

                # キーワード長フィルタ（2文字以上のみマッチング）
                if len(search_term_lower) < 2:
                    continue

                for word in words:
                    word_text = word['text'].lower()

                    # 単方向部分一致（キーワードが単語に含まれる場合のみ）
                    if search_term_lower in word_text:
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
