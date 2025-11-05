"""
Streamlit Application for PDF RAG System
"""

import streamlit as st
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# src モジュールのインポート
from src.utils import load_config, load_environment, ensure_directories, setup_logging, encode_pdf_to_base64
from src.document_processor import DocumentProcessor
from src.text_embedder import TextEmbedder
from src.vision_analyzer import VisionAnalyzer
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.pdf_manager import PDFManager
from src.pdf_page_renderer import extract_page_as_image, extract_multiple_pages, extract_page_with_highlight, PDF2IMAGE_AVAILABLE, get_pdf_path, create_pdf_annotations_pymupdf

# streamlit-pdf-viewer のインポート
try:
    from streamlit_pdf_viewer import pdf_viewer
    STREAMLIT_PDF_VIEWER_AVAILABLE = True
    logger.info("✅ streamlit-pdf-viewer is available")
except ImportError:
    STREAMLIT_PDF_VIEWER_AVAILABLE = False
    logger.warning("❌ streamlit-pdf-viewer not available - using fallback image display")

# ロガー設定
logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)


def initialize_app():
    """アプリケーションの初期化"""
    # 環境変数読み込み
    load_environment()

    # ディレクトリ作成
    ensure_directories()

    # 設定読み込み
    config = load_config()

    # ログ設定
    logger = setup_logging(config)

    # セッション状態の初期化
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = config
        st.session_state.document_processor = DocumentProcessor(config)
        st.session_state.embedder = TextEmbedder(config)
        st.session_state.vision_analyzer = VisionAnalyzer(config)
        st.session_state.vector_store = VectorStore(config)
        st.session_state.rag_engine = RAGEngine(
            config,
            st.session_state.vector_store,
            st.session_state.embedder
        )
        st.session_state.pdf_manager = PDFManager(
            st.session_state.vector_store,
            config
        )
        st.session_state.chat_history = []
        st.session_state.selected_category = "全カテゴリー"
        st.session_state.selected_model = "openai"

        # Vision Analyzerの状態チェック
        if not st.session_state.vision_analyzer.api_key_valid:
            st.session_state.vision_disabled = True
            logger.warning("Vision analysis is disabled due to missing or invalid OPENAI_API_KEY")
        else:
            st.session_state.vision_disabled = False

    return config


def sidebar():
    """サイドバーのUI"""
    st.sidebar.title("📁 ドキュメント管理")

    # Vision Analyzer警告表示
    if st.session_state.get('vision_disabled', False):
        st.sidebar.warning(
            "⚠️ 画像解析機能が無効です\n\n"
            "OPENAI_API_KEYが設定されていません。\n"
            "画像やグラフの解析を有効にするには、.envファイルにOPENAI_API_KEYを設定してください。"
        )

    # PDF Page Preview警告表示
    if not PDF2IMAGE_AVAILABLE:
        st.sidebar.info(
            "ℹ️ PDFページプレビュー機能が無効です\n\n"
            "**ローカル環境（Windows）:**\n"
            "Popplerをインストールしてください。\n"
            "`choco install poppler` または\n"
            "[手動ダウンロード](https://github.com/oschwartz10612/poppler-windows/releases)\n\n"
            "**Streamlit Cloud:**\n"
            "自動的に有効になります（packages.txt対応済み）"
        )

    # ドキュメントアップロード
    st.sidebar.subheader("📄 ドキュメントアップロード")
    uploaded_files = st.sidebar.file_uploader(
        "ファイルを選択 (PDF, Word, Excel, PowerPoint, Text)",
        type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'txt'],
        accept_multiple_files=True,
        help="対応形式: PDF, Word, Excel, PowerPoint, Text"
    )

    # カテゴリー入力
    category = st.sidebar.text_input(
        "カテゴリー名",
        placeholder="例: 製品マニュアル"
    )

    # インデックス作成ボタン
    if st.sidebar.button("📑 インデックス作成", type="primary"):
        if uploaded_files and category:
            process_documents(uploaded_files, category)
        else:
            st.sidebar.error("ドキュメントファイルとカテゴリー名を入力してください")

    # 登録済みカテゴリー表示
    st.sidebar.subheader("📂 登録済みカテゴリー")
    try:
        categories = st.session_state.vector_store.get_all_categories()
        if categories:
            for cat in categories:
                st.sidebar.text(f"• {cat}")
        else:
            st.sidebar.info("まだカテゴリーが登録されていません")
    except Exception as e:
        st.sidebar.error(f"カテゴリー取得エラー: {str(e)}")
        categories = []

    # 登録済みPDF管理
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 登録済みPDF管理")

    registered_pdfs = st.session_state.pdf_manager.get_registered_pdfs()
    if registered_pdfs:
        for pdf in registered_pdfs:
            with st.sidebar.expander(f"📄 {pdf['source_file']}", expanded=False):
                st.write(f"**カテゴリー**: {pdf['category']}")
                st.write(f"**テキストデータ**: {pdf['text_count']} 件")
                st.write(f"**画像データ**: {pdf['image_count']} 件")
                st.write(f"**合計**: {pdf['total_count']} 件")

                # ボタンを2列に配置
                col1, col2 = st.columns(2)

                with col1:
                    # 閲覧ボタン
                    pdf_path = Path("data/uploaded_pdfs") / pdf['source_file']
                    # Supabase Storageを使用している場合はローカルファイルチェックをスキップ
                    if st.session_state.vector_store.provider == 'supabase' or pdf_path.exists():
                        show_pdf_link(pdf_path, pdf['source_file'], key_suffix="sidebar")
                    else:
                        st.error(f"PDFファイルが見つかりません: {pdf['source_file']}")

                with col2:
                    # 削除ボタン（アイコンのみ）
                    delete_key = f"delete_{pdf['source_file']}"
                    if st.button("🗑️", key=delete_key, type="secondary", use_container_width=True, help="PDFを削除する"):
                        # 削除確認用のセッション状態を設定
                        st.session_state.delete_target = pdf['source_file']
                        st.session_state.show_delete_confirm = True
                        st.rerun()
    else:
        st.sidebar.info("登録済みPDFがありません")

    # チャット設定
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 チャット設定")

    # カテゴリー選択
    try:
        categories = ["全カテゴリー"] + st.session_state.vector_store.get_all_categories()
    except Exception as e:
        st.sidebar.warning(f"カテゴリー取得エラー: {str(e)}")
        categories = ["全カテゴリー"]

    st.session_state.selected_category = st.sidebar.selectbox(
        "🔍 検索対象カテゴリー",
        categories,
        index=categories.index(st.session_state.selected_category) if st.session_state.selected_category in categories else 0,
        help="質問する対象のカテゴリーを選択してください"
    )

    # AIモデル選択
    model_options = {
        "GPT-4.1": "openai",
        "Gemini-2.5-Pro": "gemini"
    }
    current_model_display = [k for k, v in model_options.items() if v == st.session_state.selected_model][0]
    selected_model_display = st.sidebar.selectbox(
        "🤖 AIモデル",
        list(model_options.keys()),
        index=list(model_options.keys()).index(current_model_display),
        help="使用するAIモデルを選択"
    )
    st.session_state.selected_model = model_options[selected_model_display]

    # チャットリセットボタン
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ チャット履歴をリセット", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


def process_documents(uploaded_files, category):
    """ドキュメントファイル（PDF、Word、Excel）を処理"""
    # カテゴリーはSupabaseのregistered_pdfsテーブルに自動保存されるため、
    # ローカルファイルへの保存は不要

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # ファイルサイズの上限を取得
    max_size_mb = st.session_state.config.get('pdf_upload', {}).get('max_file_size_mb', 50)

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # ファイルサイズチェック
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                st.sidebar.error(f"{uploaded_file.name}: ファイルサイズが上限（{max_size_mb}MB）を超えています（{file_size_mb:.1f}MB）")
                continue

            # 1. ドキュメントを保存（data/uploaded_pdfs/ と static/pdfs/ の両方）
            # 注: ディレクトリ名はPDF時代の名残だが、全ドキュメント形式で使用
            status_text.text(f"処理中: {uploaded_file.name} (1/?) - ファイル保存中...")
            doc_path = Path("data/uploaded_pdfs") / uploaded_file.name
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            static_doc_path = Path("static/pdfs") / uploaded_file.name
            static_doc_path.parent.mkdir(parents=True, exist_ok=True)

            doc_bytes = uploaded_file.getbuffer()
            with open(doc_path, "wb") as f:
                f.write(doc_bytes)
            with open(static_doc_path, "wb") as f:
                f.write(doc_bytes)

            # 2. テキスト・画像抽出
            status_text.text(f"処理中: {uploaded_file.name} (2/?) - テキスト・画像抽出中...")
            try:
                logging.info(f"Starting document processing for {uploaded_file.name}")
                doc_result = st.session_state.document_processor.process_document(str(doc_path), category)
                logging.info(f"Document processing completed for {uploaded_file.name}: {len(doc_result.get('text_chunks', []))} text chunks, {len(doc_result.get('images', []))} images")
            except Exception as e:
                error_msg = f"ドキュメント処理中にエラーが発生しました: {str(e)}"
                logging.error(error_msg, exc_info=True)
                st.sidebar.error(error_msg)
                continue

            # 総ステップ数を決定（画像があれば5、なければ4）
            total_steps = 5 if doc_result['images'] else 4
            num_pages = doc_result.get('total_pages', '?')
            num_chunks = len(doc_result['text_chunks'])
            num_images = len(doc_result['images'])

            # 3. テキストチャンクをエンベディング（バッチ処理）
            status_text.text(f"処理中: {uploaded_file.name} (3/{total_steps}) - テキストエンベディング中（{num_chunks}チャンク）...")
            if doc_result['text_chunks']:
                # 全テキストをまとめてバッチ処理
                texts = [chunk['text'] for chunk in doc_result['text_chunks']]
                text_embeddings = st.session_state.embedder.embed_batch(texts)

                # ベクトルストアに追加
                st.session_state.vector_store.add_text_chunks(
                    doc_result['text_chunks'],
                    text_embeddings
                )

            # 4. 画像をVision AIで解析（並列処理）- 画像がある場合のみ
            analyzed_images = []
            failed_images = []

            if doc_result['images']:
                status_text.text(f"処理中: {uploaded_file.name} (4/{total_steps}) - 画像解析中（{num_images}枚）...")
                max_workers = st.session_state.config.get('performance', {}).get('max_workers', 4)

                # VisionAnalyzerインスタンスをローカル変数に保存（スレッドセーフ）
                vision_analyzer = st.session_state.vision_analyzer

                # 画像解析を並列処理
                def analyze_single_image(image_data, analyzer):
                    try:
                        actual_content_type = image_data.get('content_type', 'image')
                        image_path = image_data['image_path']
                        logging.info(f"Starting analysis for {actual_content_type}: {image_path}")

                        analysis = analyzer.analyze_image(
                            image_path,
                            content_type=actual_content_type
                        )

                        # メタデータを統合
                        image_data.update({
                            'category': category,
                            'content_type': analysis.get('content_type', 'image'),
                            'description': analysis['description']
                        })

                        logging.info(f"Successfully analyzed {actual_content_type}: {image_path}")
                        return {'success': True, 'data': image_data}

                    except Exception as e:
                        error_msg = f"画像解析失敗 ({image_data.get('image_path', 'unknown')}): {type(e).__name__}: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        return {
                            'success': False,
                            'data': image_data,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }

                # ThreadPoolExecutorで並列処理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(analyze_single_image, img, vision_analyzer): img for img in doc_result['images']}
                    for future in as_completed(futures):
                        result = future.result()
                        if result['success']:
                            analyzed_images.append(result['data'])
                        else:
                            failed_images.append(result)

                # 解析結果の集計
                success_count = len(analyzed_images)
                failed_count = len(failed_images)

                logging.info(f"Image analysis complete: {success_count} succeeded, {failed_count} failed")

                # 失敗した画像がある場合、警告を表示
                if failed_images:
                    error_types = {}
                    for failure in failed_images:
                        error_type = failure.get('error_type', 'Unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1

                    error_summary = ", ".join([f"{err_type}: {count}件" for err_type, count in error_types.items()])
                    warning_msg = f"⚠️ 画像解析エラー: {failed_count}/{num_images}枚失敗 ({error_summary})"
                    st.sidebar.warning(warning_msg)
                    logging.warning(warning_msg)

                    # 最初のエラーの詳細をログに出力
                    if failed_images:
                        first_error = failed_images[0]
                        logging.error(f"First error details: {first_error.get('error')}")

                # 解析結果をバッチでエンベディング
                if analyzed_images:
                    descriptions = [img['description'] for img in analyzed_images]
                    image_embeddings = st.session_state.embedder.embed_batch(descriptions)

                    # ベクトルストアにバッチで追加
                    st.session_state.vector_store.add_image_contents_batch(analyzed_images, image_embeddings)
                    logging.info(f"Added {len(analyzed_images)} images to vector store")

                    # Vision APIで抽出したテキストをtext_chunksとしても保存（検索精度向上）
                    text_chunks_from_vision = []
                    for img in analyzed_images:
                        text_chunks_from_vision.append({
                            'text': img['description'],  # 'content'ではなく'text'を使用
                            'page_number': img['page_number'],
                            'source_file': img['source_file'],
                            'category': img['category'],
                            'content_type': img.get('content_type', 'image')
                        })

                    if text_chunks_from_vision:
                        # テキストチャンクとしてもベクトルストアに追加
                        st.session_state.vector_store.add_text_chunks(text_chunks_from_vision, image_embeddings)
                        logging.info(f"Added {len(text_chunks_from_vision)} vision-extracted text chunks to vector store")
                else:
                    # 全ての画像解析が失敗した場合
                    error_msg = f"❌ 全ての画像解析が失敗しました ({num_images}枚)"
                    st.sidebar.error(error_msg)
                    logging.error(error_msg)

            # PDFをSupabase Storageにアップロード（Supabaseの場合）
            storage_path = None
            if st.session_state.vector_store.provider == 'supabase':
                try:
                    storage_path = st.session_state.vector_store.upload_pdf_to_storage(
                        str(doc_path), uploaded_file.name, category
                    )
                    logging.info(f"Document uploaded to Supabase Storage: {storage_path}")
                except Exception as e:
                    logging.warning(f"Failed to upload PDF to Supabase Storage: {e}")
                    # ストレージアップロード失敗してもローカルファイルはあるので処理継続

            # PDFをregistered_pdfsテーブルに登録（Supabaseの場合）
            st.session_state.vector_store.register_pdf(uploaded_file.name, category, storage_path)

            # 完了メッセージの作成
            completion_msg = f"✅ {uploaded_file.name}: テキスト {len(doc_result['text_chunks'])}件"
            if doc_result['images']:
                if analyzed_images:
                    completion_msg += f", 画像 {len(analyzed_images)}/{num_images}件"
                else:
                    completion_msg += f", 画像 0/{num_images}件（全て失敗）"
            status_text.text(completion_msg)
            progress_bar.progress((i + 1) / len(uploaded_files))

        except Exception as e:
            st.sidebar.error(f"エラー ({uploaded_file.name}): {str(e)}")
            logging.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
            continue

    status_text.success("✅ すべてのPDFの処理が完了しました！")


@st.dialog("PDF削除の確認")
def confirm_delete_dialog():
    """PDF削除の確認ダイアログ"""
    if 'delete_target' not in st.session_state:
        st.error("削除対象が指定されていません")
        return

    target_file = st.session_state.delete_target
    pdf_info = st.session_state.pdf_manager.get_pdf_info(target_file)

    if pdf_info:
        st.warning(f"以下のPDFとその関連データを削除しますか？")
        st.write(f"**ファイル名**: {pdf_info['source_file']}")
        st.write(f"**カテゴリー**: {pdf_info['category']}")
        st.write(f"**削除されるデータ**:")
        st.write(f"- テキストデータ: {pdf_info['text_count']} 件")
        st.write(f"- 画像データ: {pdf_info['image_count']} 件")
        st.write(f"- PDFファイル本体")
        st.write(f"- 抽出された画像ファイル")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 削除する", type="primary", use_container_width=True):
                # 削除実行
                with st.spinner("削除中..."):
                    result = st.session_state.pdf_manager.delete_pdf(target_file)

                if result['success']:
                    st.success(result['message'])
                    if result['category_deleted']:
                        st.info(f"カテゴリー「{pdf_info['category']}」も削除されました（他にPDFがないため）")

                    # セッション状態をクリア
                    if 'delete_target' in st.session_state:
                        del st.session_state.delete_target
                    if 'show_delete_confirm' in st.session_state:
                        del st.session_state.show_delete_confirm

                    st.rerun()
                else:
                    st.error(result['message'])

        with col2:
            if st.button("❌ キャンセル", use_container_width=True):
                # セッション状態をクリア
                if 'delete_target' in st.session_state:
                    del st.session_state.delete_target
                if 'show_delete_confirm' in st.session_state:
                    del st.session_state.show_delete_confirm
                st.rerun()
    else:
        st.error("PDFが見つかりませんでした")


def show_pdf_link(pdf_path: Path, target_file: str, key_suffix: str = ""):
    """PDFを新しいタブで開くリンクを表示（Supabase Storage対応）"""
    import os

    # Supabase ProviderでStorage URLが利用可能か確認
    if st.session_state.vector_store.provider == 'supabase':
        try:
            # Supabase StorageからPDFの署名付きURLを取得
            pdf_url = st.session_state.vector_store.get_pdf_url_from_storage(target_file)

            if pdf_url:
                # 署名付きURLを新しいタブで開くリンクとして表示
                st.markdown(
                    f'<a href="{pdf_url}" target="_blank" rel="noopener noreferrer" style="'
                    f'display: inline-block; '
                    f'width: 100%; '
                    f'padding: 0.5rem 1rem; '
                    f'background-color: #ff4b4b; '
                    f'color: white; '
                    f'text-align: center; '
                    f'text-decoration: none; '
                    f'border-radius: 0.5rem; '
                    f'font-weight: 500; '
                    f'">📖 PDFを開く（新しいタブ）</a>',
                    unsafe_allow_html=True
                )
                return
            else:
                logging.warning(f"No Supabase Storage URL found for {target_file}, falling back to local file")
        except Exception as e:
            logging.warning(f"Error getting PDF from Supabase Storage: {e}, falling back to local file")

    # フォールバック: ローカルファイルを使用
    # Streamlit Cloud環境を検出
    is_streamlit_cloud = (
        os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
        os.path.exists('/mount/src') or
        'STREAMLIT_SHARING_MODE' in os.environ
    )

    if is_streamlit_cloud:
        # Streamlit Cloudではダウンロードボタンを表示
        if pdf_path.exists():
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(
                    label="📖 PDFをダウンロード",
                    data=pdf_bytes,
                    file_name=target_file,
                    mime="application/pdf",
                    key=f"download_pdf_{key_suffix}_{target_file.replace('.', '_')}",
                    use_container_width=True
                )
        else:
            st.error(f"PDFファイルが見つかりません: {target_file}")
    else:
        # ローカル環境ではリンクを表示
        pdf_url = f"/app/static/pdfs/{target_file}"

        # 新しいタブで開くリンクを表示
        st.markdown(
            f'<a href="{pdf_url}" target="_blank" rel="noopener noreferrer" style="'
            f'display: inline-block; '
            f'width: 100%; '
            f'padding: 0.5rem 1rem; '
            f'background-color: #ff4b4b; '
            f'color: white; '
            f'text-align: center; '
            f'text-decoration: none; '
            f'border-radius: 0.5rem; '
            f'font-weight: 500; '
            f'">📖 PDFを開く（新しいタブ）</a>',
            unsafe_allow_html=True
        )


def get_pdf_path_for_preview(source_file: str) -> Path:
    """
    PDFプレビュー用のパスを取得（Supabase Storageから一時ダウンロード対応）

    Args:
        source_file: PDFファイル名

    Returns:
        Path: PDFファイルのパス
    """
    import tempfile
    import os

    pdf_path = Path("data/uploaded_pdfs") / source_file

    # ローカルファイルが存在する場合はそれを使用
    if pdf_path.exists():
        return pdf_path

    # Supabase Storageからダウンロードを試行
    if st.session_state.vector_store.provider == 'supabase':
        try:
            # 一時ディレクトリにダウンロード
            temp_dir = Path(tempfile.gettempdir()) / "pdf_preview_cache"
            temp_dir.mkdir(exist_ok=True)
            temp_pdf_path = temp_dir / source_file

            # キャッシュが存在すればそれを使用
            if temp_pdf_path.exists():
                return temp_pdf_path

            # Supabase Storageからダウンロード
            success = st.session_state.vector_store.download_pdf_from_storage(
                source_file, str(temp_pdf_path)
            )

            if success:
                logging.info(f"Downloaded PDF from Supabase Storage for preview: {source_file}")
                return temp_pdf_path
            else:
                logging.warning(f"Failed to download PDF from Supabase Storage: {source_file}")
        except Exception as e:
            logging.error(f"Error downloading PDF from Supabase Storage: {e}")

    # どちらも失敗した場合は元のパスを返す（存在しないが）
    return pdf_path


def main_area():
    """メインエリアのUI"""
    # 削除確認ダイアログの表示
    if st.session_state.get('show_delete_confirm', False):
        confirm_delete_dialog()

    st.title("📚 PDF RAG System")
    st.markdown("---")

    # 使い方ガイド（折りたたみ可能）
    # 登録済みPDFがない場合は自動展開
    registered_pdfs = st.session_state.pdf_manager.get_registered_pdfs()
    auto_expand = len(registered_pdfs) == 0

    with st.expander("📖 使い方ガイド", expanded=auto_expand):
        st.markdown("""
        ### 基本的な使い方の流れ

        このシステムは、PDFファイルをアップロードして質問に答えるRAG（Retrieval-Augmented Generation）システムです。

        #### **Step 1: PDFのアップロード** 📁
        - 左サイドバーの「PDFファイルを選択」から、PDF文書を1つまたは複数選択します
        - 最大ファイルサイズ: 50MB/ファイル

        #### **Step 2: カテゴリーの設定** 🏷️
        - PDFを分類するためのカテゴリー名を入力します
        - 例: 「製品マニュアル」「技術仕様書」「ユーザーガイド」など
        - **同じカテゴリー名**を使うことで、複数のPDFをグループ化できます

        #### **Step 3: インデックス作成** ⚙️
        - 「📑 インデックス作成」ボタンをクリックします
        - システムがPDFを解析し、テキスト・画像・グラフを抽出します
        - **処理時間の目安**: 1ページあたり2-5秒（画像の数により変動）

        #### **Step 4: 質問の入力** 💬
        - サイドバーで「🔍 検索対象カテゴリー」と「🤖 AIモデル」を選択
          - **検索対象カテゴリー**: 「全カテゴリー」またはドキュメント範囲を指定
          - **GPT-4.1**: 最新のOpenAIモデル、高度な推論能力と安定した応答品質
          - **Gemini-2.5-Pro**: マルチモーダルに強く、画像・グラフ・複雑な文書の理解に優れる
        - 最下部の入力欄に質問を入力してEnterキーまたは送信ボタンをクリック

        #### **Step 5: 回答の確認** ✅
        - AIが関連情報を元に回答を生成します
        - 各回答の下に**参照元PDFファイル**が折りたたまれて表示されます
        - 参照元を展開すると、PDFファイル名と参照したページ番号が表示されます
        - **📖 PDFを開く**ボタンをクリックすると、ブラウザの新しいタブでPDFが開きます

        ---

        ### 💡 使い方のコツ

        - **カテゴリー分けの推奨**: 製品ごと、プロジェクトごとにカテゴリーを分けると検索精度が向上します
        - **具体的な質問**: 「〇〇の仕様は？」「△△の手順を教えて」など具体的に質問すると良い結果が得られます
        - **会話メモリ機能**: 前の質問を踏まえた追加質問が可能です。セッション中の全ての会話履歴を記憶して回答します

        ---

        ### ⚠️ 注意事項

        - **データの永続化**: Streamlit Cloudでは、アプリ再起動時にアップロードしたデータは消去されます
        - **API制限**: OpenAI/Gemini APIの利用制限にご注意ください
        - **画像解析**: GEMINI_API_KEYが未設定の場合、画像解析機能は無効になります
        """)

    st.markdown("---")

    # チャット履歴表示
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

            # アシスタントの回答の場合、参照元を表示
            if chat["role"] == "assistant" and "sources" in chat and chat["sources"]:
                sources = chat["sources"]
                # sourcesは辞書形式 {"text": [...], "images": [...]}
                text_sources = sources.get("text", [])
                image_sources = sources.get("images", [])
                total_sources = len(text_sources) + len(image_sources)

                if total_sources > 0:
                    # PDFファイルごとに参照情報を集約
                    pdf_references = {}

                    # テキスト参照元を集約
                    for result in text_sources:
                        metadata = result.get("metadata", {})
                        source_file = metadata.get('source_file', 'Unknown')
                        page_number = metadata.get('page_number', 'Unknown')
                        category = metadata.get('category', 'Unknown')

                        if source_file not in pdf_references:
                            pdf_references[source_file] = {
                                'category': category,
                                'pages': set()
                            }
                        pdf_references[source_file]['pages'].add(page_number)

                    # 画像参照元を集約
                    for result in image_sources:
                        metadata = result.get("metadata", {})
                        source_file = metadata.get('source_file', 'Unknown')
                        page_number = metadata.get('page_number', 'Unknown')
                        category = metadata.get('category', 'Unknown')

                        if source_file not in pdf_references:
                            pdf_references[source_file] = {
                                'category': category,
                                'pages': set()
                            }
                        pdf_references[source_file]['pages'].add(page_number)

                    # 📸 参照ページプレビュー（上位3-5ページ）
                    top_pages = st.session_state.rag_engine.get_top_reference_pages(
                        sources,
                        top_n=5
                    )

                    if top_pages:
                        with st.expander(f"📸 参照ページプレビュー ({len(top_pages)}ページ)", expanded=True):
                            st.caption("関連度の高い順に表示しています（検索クエリをハイライト表示）")

                            # 対応する質問を取得（履歴から直前のユーザーメッセージ）
                            user_query = ""
                            if idx > 0 and st.session_state.chat_history[idx - 1]["role"] == "user":
                                user_query = st.session_state.chat_history[idx - 1]["content"]

                            # ページごとにグループ化（PDFファイル別）
                            pages_by_pdf = {}
                            for page_info in top_pages:
                                source_file = page_info['source_file']
                                if source_file not in pages_by_pdf:
                                    pages_by_pdf[source_file] = []
                                pages_by_pdf[source_file].append(page_info)

                            # PDFファイルごとに1ページずつ表示
                            for source_file, pages in pages_by_pdf.items():
                                st.markdown(f"**📄 {source_file}**")

                                if STREAMLIT_PDF_VIEWER_AVAILABLE:
                                    # pdf_viewerを使用して各ページを個別に表示
                                    try:
                                        # PDFパスを取得
                                        pdf_path = get_pdf_path(source_file, st.session_state.vector_store)

                                        if pdf_path and pdf_path.exists():
                                            # キーワード抽出（LLM使用）
                                            from src.pdf_page_renderer import extract_keywords_llm
                                            keywords = extract_keywords_llm(user_query, st.session_state.rag_engine)

                                            # 最大3列でページを表示（グリッドレイアウト維持）
                                            cols_per_row = min(3, len(pages))
                                            for i in range(0, len(pages), cols_per_row):
                                                cols = st.columns(cols_per_row)
                                                for col_idx, page_info in enumerate(pages[i:i + cols_per_row]):
                                                    page_num = page_info['page_number']
                                                    score = page_info.get('score')

                                                    with cols[col_idx]:
                                                        # 該当ページのアノテーションのみ生成
                                                        annotations = create_pdf_annotations_pymupdf(
                                                            pdf_path=pdf_path,
                                                            search_terms=keywords,
                                                            page_numbers=[page_num]  # 1ページのみ
                                                        )

                                                        # キャプション作成
                                                        caption = f"ページ {page_num}"
                                                        if score is not None:
                                                            caption += f" (関連度: {score:.3f})"
                                                        st.markdown(f"**{caption}**")

                                                        # PDFビューアーで1ページのみ表示
                                                        logger.info(f"📄 [HISTORY] Displaying page {page_num} with {len(annotations)} annotations")
                                                        pdf_viewer(
                                                            str(pdf_path),
                                                            annotations=annotations,
                                                            pages_to_render=[page_num],  # 該当ページのみ
                                                            width=350,
                                                            height=500,
                                                            render_text=True
                                                        )

                                                        # 内容プレビュー
                                                        with st.expander("📝 内容プレビュー"):
                                                            st.text(page_info.get('content_preview', ''))
                                        else:
                                            st.error(f"PDFファイルが見つかりません: {source_file}")

                                    except Exception as e:
                                        logger.error(f"PDF display error: {e}", exc_info=True)
                                        st.error(f"PDFの表示に失敗しました: {e}")

                                else:
                                    # フォールバック: 画像ベースの表示
                                    st.warning("streamlit-pdf-viewerが利用できません。画像表示にフォールバックします。")

                                    # 最大3列でページを表示
                                    cols_per_row = min(3, len(pages))
                                    for i in range(0, len(pages), cols_per_row):
                                        cols = st.columns(cols_per_row)
                                        for col_idx, page_info in enumerate(pages[i:i + cols_per_row]):
                                            page_num = page_info['page_number']
                                            score = page_info.get('score')

                                            with cols[col_idx]:
                                                # ハイライト付き画像を取得
                                                logger.info(f"📸 [HISTORY] About to call extract_page_with_highlight: {source_file} page {page_num}")
                                                image = extract_page_with_highlight(
                                                    source_file=source_file,
                                                    page_number=page_num,
                                                    query=user_query,
                                                    _vector_store=st.session_state.vector_store,
                                                    _rag_engine=st.session_state.rag_engine,
                                                    _vision_analyzer=st.session_state.vision_analyzer,
                                                    dpi=150,
                                                    target_width=1000
                                                )
                                                logger.info(f"📸 [HISTORY] extract_page_with_highlight returned: {type(image).__name__ if image else 'None'}")

                                                if image:
                                                    # キャプション作成
                                                    caption = f"ページ {page_num}"
                                                    if score is not None:
                                                        caption += f" (関連度: {score:.3f})"

                                                    st.image(image, caption=caption, use_container_width=True)

                                                    # 内容プレビュー
                                                with st.expander("📝 内容プレビュー"):
                                                    st.text(page_info.get('content_preview', ''))
                                            else:
                                                st.warning(f"ページ {page_num} の画像を取得できませんでした")

                                st.markdown("---")

                    # PDFファイルごとに表示
                    with st.expander(f"📄 参照元PDFファイル ({len(pdf_references)}件)"):
                        for pdf_idx, (source_file, info) in enumerate(pdf_references.items(), 1):
                            pages_list = sorted(list(info['pages']))
                            pages_str = ', '.join(map(str, pages_list))

                            st.markdown(f"**{pdf_idx}. {source_file}**")
                            st.write(f"📂 カテゴリー: {info['category']}")
                            st.write(f"📄 参照ページ: {pages_str}")

                            # PDFダウンロードボタン
                            pdf_path = Path("data/uploaded_pdfs") / source_file
                            show_pdf_link(pdf_path, source_file, key_suffix=f"hist_{idx}_pdf_{pdf_idx}")

                            if pdf_idx < len(pdf_references):
                                st.markdown("---")

    # 標準チャット入力
    question = st.chat_input("💬 質問を入力してください（例: この製品の主な特徴は何ですか？）")

    # 質問が送信された場合
    if question:
        # カテゴリーフィルター設定
        category_filter = None if st.session_state.selected_category == "全カテゴリー" else st.session_state.selected_category

        # モデル表示名を取得
        model_display_names = {
            "openai": "GPT-4.1",
            "gemini": "Gemini-2.5-Pro"
        }
        current_model_display = model_display_names.get(st.session_state.selected_model, "GPT-4.1")

        try:
            # ユーザーの質問を表示
            with st.chat_message("user"):
                st.markdown(question)

            # チャット履歴にユーザーの質問を追加
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            # アシスタントの回答を表示
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                full_answer = ""
                result_data = None
                context_data = None

                try:
                    # ストリーミング表示
                    # 最後のユーザーメッセージを除いた履歴を渡す（現在の質問は含めない）
                    chat_history_for_query = [msg for msg in st.session_state.chat_history[:-1]]

                    for chunk_data in st.session_state.rag_engine.query_stream(
                        question,
                        category_filter,
                        model_type=st.session_state.selected_model,
                        chat_history=chat_history_for_query
                    ):
                        if chunk_data["type"] == "context":
                            # コンテキスト情報を保存
                            context_data = chunk_data
                            logger.info(f"[DEBUG] Context data received: sources={len(chunk_data.get('sources', {}).get('text', []))} text, {len(chunk_data.get('sources', {}).get('images', []))} images")
                        elif chunk_data["type"] == "chunk":
                            full_answer += chunk_data["content"]
                            answer_placeholder.markdown(full_answer + "▌")  # カーソル表示

                    # ストリーミング完了後、最終的な回答を表示
                    answer_placeholder.markdown(full_answer)

                    # 結果データを構築
                    if context_data:
                        result_data = {
                            "answer": full_answer,
                            "sources": context_data.get("sources", {}),
                            "context": context_data.get("context", ""),
                            "images": context_data.get("images", [])
                        }
                        logger.info(f"[DEBUG] result_data constructed: sources={len(result_data.get('sources', {}).get('text', []))} text, {len(result_data.get('sources', {}).get('images', []))} images")
                    else:
                        logger.warning("[DEBUG] context_data is None - result_data remains None")

                except Exception as stream_error:
                    # ストリーミングエラー時は通常モードにフォールバック
                    if "stream" in str(stream_error).lower() or "unsupported_value" in str(stream_error).lower():
                        st.warning("⚠️ ストリーミングモードが利用できません。通常モードで回答を生成します...")
                        answer_placeholder.empty()
                        with st.spinner(f"回答を生成中... ({current_model_display})"):
                            chat_history_for_query = [msg for msg in st.session_state.chat_history[:-1]]

                            result_data = st.session_state.rag_engine.query(
                                question,
                                category_filter,
                                model_type=st.session_state.selected_model,
                                chat_history=chat_history_for_query
                            )
                        answer_placeholder.markdown(result_data['answer'])
                    else:
                        raise stream_error

                # 参照元を折りたたみ表示
                if result_data and result_data.get('sources'):
                    sources = result_data['sources']
                    # sourcesは辞書形式 {"text": [...], "images": [...]}
                    text_sources = sources.get("text", [])
                    image_sources = sources.get("images", [])
                    total_sources = len(text_sources) + len(image_sources)

                    if total_sources > 0:
                        # PDFファイルごとに参照情報を集約
                        pdf_references = {}

                        # テキスト参照元を集約
                        for result in text_sources:
                            metadata = result.get("metadata", {})
                            source_file = metadata.get('source_file', 'Unknown')
                            page_number = metadata.get('page_number', 'Unknown')
                            category = metadata.get('category', 'Unknown')

                            if source_file not in pdf_references:
                                pdf_references[source_file] = {
                                    'category': category,
                                    'pages': set()
                                }
                            pdf_references[source_file]['pages'].add(page_number)

                        # 画像参照元を集約
                        for result in image_sources:
                            metadata = result.get("metadata", {})
                            source_file = metadata.get('source_file', 'Unknown')
                            page_number = metadata.get('page_number', 'Unknown')
                            category = metadata.get('category', 'Unknown')

                            if source_file not in pdf_references:
                                pdf_references[source_file] = {
                                    'category': category,
                                    'pages': set()
                                }
                            pdf_references[source_file]['pages'].add(page_number)

                        # 📸 参照ページプレビュー（上位3-5ページ）
                        logger.info(f"[DEBUG] Checking page preview condition: result_data={result_data is not None}, has_sources={result_data.get('sources') if result_data else None}")
                        if result_data and result_data.get('sources'):
                            logger.info(f"[DEBUG] Calling get_top_reference_pages with sources")
                            top_pages = st.session_state.rag_engine.get_top_reference_pages(
                                result_data['sources'],
                                top_n=5
                            )
                            logger.info(f"[DEBUG] get_top_reference_pages returned {len(top_pages)} pages")
                        else:
                            logger.warning("[DEBUG] Skipping page preview - result_data or sources missing")
                            top_pages = []

                        if top_pages:
                            with st.expander(f"📸 参照ページプレビュー ({len(top_pages)}ページ)", expanded=True):
                                st.caption("関連度の高い順に表示しています（検索クエリをハイライト表示）")

                                # ページごとにグループ化（PDFファイル別）
                                pages_by_pdf = {}
                                for page_info in top_pages:
                                    source_file = page_info['source_file']
                                    if source_file not in pages_by_pdf:
                                        pages_by_pdf[source_file] = []
                                    pages_by_pdf[source_file].append(page_info)

                                # PDFファイルごとに1ページずつ表示
                                for source_file, pages in pages_by_pdf.items():
                                    st.markdown(f"**📄 {source_file}**")

                                    if STREAMLIT_PDF_VIEWER_AVAILABLE:
                                        # pdf_viewerを使用して各ページを個別に表示
                                        try:
                                            # PDFパスを取得
                                            pdf_path = get_pdf_path(source_file, st.session_state.vector_store)

                                            if pdf_path and pdf_path.exists():
                                                # キーワード抽出（LLM使用）
                                                from src.pdf_page_renderer import extract_keywords_llm
                                                keywords = extract_keywords_llm(question, st.session_state.rag_engine)

                                                # 最大3列でページを表示（グリッドレイアウト維持）
                                                cols_per_row = min(3, len(pages))
                                                for i in range(0, len(pages), cols_per_row):
                                                    cols = st.columns(cols_per_row)
                                                    for col_idx, page_info in enumerate(pages[i:i + cols_per_row]):
                                                        page_num = page_info['page_number']
                                                        score = page_info.get('score')

                                                        with cols[col_idx]:
                                                            # 該当ページのアノテーションのみ生成
                                                            annotations = create_pdf_annotations_pymupdf(
                                                                pdf_path=pdf_path,
                                                                search_terms=keywords,
                                                                page_numbers=[page_num]  # 1ページのみ
                                                            )

                                                            # キャプション作成
                                                            caption = f"ページ {page_num}"
                                                            if score is not None:
                                                                caption += f" (関連度: {score:.3f})"
                                                            st.markdown(f"**{caption}**")

                                                            # PDFビューアーで1ページのみ表示
                                                            logger.info(f"📄 [NEW ANSWER] Displaying page {page_num} with {len(annotations)} annotations")
                                                            pdf_viewer(
                                                                str(pdf_path),
                                                                annotations=annotations,
                                                                pages_to_render=[page_num],  # 該当ページのみ
                                                                width=350,
                                                                height=500,
                                                                render_text=True
                                                            )

                                                            # 内容プレビュー
                                                            with st.expander("📝 内容プレビュー"):
                                                                st.text(page_info.get('content_preview', ''))
                                            else:
                                                st.error(f"PDFファイルが見つかりません: {source_file}")

                                        except Exception as e:
                                            logger.error(f"PDF display error: {e}", exc_info=True)
                                            st.error(f"PDFの表示に失敗しました: {e}")

                                    else:
                                        # フォールバック: 画像ベースの表示
                                        st.warning("streamlit-pdf-viewerが利用できません。画像表示にフォールバックします。")

                                        # 最大3列でページを表示
                                        cols_per_row = min(3, len(pages))
                                        for i in range(0, len(pages), cols_per_row):
                                            cols = st.columns(cols_per_row)
                                            for col_idx, page_info in enumerate(pages[i:i + cols_per_row]):
                                                page_num = page_info['page_number']
                                                score = page_info.get('score')

                                                with cols[col_idx]:
                                                    # ハイライト付き画像を取得
                                                    logger.info(f"📸 [NEW ANSWER] About to call extract_page_with_highlight: {source_file} page {page_num}")
                                                    image = extract_page_with_highlight(
                                                        source_file=source_file,
                                                        page_number=page_num,
                                                        query=question,  # 検索クエリをハイライト
                                                        _vector_store=st.session_state.vector_store,
                                                        _rag_engine=st.session_state.rag_engine,
                                                        _vision_analyzer=st.session_state.vision_analyzer,
                                                        dpi=150,
                                                        target_width=1000
                                                    )
                                                    logger.info(f"📸 [NEW ANSWER] extract_page_with_highlight returned: {type(image).__name__ if image else 'None'}")

                                                    if image:
                                                        # キャプション作成
                                                        caption = f"ページ {page_num}"
                                                        if score is not None:
                                                            caption += f" (関連度: {score:.3f})"

                                                        st.image(image, caption=caption, use_container_width=True)

                                                        # 内容プレビュー
                                                        with st.expander("📝 内容プレビュー"):
                                                            st.text(page_info.get('content_preview', ''))
                                                    else:
                                                        st.warning(f"ページ {page_num} の画像を取得できませんでした")

                                    st.markdown("---")

                        # PDFファイルごとに表示
                        with st.expander(f"📄 参照元PDFファイル ({len(pdf_references)}件)"):
                            for pdf_idx, (source_file, info) in enumerate(pdf_references.items(), 1):
                                pages_list = sorted(list(info['pages']))
                                pages_str = ', '.join(map(str, pages_list))

                                st.markdown(f"**{pdf_idx}. {source_file}**")
                                st.write(f"📂 カテゴリー: {info['category']}")
                                st.write(f"📄 参照ページ: {pages_str}")

                                # PDFダウンロードボタン
                                pdf_path = Path("data/uploaded_pdfs") / source_file
                                show_pdf_link(pdf_path, source_file, key_suffix=f"new_pdf_{pdf_idx}")

                                if pdf_idx < len(pdf_references):
                                    st.markdown("---")

            # チャット履歴にアシスタントの回答を追加（参照元も含む）
            if result_data:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result_data['answer'],
                    "sources": result_data.get('sources', [])
                })

                # 再描画して履歴を更新
                st.rerun()
            else:
                st.error("回答の生成に失敗しました")
                # ユーザーの質問を履歴から削除
                st.session_state.chat_history.pop()

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logging.error(f"Error during query: {e}", exc_info=True)
            # エラー時はユーザーの質問を履歴から削除
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                st.session_state.chat_history.pop()


def main():
    """メインエントリーポイント"""
    # 初期化
    config = initialize_app()

    # サイドバー
    sidebar()

    # メインエリア
    main_area()

    # フッター
    st.markdown("---")
    st.caption("PDF RAG System v0.1.0 - PoC")


if __name__ == "__main__":
    main()
