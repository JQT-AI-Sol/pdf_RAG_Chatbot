"""
Streamlit Application for PDF RAG System
"""

import streamlit as st
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# src モジュールのインポート
from src.utils import load_config, load_environment, ensure_directories, setup_logging, encode_pdf_to_base64
from src.pdf_processor import PDFProcessor
from src.text_embedder import TextEmbedder
from src.vision_analyzer import VisionAnalyzer
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.pdf_manager import PDFManager


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
        st.session_state.pdf_processor = PDFProcessor(config)
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
        st.session_state.uploaded_chat_images = []  # チャット入力時の画像添付用

        # Vision Analyzerの状態チェック
        if not st.session_state.vision_analyzer.api_key_valid:
            st.session_state.vision_disabled = True
            logger.warning("Vision analysis is disabled due to missing or invalid GEMINI_API_KEY")
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
            "GEMINI_API_KEYが設定されていません。\n"
            "画像やグラフの解析を有効にするには、.envファイルにGEMINI_API_KEYを設定してください。"
        )

    # PDFアップロード
    st.sidebar.subheader("PDFアップロード")
    uploaded_files = st.sidebar.file_uploader(
        "PDFファイルを選択",
        type=['pdf'],
        accept_multiple_files=True
    )

    # カテゴリー入力
    category = st.sidebar.text_input(
        "カテゴリー名",
        placeholder="例: 製品マニュアル"
    )

    # インデックス作成ボタン
    if st.sidebar.button("📑 インデックス作成", type="primary"):
        if uploaded_files and category:
            process_pdfs(uploaded_files, category)
        else:
            st.sidebar.error("PDFファイルとカテゴリー名を入力してください")

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
                    if pdf_path.exists():
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
        "GPT-4o": "openai",
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
        st.session_state.uploaded_chat_images = []
        # 入力フォームをクリアするためのフラグ
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0
        st.session_state.chat_input_key += 1
        st.rerun()


def process_pdfs(uploaded_files, category):
    """PDFファイルを処理"""
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

            # 1. PDFを保存（data/uploaded_pdfs/ と static/pdfs/ の両方）
            status_text.text(f"処理中: {uploaded_file.name} (1/?) - PDF保存中...")
            pdf_path = Path("data/uploaded_pdfs") / uploaded_file.name
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            static_pdf_path = Path("static/pdfs") / uploaded_file.name
            static_pdf_path.parent.mkdir(parents=True, exist_ok=True)

            pdf_bytes = uploaded_file.getbuffer()
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            with open(static_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            # 2. テキスト・画像抽出
            status_text.text(f"処理中: {uploaded_file.name} (2/?) - テキスト・画像抽出中...")
            try:
                logging.info(f"Starting PDF processing for {uploaded_file.name}")
                pdf_result = st.session_state.pdf_processor.process_pdf(str(pdf_path), category)
                logging.info(f"PDF processing completed for {uploaded_file.name}: {len(pdf_result.get('text_chunks', []))} text chunks, {len(pdf_result.get('images', []))} images")
            except Exception as e:
                error_msg = f"PDF処理中にエラーが発生しました: {str(e)}"
                logging.error(error_msg, exc_info=True)
                st.sidebar.error(error_msg)
                continue

            # 総ステップ数を決定（画像があれば5、なければ4）
            total_steps = 5 if pdf_result['images'] else 4
            num_pages = pdf_result.get('total_pages', '?')
            num_chunks = len(pdf_result['text_chunks'])
            num_images = len(pdf_result['images'])

            # 3. テキストチャンクをエンベディング（バッチ処理）
            status_text.text(f"処理中: {uploaded_file.name} (3/{total_steps}) - テキストエンベディング中（{num_chunks}チャンク）...")
            if pdf_result['text_chunks']:
                # 全テキストをまとめてバッチ処理
                texts = [chunk['text'] for chunk in pdf_result['text_chunks']]
                text_embeddings = st.session_state.embedder.embed_batch(texts)

                # ベクトルストアに追加
                st.session_state.vector_store.add_text_chunks(
                    pdf_result['text_chunks'],
                    text_embeddings
                )

            # 4. 画像をVision AIで解析（並列処理）- 画像がある場合のみ
            analyzed_images = []
            failed_images = []

            if pdf_result['images']:
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
                    futures = {executor.submit(analyze_single_image, img, vision_analyzer): img for img in pdf_result['images']}
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

            # PDFをregistered_pdfsテーブルに登録（Supabaseの場合）
            st.session_state.vector_store.register_pdf(uploaded_file.name, category)

            # 完了メッセージの作成
            completion_msg = f"✅ {uploaded_file.name}: テキスト {len(pdf_result['text_chunks'])}件"
            if pdf_result['images']:
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


@st.dialog("📎 画像を添付")
def show_image_upload_dialog():
    """画像アップロードダイアログ"""
    st.write("質問と一緒に送信する画像を選択してください（最大5枚）")

    # 画像アップローダー
    uploaded_files = st.file_uploader(
        "画像を選択（PNG, JPG, JPEG, WEBP）",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        key="image_uploader_dialog"
    )

    # アップロードされた画像をセッション状態に保存
    if uploaded_files:
        from io import BytesIO
        st.session_state.uploaded_chat_images = [BytesIO(f.read()) for f in uploaded_files[:5]]

    # 現在添付されている画像の表示
    if st.session_state.get('uploaded_chat_images', []):
        st.markdown("---")
        st.subheader(f"📷 添付された画像: {len(st.session_state.uploaded_chat_images)}枚")

        # プレビュー表示
        cols = st.columns(min(len(st.session_state.uploaded_chat_images), 3))
        for idx, (col, img_bytes) in enumerate(zip(cols, st.session_state.uploaded_chat_images)):
            with col:
                img_bytes.seek(0)
                st.image(img_bytes, use_container_width=True, caption=f"画像 {idx+1}")
                if st.button(f"🗑️ 削除", key=f"remove_img_dialog_{idx}", use_container_width=True):
                    st.session_state.uploaded_chat_images.pop(idx)
                    st.rerun()

        st.markdown("---")

        # アクションボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 完了", type="primary", use_container_width=True):
                st.session_state.show_image_dialog = False
                st.rerun()
        with col2:
            if st.button("🗑️ すべて削除", type="secondary", use_container_width=True):
                st.session_state.uploaded_chat_images = []
                st.rerun()
    else:
        st.info("画像がまだ選択されていません")

        # 閉じるボタン
        if st.button("閉じる", use_container_width=True):
            st.session_state.show_image_dialog = False
            st.rerun()


def show_pdf_link(pdf_path: Path, target_file: str, key_suffix: str = ""):
    """PDFを新しいタブで開くリンクまたはダウンロードボタンを表示"""
    import os

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

        # アイコンのみのリンクを表示（ホバー時に説明文表示）
        st.markdown(
            f'<a href="{pdf_url}" target="_blank" title="PDFを閲覧する" style="'
            f'display: inline-block; '
            f'width: 100%; '
            f'padding: 0.5rem 1rem; '
            f'background-color: #ff4b4b; '
            f'color: white; '
            f'text-align: center; '
            f'text-decoration: none; '
            f'border-radius: 0.5rem; '
            f'font-size: 1.2rem; '
            f'">📖</a>',
            unsafe_allow_html=True
        )


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
          - **GPT-4o**: 高度な推論能力と安定した応答品質
          - **Gemini-2.5-Pro**: マルチモーダルに強く、画像・グラフ・複雑な文書の理解に優れる
        - 最下部の入力欄に質問を入力してEnterキーまたは送信ボタンをクリック

        #### **Step 5: 回答の確認** ✅
        - AIが関連情報を元に回答を生成します
        - 各回答の下に**参照元**が折りたたまれて表示されます
        - 参照元を展開すると、回答の根拠となったPDFのページを確認できます

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
                    with st.expander(f"📄 参照元 ({total_sources}件)"):
                        source_idx = 1

                        # テキスト参照元
                        for result in text_sources:
                            metadata = result.get("metadata", {})
                            st.markdown(f"**参照 {source_idx}: {metadata.get('source_file', 'Unknown')} (ページ {metadata.get('page_number', 'Unknown')})**")
                            st.write(f"**カテゴリー**: {metadata.get('category', 'Unknown')}")
                            st.write(f"**タイプ**: テキスト")

                            # PDF全体を閲覧ボタン
                            source_file = metadata.get('source_file')
                            if source_file:
                                pdf_path = Path("data/uploaded_pdfs") / source_file
                                if pdf_path.exists():
                                    show_pdf_link(pdf_path, source_file, key_suffix=f"hist_{idx}_text_ref_{source_idx}")

                                st.markdown("---")

                                # 元のPDFページを表示
                                if pdf_path.exists():
                                    try:
                                        import pdfplumber

                                        page_number = metadata.get('page_number', 1)
                                        with pdfplumber.open(str(pdf_path)) as pdf:
                                            if page_number <= len(pdf.pages):
                                                page = pdf.pages[page_number - 1]
                                                page_img = page.to_image(resolution=150)
                                                st.image(page_img.original, use_container_width=True)
                                            else:
                                                st.warning(f"ページ {page_number} が見つかりません")
                                    except Exception as e:
                                        st.error(f"PDFページの表示に失敗しました: {e}")

                            if source_idx < total_sources:
                                st.markdown("---")
                            source_idx += 1

                        # 画像参照元
                        for result in image_sources:
                            metadata = result.get("metadata", {})
                            st.markdown(f"**参照 {source_idx}: {metadata.get('source_file', 'Unknown')} (ページ {metadata.get('page_number', 'Unknown')})**")
                            st.write(f"**カテゴリー**: {metadata.get('category', 'Unknown')}")
                            st.write(f"**タイプ**: {metadata.get('content_type', '画像')}")

                            # PDF全体を閲覧ボタン
                            source_file = metadata.get('source_file')
                            if source_file:
                                pdf_path = Path("data/uploaded_pdfs") / source_file
                                if pdf_path.exists():
                                    show_pdf_link(pdf_path, source_file, key_suffix=f"hist_{idx}_image_ref_{source_idx}")

                                st.markdown("---")

                                # 元のPDFページを表示
                                if pdf_path.exists():
                                    try:
                                        import pdfplumber

                                        page_number = metadata.get('page_number', 1)
                                        with pdfplumber.open(str(pdf_path)) as pdf:
                                            if page_number <= len(pdf.pages):
                                                page = pdf.pages[page_number - 1]
                                                page_img = page.to_image(resolution=150)
                                                st.image(page_img.original, use_container_width=True)
                                            else:
                                                st.warning(f"ページ {page_number} が見つかりません")
                                    except Exception as e:
                                        st.error(f"PDFページの表示に失敗しました: {e}")

                            if source_idx < total_sources:
                                st.markdown("---")
                            source_idx += 1

    # 添付画像のプレビュー表示
    num_images = len(st.session_state.get('uploaded_chat_images', []))
    if num_images > 0:
        st.caption(f"📷 {num_images}枚の画像が添付されています")

    # 画像アップロードダイアログ
    if st.session_state.get('show_image_dialog', False):
        show_image_upload_dialog()

    # カスタムチャット入力（ボタンを同じ行に配置）
    col1, col2, col3 = st.columns([0.6, 8, 1])

    with col1:
        # 📎ボタン（画像添付）
        button_label = f"📎 {num_images}" if num_images > 0 else "📎"
        if st.button(button_label, key="open_image_dialog", help="画像を添付する", use_container_width=True):
            st.session_state.show_image_dialog = True

    with col2:
        # テキスト入力（リセット時にクリアするため動的キーを使用）
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0
        question = st.text_input(
            "質問を入力",
            key=f"chat_input_{st.session_state.chat_input_key}",
            placeholder="💬 質問を入力してください（例: この製品の主な特徴は何ですか？）",
            label_visibility="collapsed"
        )

    with col3:
        # 送信ボタン（動的キーを使用）
        send_button = st.button("▶", key=f"send_button_{st.session_state.chat_input_key}", help="送信", use_container_width=True, type="primary")

    # 質問が送信された場合（送信ボタンのみ）
    if send_button and question:
        # カテゴリーフィルター設定
        category_filter = None if st.session_state.selected_category == "全カテゴリー" else st.session_state.selected_category

        # モデル表示名を取得
        model_display_names = {
            "openai": "GPT-4o",
            "gemini": "Gemini-2.5-Pro"
        }
        current_model_display = model_display_names.get(st.session_state.selected_model, "GPT-4o")

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

                    # アップロードされた画像を取得（コピーを作成）
                    from io import BytesIO
                    uploaded_images = None
                    if st.session_state.uploaded_chat_images:
                        # BytesIOオブジェクトのコピーを作成
                        uploaded_images = [BytesIO(img.getvalue()) for img in st.session_state.uploaded_chat_images]

                    for chunk_data in st.session_state.rag_engine.query_stream(
                        question,
                        category_filter,
                        model_type=st.session_state.selected_model,
                        chat_history=chat_history_for_query,
                        uploaded_images=uploaded_images
                    ):
                        if chunk_data["type"] == "context":
                            # コンテキスト情報を保存
                            context_data = chunk_data
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

                except Exception as stream_error:
                    # ストリーミングエラー時は通常モードにフォールバック
                    if "stream" in str(stream_error).lower() or "unsupported_value" in str(stream_error).lower():
                        st.warning("⚠️ ストリーミングモードが利用できません。通常モードで回答を生成します...")
                        answer_placeholder.empty()
                        with st.spinner(f"回答を生成中... ({current_model_display})"):
                            chat_history_for_query = [msg for msg in st.session_state.chat_history[:-1]]
                            # アップロードされた画像を取得（コピーを作成）
                            from io import BytesIO
                            uploaded_images = None
                            if st.session_state.uploaded_chat_images:
                                # BytesIOオブジェクトのコピーを作成
                                uploaded_images = [BytesIO(img.getvalue()) for img in st.session_state.uploaded_chat_images]

                            result_data = st.session_state.rag_engine.query(
                                question,
                                category_filter,
                                model_type=st.session_state.selected_model,
                                chat_history=chat_history_for_query,
                                uploaded_images=uploaded_images
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
                        with st.expander(f"📄 参照元 ({total_sources}件)"):
                            source_idx = 1

                            # テキスト参照元
                            for result in text_sources:
                                metadata = result.get("metadata", {})
                                st.markdown(f"**参照 {source_idx}: {metadata.get('source_file', 'Unknown')} (ページ {metadata.get('page_number', 'Unknown')})**")
                                st.write(f"**カテゴリー**: {metadata.get('category', 'Unknown')}")
                                st.write(f"**タイプ**: テキスト")

                                # PDF全体を閲覧ボタン
                                source_file = metadata.get('source_file')
                                if source_file:
                                    pdf_path = Path("data/uploaded_pdfs") / source_file
                                    if pdf_path.exists():
                                        show_pdf_link(pdf_path, source_file, key_suffix=f"new_text_ref_{source_idx}")

                                    st.markdown("---")

                                    # 元のPDFページを表示
                                    if pdf_path.exists():
                                        try:
                                            import pdfplumber

                                            page_number = metadata.get('page_number', 1)
                                            with pdfplumber.open(str(pdf_path)) as pdf:
                                                if page_number <= len(pdf.pages):
                                                    page = pdf.pages[page_number - 1]
                                                    page_img = page.to_image(resolution=150)
                                                    st.image(page_img.original, use_container_width=True)
                                                else:
                                                    st.warning(f"ページ {page_number} が見つかりません")
                                        except Exception as e:
                                            st.error(f"PDFページの表示に失敗しました: {e}")

                                if source_idx < total_sources:
                                    st.markdown("---")
                                source_idx += 1

                            # 画像参照元
                            for result in image_sources:
                                metadata = result.get("metadata", {})
                                st.markdown(f"**参照 {source_idx}: {metadata.get('source_file', 'Unknown')} (ページ {metadata.get('page_number', 'Unknown')})**")
                                st.write(f"**カテゴリー**: {metadata.get('category', 'Unknown')}")
                                st.write(f"**タイプ**: {metadata.get('content_type', '画像')}")

                                # PDF全体を閲覧ボタン
                                source_file = metadata.get('source_file')
                                if source_file:
                                    pdf_path = Path("data/uploaded_pdfs") / source_file
                                    if pdf_path.exists():
                                        show_pdf_link(pdf_path, source_file, key_suffix=f"new_image_ref_{source_idx}")

                                    st.markdown("---")

                                    # 元のPDFページを表示
                                    if pdf_path.exists():
                                        try:
                                            import pdfplumber

                                            page_number = metadata.get('page_number', 1)
                                            with pdfplumber.open(str(pdf_path)) as pdf:
                                                if page_number <= len(pdf.pages):
                                                    page = pdf.pages[page_number - 1]
                                                    page_img = page.to_image(resolution=150)
                                                    st.image(page_img.original, use_container_width=True)
                                                else:
                                                    st.warning(f"ページ {page_number} が見つかりません")
                                        except Exception as e:
                                            st.error(f"PDFページの表示に失敗しました: {e}")

                                if source_idx < total_sources:
                                    st.markdown("---")
                                source_idx += 1

            # チャット履歴にアシスタントの回答を追加（参照元も含む）
            if result_data:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result_data['answer'],
                    "sources": result_data.get('sources', [])
                })

                # アップロードされた画像をクリア
                st.session_state.uploaded_chat_images = []

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
