"""
Streamlit Application for PDF RAG System
"""

import streamlit as st
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# src モジュールのインポート
from src.utils import load_config, load_environment, ensure_directories, setup_logging, encode_pdf_to_base64
from src.category_manager import CategoryManager
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
        st.session_state.category_manager = CategoryManager(
            config['category']['storage_file']
        )
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
            st.session_state.category_manager,
            config
        )
        st.session_state.chat_history = []

    return config


def sidebar():
    """サイドバーのUI"""
    st.sidebar.title("📁 ドキュメント管理")

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
    categories = st.session_state.category_manager.get_all_categories()
    if categories:
        for cat in categories:
            st.sidebar.text(f"• {cat}")
    else:
        st.sidebar.info("まだカテゴリーが登録されていません")

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

    # チャットリセットボタン
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ チャット履歴をリセット", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


def process_pdfs(uploaded_files, category):
    """PDFファイルを処理"""
    # カテゴリー登録
    st.session_state.category_manager.add_category(category)

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
            pdf_result = st.session_state.pdf_processor.process_pdf(str(pdf_path), category)

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
            if pdf_result['images']:
                status_text.text(f"処理中: {uploaded_file.name} (4/{total_steps}) - 画像解析中（{num_images}枚）...")
                max_workers = st.session_state.config.get('performance', {}).get('max_workers', 4)
                analyzed_images = []

                # 画像解析を並列処理
                def analyze_single_image(image_data):
                    try:
                        actual_content_type = image_data.get('content_type', 'image')
                        analysis = st.session_state.vision_analyzer.analyze_image(
                            image_data['image_path'],
                            content_type=actual_content_type
                        )
                        # メタデータを統合
                        image_data.update({
                            'category': category,
                            'content_type': analysis.get('content_type', 'image'),
                            'description': analysis['description']
                        })
                        return image_data
                    except Exception as e:
                        logging.error(f"Error analyzing image: {e}")
                        return None

                # ThreadPoolExecutorで並列処理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(analyze_single_image, img): img for img in pdf_result['images']}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            analyzed_images.append(result)

                # 解析結果をバッチでエンベディング
                if analyzed_images:
                    descriptions = [img['description'] for img in analyzed_images]
                    image_embeddings = st.session_state.embedder.embed_batch(descriptions)

                    # ベクトルストアにバッチで追加
                    st.session_state.vector_store.add_image_contents_batch(analyzed_images, image_embeddings)

            # 完了
            status_text.text(f"処理中: {uploaded_file.name} ({total_steps}/{total_steps}) - 完了！")
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
    """PDFを新しいタブで開くリンクを表示（アイコンのみ）"""
    # 静的ファイルのURLを生成
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

    # カテゴリーとモデル選択を横並びに
    col1, col2 = st.columns([2, 1])

    with col1:
        categories = ["全カテゴリー"] + st.session_state.category_manager.get_all_categories()
        selected_category = st.selectbox(
            "🔍 検索対象カテゴリー",
            categories,
            help="質問する対象のカテゴリーを選択してください"
        )

    with col2:
        model_options = {
            "GPT-4o-mini": "openai",
            "Gemini-2.5-flash": "gemini"
        }
        selected_model_display = st.selectbox(
            "🤖 AIモデル",
            list(model_options.keys()),
            help="使用するAIモデルを選択"
        )
        selected_model = model_options[selected_model_display]

    # 質問入力
    question = st.text_input(
        "💬 質問を入力してください",
        placeholder="例: この製品の主な特徴は何ですか？"
    )

    # 質問ボタン
    if st.button("🔍 質問する", type="primary"):
        if question:
            # カテゴリーフィルター設定
            category_filter = None if selected_category == "全カテゴリー" else selected_category

            try:
                # チャット履歴にユーザーの質問を追加
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })

                # 回答生成（ストリーミング表示、失敗時は通常モードにフォールバック）
                st.markdown("### 💬 回答")
                answer_placeholder = st.empty()
                full_answer = ""
                result_data = None

                try:
                    # ストリーミング表示
                    for chunk_data in st.session_state.rag_engine.query_stream(question, category_filter, model_type=selected_model):
                        if chunk_data["type"] == "chunk":
                            full_answer += chunk_data["content"]
                            answer_placeholder.markdown(full_answer + "▌")  # カーソル表示
                        elif chunk_data["type"] == "final":
                            answer_placeholder.markdown(full_answer)
                            result_data = chunk_data
                except Exception as stream_error:
                    # ストリーミングエラー時は通常モードにフォールバック
                    if "stream" in str(stream_error).lower() or "unsupported_value" in str(stream_error).lower():
                        st.warning("⚠️ ストリーミングモードが利用できません。通常モードで回答を生成します...")
                        answer_placeholder.empty()
                        with st.spinner(f"回答を生成中... ({selected_model_display})"):
                            result_data = st.session_state.rag_engine.query(question, category_filter, model_type=selected_model)
                        answer_placeholder.markdown(result_data['answer'])
                    else:
                        raise stream_error

                # 参照元表示
                if result_data:
                    st.markdown("### 📄 参照元")
                    if result_data['sources']:
                        for idx, source in enumerate(result_data['sources'], 1):
                            with st.expander(f"参照 {idx}: {source['file']} (ページ {source['page']})"):
                                st.write(f"**カテゴリー**: {source['category']}")
                                st.write(f"**タイプ**: {source['type']}")

                                # PDF全体を閲覧ボタン
                                pdf_path = Path("data/uploaded_pdfs") / source['file']
                                if pdf_path.exists():
                                    show_pdf_link(pdf_path, source['file'], key_suffix=f"ref_{idx}")
                                else:
                                    st.error(f"PDFファイルが見つかりません: {source['file']}")

                                st.markdown("---")

                                # 元のPDFページを表示
                                pdf_path = Path("data/uploaded_pdfs") / source['file']
                                if pdf_path.exists():
                                    try:
                                        import pdfplumber
                                        from PIL import Image
                                        import io

                                        with pdfplumber.open(str(pdf_path)) as pdf:
                                            if source['page'] <= len(pdf.pages):
                                                page = pdf.pages[source['page'] - 1]
                                                # ページを画像に変換
                                                page_img = page.to_image(resolution=150)
                                                st.image(page_img.original, use_container_width=True)
                                            else:
                                                st.warning(f"ページ {source['page']} が見つかりません")
                                    except Exception as e:
                                        st.error(f"PDFページの表示に失敗しました: {e}")
                                else:
                                    st.warning(f"PDF not found: {pdf_path}")
                    else:
                        st.info("参照元が見つかりませんでした")

                    # チャット履歴にアシスタントの回答を追加
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result_data['answer']
                    })

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                logging.error(f"Error during query: {e}", exc_info=True)

        else:
            st.warning("質問を入力してください")

    # チャット履歴表示
    st.markdown("---")
    st.subheader("💬 チャット履歴")
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])


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
