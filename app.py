"""
Streamlit Application for PDF RAG System
"""

import streamlit as st
import logging
from pathlib import Path

# src モジュールのインポート
from src.utils import load_config, load_environment, ensure_directories, setup_logging
from src.category_manager import CategoryManager
from src.pdf_processor import PDFProcessor
from src.text_embedder import TextEmbedder
from src.vision_analyzer import VisionAnalyzer
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


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

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"処理中: {uploaded_file.name} (1/5) - PDF保存中...")

            # 1. PDFを保存
            pdf_path = Path("data/uploaded_pdfs") / uploaded_file.name
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. テキスト・画像抽出
            status_text.text(f"処理中: {uploaded_file.name} (2/5) - テキスト・画像抽出中...")
            pdf_result = st.session_state.pdf_processor.process_pdf(str(pdf_path), category)

            # 3. テキストチャンクをエンベディング
            status_text.text(f"処理中: {uploaded_file.name} (3/5) - テキストエンベディング中...")
            if pdf_result['text_chunks']:
                text_embeddings = []
                for chunk in pdf_result['text_chunks']:
                    embedding = st.session_state.embedder.embed_text(chunk['text'])
                    text_embeddings.append(embedding)

                # ベクトルストアに追加
                st.session_state.vector_store.add_text_chunks(
                    pdf_result['text_chunks'],
                    text_embeddings
                )

            # 4. 画像をVision AIで解析
            status_text.text(f"処理中: {uploaded_file.name} (4/5) - 画像解析中...")
            if pdf_result['images']:
                for image_data in pdf_result['images']:
                    try:
                        # Vision AI解析（pdf_processorが設定したcontent_typeを使用）
                        actual_content_type = image_data.get('content_type', 'image')
                        analysis = st.session_state.vision_analyzer.analyze_image(
                            image_data['image_path'],
                            content_type=actual_content_type
                        )

                        # 解析結果をエンベディング
                        image_embedding = st.session_state.embedder.embed_text(
                            analysis['description']
                        )

                        # メタデータを統合
                        image_data.update({
                            'category': category,
                            'content_type': analysis.get('content_type', 'image'),
                            'description': analysis['description']
                        })

                        # ベクトルストアに追加
                        st.session_state.vector_store.add_image_content(
                            image_data,
                            image_embedding
                        )

                    except Exception as e:
                        logging.error(f"Error processing image: {e}")
                        continue

            status_text.text(f"処理中: {uploaded_file.name} (5/5) - 完了！")
            progress_bar.progress((i + 1) / len(uploaded_files))

        except Exception as e:
            st.sidebar.error(f"エラー ({uploaded_file.name}): {str(e)}")
            logging.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
            continue

    status_text.success("✅ すべてのPDFの処理が完了しました！")


def main_area():
    """メインエリアのUI"""
    st.title("📚 PDF RAG System")
    st.markdown("---")

    # カテゴリー選択
    categories = ["全カテゴリー"] + st.session_state.category_manager.get_all_categories()
    selected_category = st.selectbox(
        "🔍 検索対象カテゴリー",
        categories,
        help="質問する対象のカテゴリーを選択してください"
    )

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

                # 回答生成（ストリーミング反映待ち - 最大15分）
                st.markdown("### 💬 回答")
                with st.spinner("回答を生成中..."):
                    result_data = st.session_state.rag_engine.query(question, category_filter)

                # 回答表示
                st.markdown(result_data['answer'])

                # 参照元表示
                if result_data:
                    st.markdown("### 📄 参照元")
                    if result_data['sources']:
                        for idx, source in enumerate(result_data['sources'], 1):
                            with st.expander(f"参照 {idx}: {source['file']} (ページ {source['page']})"):
                                st.write(f"**カテゴリー**: {source['category']}")
                                st.write(f"**タイプ**: {source['type']}")

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
