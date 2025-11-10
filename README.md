---
title: JQIT RAG System
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# PDF RAGシステム（PoC版）

技術文書・マニュアルのPDFから情報を検索・抽出できるRAGシステムのPoC実装です。表やグラフを含むPDFに対応し、カテゴリー機能で検索範囲を限定できます。

**🚀 Hugging Face Space**: [https://huggingface.co/spaces/ACRMiyamoto/jqit-rag-system](https://huggingface.co/spaces/ACRMiyamoto/jqit-rag-system)

## 主な機能

- **マルチフォーマット対応**: PDF、Word、Excel、PowerPoint、テキストファイルをカテゴリー付きでアップロード
- **ハイブリッド検索**: テキストと画像（表・グラフ）の両方から検索
- **Vision AI解析**: GPT-4oまたはGemini Pro Visionで表・グラフを解析
- **カテゴリーフィルタリング**: ドキュメントカテゴリーで検索範囲を限定
- **質問応答**: 自然言語での質問に対して、参照元と画像付きで回答

## 技術スタック

- **UI**: Streamlit
- **RAGフレームワーク**: LangChain
- **AI/ML**: OpenAI API (GPT-4o, text-embedding-3-small)、Google Gemini API (Gemini 1.5 Pro, Gemini 1.5 Flash)
- **ベクトルDB**: ChromaDB
- **ドキュメント処理**: pdfplumber、python-docx、openpyxl、python-pptx、pdf2image
- **言語**: Python 3.10+

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example` をコピーして `.env` を作成し、OpenAI APIキーを設定：

```bash
cp .env.example .env
```

`.env` ファイルを編集：

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 設定ファイルの確認

`config.yaml` でパラメータを調整できます：

- モデルの選択（OpenAI GPT-4o、Gemini 1.5 Pro/Flash）
- チャンクサイズ、検索パラメータ
- Vision解析のプロンプト
- カテゴリー設定

## デプロイ

### Hugging Face Spaceへのデプロイ

1. **リモートリポジトリの追加**

```bash
git remote add hf https://<USERNAME>:<TOKEN>@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system
```

2. **コードのプッシュ**

```bash
git push hf master:main
```

詳細な手順は `CLAUDE.md` を参照してください。

3. **環境変数の設定**

Hugging Face Spaceの管理画面（Settings）で以下を設定：

- `SUPABASE_URL`: Supabaseプロジェクトの接続URL
- `SUPABASE_KEY`: Supabaseのサービスロールキー
- `OPENAI_API_KEY`: OpenAI APIキー
- `GOOGLE_API_KEY`: Google Gemini APIキー
- `COHERE_API_KEY`: Cohere Rerank APIキー（オプション）
- `LANGFUSE_PUBLIC_KEY`: Langfuse公開キー（オプション）
- `LANGFUSE_SECRET_KEY`: Langfuseシークレットキー（オプション）

## 使い方

### アプリケーションの起動

```bash
streamlit run app.py
```

### 基本的な使い方

1. **ドキュメントアップロード**
   - サイドバーからファイルを選択（PDF、Word、Excel、PowerPoint、テキスト）
   - カテゴリー名を入力（例：「製品マニュアル」「技術仕様書」）
   - 「インデックス作成」ボタンをクリック

2. **質問応答**
   - カテゴリードロップダウンから対象カテゴリーを選択
   - 質問を入力して送信
   - 回答、参照元、関連画像が表示されます

### カテゴリー機能

- **カテゴリーの追加**: ドキュメントアップロード時に新しいカテゴリー名を入力すると自動登録
- **カテゴリーの選択**: 質問時にドロップダウンから選択（カテゴリー変更時、チャット履歴は自動的にリセットされます）
- **全カテゴリー検索**: 「全カテゴリー」を選択すると全ドキュメントが対象

## プロジェクト構造

```
PoC_chatbot/
├── app.py                          # Streamlitメインアプリ
├── requirements.txt                # 依存パッケージ
├── .env                            # 環境変数（gitignore対象）
├── .env.example                    # 環境変数テンプレート
├── config.yaml                     # 設定ファイル
├── README.md                       # 本ファイル
├── requirements_definition.md      # 要件定義書
│
├── src/                            # ソースコード
│   ├── pdf_processor.py            # PDF処理
│   ├── text_embedder.py            # エンベディング
│   ├── vision_analyzer.py          # Vision AI
│   ├── vector_store.py             # ChromaDB操作
│   ├── rag_engine.py               # RAGエンジン
│   ├── category_manager.py         # カテゴリー管理
│   ├── prompt_templates.py         # プロンプト
│   └── utils.py                    # ユーティリティ
│
├── data/                           # データ保存用
│   ├── uploaded_pdfs/              # アップロードPDF
│   ├── extracted_images/           # 抽出画像
│   ├── chroma_db/                  # ベクトルDB
│   └── categories.json             # カテゴリー一覧
│
└── logs/                           # ログファイル
```

## 開発状況

現在PoC（Proof of Concept）として、基本機能は実装完了しています。

### 実装済み機能
- ✅ マルチフォーマットドキュメント対応（PDF、Word、Excel、PowerPoint、テキスト）
- ✅ Office文書のPDF変換と画像抽出
- ✅ ハイブリッド検索（テキスト + 画像）
- ✅ Vision AI統合（OpenAI GPT-4o、Google Gemini）
- ✅ ベクトルDB（ChromaDB）によるセマンティック検索
- ✅ カテゴリー管理とフィルタリング
- ✅ StreamlitによるWebUI
- ✅ チャット履歴管理
- ✅ 参照元表示（ページプレビュー付き）

### 継続的改善中
- 🔄 検索精度の向上
- 🔄 レスポンス時間の最適化
- 🔄 UIUXの改善

## トラブルシューティング

### よくある問題

**Q: OpenAI APIのレート制限エラーが発生する**
A: `config.yaml` で処理するファイル数を減らすか、APIキーのレート制限を確認してください。

**Q: 画像が表示されない**
A: `data/extracted_images/` ディレクトリの権限を確認してください。

**Q: メモリ不足エラーが発生する**
A: `config.yaml` の `chunk_size` を小さくするか、処理するPDFのサイズを減らしてください。

## ライセンス

このプロジェクトはPoC（概念実証）目的で作成されています。

## 参考資料

- [要件定義書](requirements_definition.md)
- [LangChain公式ドキュメント](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [ChromaDB Documentation](https://docs.trychroma.com/)
