# PDF RAGシステム（PoC版）

技術文書・マニュアルのPDFから情報を検索・抽出できるRAGシステムのPoC実装です。表やグラフを含むPDFに対応し、カテゴリー機能で検索範囲を限定できます。

## 主な機能

- **PDFアップロード**: 複数のPDFファイルをカテゴリー付きでアップロード
- **ハイブリッド検索**: テキストと画像（表・グラフ）の両方から検索
- **Vision AI解析**: GPT-4o（将来的にGPT-5）で表・グラフを解析
- **カテゴリーフィルタリング**: ドキュメントカテゴリーで検索範囲を限定
- **質問応答**: 自然言語での質問に対して、参照元と画像付きで回答

## 技術スタック

- **UI**: Streamlit
- **RAGフレームワーク**: LangChain
- **AI/ML**: OpenAI API (GPT-4o, text-embedding-3-small)
- **ベクトルDB**: ChromaDB
- **PDF処理**: pdfplumber
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

- モデルの選択（GPT-4o / GPT-5）
- チャンクサイズ、検索パラメータ
- Vision解析のプロンプト
- カテゴリー設定

## 使い方

### アプリケーションの起動

```bash
streamlit run app.py
```

### 基本的な使い方

1. **PDFアップロード**
   - サイドバーからPDFファイルを選択
   - カテゴリー名を入力（例：「製品マニュアル」「技術仕様書」）
   - 「インデックス作成」ボタンをクリック

2. **質問応答**
   - カテゴリードロップダウンから対象カテゴリーを選択
   - 質問を入力して送信
   - 回答、参照元、関連画像が表示されます

### カテゴリー機能

- **カテゴリーの追加**: PDFアップロード時に新しいカテゴリー名を入力すると自動登録
- **カテゴリーの選択**: 質問時にドロップダウンから選択
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

現在PoC（Proof of Concept）フェーズです。

### 実装済み
- ✅ プロジェクト構造
- ✅ 設定ファイル
- ✅ 要件定義

### 実装予定（Phase順）
- [ ] Phase 1: 基本インフラ
- [ ] Phase 2: PDF処理
- [ ] Phase 3: Vision AI統合
- [ ] Phase 4: エンベディング・ベクトルDB
- [ ] Phase 5: RAGエンジン
- [ ] Phase 6: UI/UX改善
- [ ] Phase 7: テスト・調整

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
