# Claude Code による開発メモ

このファイルは、Claude Codeによる開発作業の手順や設定を記録するためのものです。

## Hugging Face Spaceへのデプロイ手順

### 初回セットアップ

#### 1. Hugging Face アクセストークンの取得

1. [Hugging Face](https://huggingface.co/) にログイン
2. 右上のアイコン → **Settings** → **Access Tokens**
3. 「New token」をクリック
4. **Name**: 任意の名前（例: `streamlit-app`）
5. **Role**: **Write**（書き込み権限）を選択
6. 「Create token」をクリック
7. **トークンをコピー**（一度だけ表示されます）

#### 2. リモートリポジトリの追加

```bash
# Hugging Face SpaceをGitリモートとして追加
git remote add hf https://<USERNAME>:<TOKEN>@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system
```

**例**（トークンは `hf_xxxxx` の形式）:
```bash
git remote add hf https://ACRMiyamoto:hf_YOUR_TOKEN_HERE@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system
```

#### 3. リモートの確認

```bash
git remote -v
```

**出力例**:
```
origin	https://github.com/JQT-AI-Sol/pdf_RAG_Chatbot.git (fetch)
origin	https://github.com/JQT-AI-Sol/pdf_RAG_Chatbot.git (push)
hf	https://ACRMiyamoto:hf_xxxxx@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system (fetch)
hf	https://ACRMiyamoto:hf_xxxxx@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system (push)
```

---

### 通常のデプロイ（更新時）

#### 方法1: 通常のプッシュ（推奨）

```bash
# 変更をコミット
git add .
git commit -m "Update: 変更内容の説明"

# GitHubにプッシュ（バックアップ）
git push origin master

# Hugging Face Spaceにプッシュ
git push hf master:main
```

#### 方法2: 大きなファイルがある場合

Hugging Face Spaceは10MB以上のファイルをプッシュできません。大きなPDFファイルなどが含まれる場合は、クリーンなブランチを作成してプッシュします。

```bash
# 新しいクリーンなブランチを作成（履歴なし）
git checkout --orphan clean-branch

# .gitignoreを尊重してファイルをステージング
git add -A

# コミット
git commit -m "Clean deployment for Hugging Face Space"

# Hugging Face Spaceにプッシュ（強制）
git push hf clean-branch:main -f

# 元のブランチに戻る
git checkout master

# クリーンブランチを削除（オプション）
git branch -D clean-branch
```

---

### トラブルシューティング

#### エラー: `Password authentication in git is no longer supported`

**原因**: パスワード認証が使えなくなりました。アクセストークンを使う必要があります。

**解決策**:
```bash
# リモートURLをトークン付きに更新
git remote set-url hf https://<USERNAME>:<TOKEN>@huggingface.co/spaces/ACRMiyamoto/jqit-rag-system
```

#### エラー: `Your push was rejected because it contains files larger than 10 MiB`

**原因**: 10MB以上のファイルが含まれています。

**解決策**:
1. `.gitignore`に大きなファイルを追加
2. 「方法2: 大きなファイルがある場合」の手順を使用

`.gitignore`の例:
```
# 大きなPDFファイルを除外
static/pdfs/
data/uploaded_pdfs/
```

#### エラー: `empty or missing yaml metadata in repo card`

**原因**: README.mdにHugging Face Space用のメタデータがありません。

**解決策**: README.mdの先頭に以下を追加:
```yaml
---
title: JQIT RAG System
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---
```

---

### 環境変数の設定

Hugging Face Spaceでアプリを動作させるには、環境変数の設定が必要です。

1. [Hugging Face Space管理画面](https://huggingface.co/spaces/ACRMiyamoto/jqit-rag-system) を開く
2. **Settings** タブをクリック
3. **Repository secrets** セクションで「New secret」をクリック
4. 以下の環境変数を追加:

#### 必須の環境変数

| 変数名 | 説明 | 取得方法 |
|--------|------|----------|
| `SUPABASE_URL` | SupabaseプロジェクトURL | Supabaseダッシュボード → Settings → API |
| `SUPABASE_KEY` | Supabaseサービスロールキー | Supabaseダッシュボード → Settings → API → service_role |
| `OPENAI_API_KEY` | OpenAI APIキー | OpenAI Platform → API Keys |
| `GOOGLE_API_KEY` | Google Gemini APIキー | Google AI Studio → Get API Key |

#### オプションの環境変数

| 変数名 | 説明 | 取得方法 |
|--------|------|----------|
| `COHERE_API_KEY` | Cohere Rerank APIキー | Cohere Dashboard → API Keys |
| `LANGFUSE_PUBLIC_KEY` | Langfuse公開キー | Langfuse → Settings → API Keys |
| `LANGFUSE_SECRET_KEY` | Langfuseシークレットキー | Langfuse → Settings → API Keys |

---

## PowerPoint ハイライト機能の実装履歴

### 2025-11-10: PowerPoint変換PDFのハイライト対応

**問題**:
- PowerPoint→PDF変換時にテキストが画像化される
- `pdfplumber.extract_text()`で取得できるのは171文字のみ
- データベースには536文字のVision API解析済みテキストが存在
- ハイライトが表示されない

**解決策**:
1. `VectorStore.get_chunks_by_page()`メソッドを追加
2. `create_pdf_annotations_hybrid()`をDBチャンクベースに変更
3. `app.py`のハイライト呼び出しに`vector_store`と`source_file`を追加
4. `config.cloud.yaml`でピンポイントハイライト設定（threshold 0.7, max_final 3）

**コミット**:
- `171172e`: feat: Enable PowerPoint highlighting using database chunks
- `cb5430e`: fix: Remove chunk_index column reference in get_chunks_by_page
- `e05a9cc`: chore: Force redeployment to apply database fix

**修正ファイル**:
- `src/vector_store.py` (lines 973-998)
- `src/pdf_page_renderer.py` (lines 487-575)
- `app.py` (lines 1043-1051, 1346-1354)
- `config.cloud.yaml` (lines 232-237)

---

## 開発のベストプラクティス

### コミットメッセージ

以下の形式を使用:

```
<type>: <subject>

<body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Type の種類**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `style`: コードの意味に影響しない変更（空白、フォーマットなど）
- `refactor`: バグ修正でも機能追加でもないコード変更
- `test`: テストの追加や修正
- `chore`: ビルドプロセスやツールの変更

### ブランチ戦略

- `master`: メインブランチ（GitHubとHugging Face両方にプッシュ）
- `clean-branch`: 大きなファイルを除外したクリーンなブランチ（Hugging Face専用、一時的）

---

## 参考リンク

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Supabase Documentation](https://supabase.com/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
