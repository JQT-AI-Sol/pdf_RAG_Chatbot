# Excel検索問題のデバッグ手順

## 問題の概要
- **現象**: Excelファイルをアップロードしてカテゴリー「マッチングシステム」で検索すると0件の結果が返る
- **確認済み**:
  - ✅ データはDBに存在（100チャンク）
  - ✅ エンベディングは有効（数値データとして保存）
  - ✅ 直接SQLでの類似度検索は成功（スコア: 1.0, 0.84, 0.74, 0.71, 0.70）
  - ✅ RPCfunction自体は機能（PDFでは動作）
- **未解決**: なぜRPC呼び出しが0件を返すのか

## 追加した診断ツール

### 1. 詳細ロギング（src/vector_store.py）
以下の情報をログに出力するよう強化：
- クエリエンベディングの型、次元、最初の10要素
- RPCパラメータの詳細
- レスポンスオブジェクトの構造

### 2. SQL診断スクリプト

#### `check_rpc_function.sql`
RPC関数定義を確認:
```sql
-- Supabase SQL Editorで実行
SELECT proname, pg_get_functiondef(oid) FROM pg_proc WHERE proname = 'match_text_chunks';
```

#### `check_category_encoding.sql`
カテゴリーのエンコーディング・空白文字の問題を確認:
```sql
-- カテゴリー値の正確な確認
SELECT DISTINCT category, length(category), encode(category::bytea, 'hex')
FROM pdf_text_chunks WHERE category IS NOT NULL;
```

#### `test_query_embedding.sql`
実際のクエリエンベディングをテスト（後述の手順で使用）

## デバッグ実行手順

### ステップ1: クエリエンベディングのキャプチャ

1. **アプリを起動**:
   ```bash
   streamlit run app.py
   ```

2. **カテゴリー「マッチングシステム」を選択**

3. **質問を入力**:
   ```
   マッチングシステムの構成図はどうなってる？
   ```

4. **ログを確認**し、以下の情報をコピー:
   ```
   🔍 DEBUG: First 10 elements of query embedding: [-0.0123, 0.0456, ...]
   🔍 DEBUG: RPC parameters: match_threshold=0.5, match_count=10, filter_category=マッチングシステム
   ```

   **重要**: 完全なエンベディング配列（3072要素すべて）が必要です。
   ログには最初の10要素しか表示されませんが、内部的には全配列が渡されています。

### ステップ2: RPC関数定義の確認

1. **Supabase SQL Editorを開く**

2. **`check_rpc_function.sql`を実行**:
   ```sql
   SELECT proname as function_name, pg_get_functiondef(oid) as function_definition
   FROM pg_proc WHERE proname = 'match_text_chunks';
   ```

3. **関数定義を確認**:
   - パラメータ名: `query_embedding`, `match_threshold`, `match_count`, `filter_category`
   - 類似度計算: `embedding <=> query_embedding`
   - 閾値チェック: `< match_threshold` (距離) or `> match_threshold` (類似度)
   - カテゴリーフィルター: `WHERE category = filter_category` or `WHERE category IS NULL OR category = filter_category`

### ステップ3: カテゴリーエンコーディングの確認

1. **`check_category_encoding.sql`を実行**:
   ```sql
   SELECT DISTINCT category, length(category), encode(category::bytea, 'hex')
   FROM pdf_text_chunks WHERE category IS NOT NULL;
   ```

2. **確認項目**:
   - 「マッチングシステム」の文字列長は正確か？
   - 前後に空白文字が含まれていないか？
   - エンコーディングに異常はないか？（UTF-8が正しい）

### ステップ4: 直接SQLテスト（オプション）

もしクエリエンベディングの完全な配列を取得できた場合:

1. **`test_query_embedding.sql`を開く**

2. **`<QUERY_EMBEDDING>`を実際の値に置換**:
   ```sql
   SELECT id, source_file,
          1 - (embedding <=> '[-0.0123,0.0456,...]'::vector) as similarity
   FROM pdf_text_chunks
   WHERE category = 'マッチングシステム'
   ORDER BY embedding <=> '[-0.0123,0.0456,...]'::vector
   LIMIT 10;
   ```

3. **結果を確認**:
   - 類似度スコアが出力されるか？
   - 閾値0.5を超える結果はあるか？

## 期待される診断結果

### シナリオA: RPC関数定義に問題がある
- **兆候**: 関数定義で`filter_category`の使い方が間違っている
- **解決策**: RPC関数を修正

### シナリオB: カテゴリーエンコーディングに問題がある
- **兆候**: カテゴリー文字列に空白や不可視文字が含まれる
- **解決策**: カテゴリー値をトリムしてから保存・検索

### シナリオC: 閾値の解釈が間違っている
- **兆候**: RPC関数が類似度を距離として扱っている（または逆）
- **解決策**: 閾値チェックを修正（`< threshold` → `> threshold` または逆）

### シナリオD: エンベディング型変換に問題がある
- **兆候**: Supabase SDKがList[float]→vectorの変換に失敗
- **解決策**: 明示的に文字列形式に変換してから送信

## 次のステップ

1. **ステップ1〜3を実行**してログと診断結果を収集
2. **結果を確認**し、上記のシナリオのどれに該当するか判断
3. **該当するシナリオに応じて修正**を実施

## 補足情報

### 正常に動作しているPDFの例
- ファイル: `キャリアアップ助成金.pdf`
- カテゴリー: `キャリアアップ助成金`
- 検索結果: 正常に返る

### 動作していないExcelの例
- ファイル: `LLMマッチングシステム開発PJ_ver1.03.xlsx`
- カテゴリー: `マッチングシステム`
- 検索結果: 0件（データは存在するのに）

### 技術詳細
- エンベディングモデル: `text-embedding-3-large` (3072次元)
- ベクトルDB: Supabase (pgvector)
- 類似度メトリック: コサイン距離 (`<=>` operator)
- 類似度閾値: 0.5（デフォルト）
