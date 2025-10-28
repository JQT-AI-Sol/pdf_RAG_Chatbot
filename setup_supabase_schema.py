"""
Supabaseのテーブルスキーマをセットアップするスクリプト
"""
import os
import yaml
from dotenv import load_dotenv
from supabase import create_client, Client

# 環境変数を読み込み
load_dotenv()

print("=== Supabaseスキーマセットアップ ===\n")

# 環境変数確認
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL または SUPABASE_KEY が設定されていません")
    exit(1)

print(f"OK: SUPABASE_URL: {supabase_url}")

# Supabaseクライアント作成
try:
    client: Client = create_client(supabase_url, supabase_key)
    print("OK: Supabaseクライアント初期化成功\n")
except Exception as e:
    print(f"ERROR: Supabaseクライアント初期化失敗: {e}")
    exit(1)

# registered_pdfsテーブルにstorage_path列を追加
print("--- registered_pdfsテーブルにstorage_path列を追加 ---")

# SQLで直接実行できないため、既存データを確認
try:
    response = client.table('registered_pdfs').select('*').limit(1).execute()

    if response.data and len(response.data) > 0:
        columns = list(response.data[0].keys())
        print(f"現在の列: {', '.join(columns)}")

        if 'storage_path' in columns:
            print("✅ storage_path列は既に存在します")
        else:
            print("❌ storage_path列が存在しません")
            print("\n以下のSQLをSupabase SQL Editorで実行してください:")
            print("-" * 60)
            print("""
ALTER TABLE registered_pdfs
ADD COLUMN storage_path TEXT;

CREATE INDEX idx_registered_pdfs_storage_path
ON registered_pdfs(storage_path);
            """)
            print("-" * 60)
            print("\nSupabase Dashboard:")
            print(f"  {supabase_url.replace('.supabase.co', '.supabase.co/project/_/sql/new')}")
    else:
        print("⚠️  registered_pdfsテーブルにデータがありません")
        print("  テーブル構造を確認できませんでした")

except Exception as e:
    print(f"ERROR: テーブル確認失敗: {e}")
    print("\n以下のSQLをSupabase SQL Editorで実行してください:")
    print("-" * 60)
    print("""
ALTER TABLE registered_pdfs
ADD COLUMN IF NOT EXISTS storage_path TEXT;

CREATE INDEX IF NOT EXISTS idx_registered_pdfs_storage_path
ON registered_pdfs(storage_path);
    """)
    print("-" * 60)

print("\n=== セットアップ完了 ===")
