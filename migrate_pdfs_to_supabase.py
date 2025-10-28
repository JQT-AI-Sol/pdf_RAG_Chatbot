"""
既存のPDFファイルをSupabase Storageに移行するスクリプト
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from src.vector_store import VectorStore

# 環境変数を読み込み
load_dotenv()

print("=== PDFファイルのSupabase Storage移行スクリプト ===\n")

# config.yaml読み込み
print("--- config.yaml読み込み ---")
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

provider = config.get('vector_store', {}).get('provider', 'chromadb')
print(f"Vector Store Provider: {provider}")

if provider != 'supabase':
    print(f"WARNING: config.yamlのproviderが '{provider}' に設定されています")
    print("   Supabaseに移行するには 'supabase' に変更してください")
    exit(1)

# VectorStore初期化
print("\n--- VectorStore初期化 ---")
try:
    vector_store = VectorStore(config)
    print("OK: VectorStore初期化成功")
except Exception as e:
    print(f"ERROR: VectorStore初期化失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 登録済みPDF一覧を取得
print("\n--- 登録済みPDF一覧を取得 ---")
try:
    pdfs = vector_store.get_registered_pdfs()
    print(f"OK: 登録済みPDF数: {len(pdfs)}")
except Exception as e:
    print(f"ERROR: PDF一覧取得失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ローカルPDFディレクトリ
pdf_dir = Path("data/uploaded_pdfs")

if not pdf_dir.exists():
    print(f"ERROR: PDFディレクトリが存在しません: {pdf_dir}")
    exit(1)

# 各PDFファイルを処理
print("\n--- PDFファイルの移行開始 ---")
success_count = 0
skip_count = 0
error_count = 0

for pdf_info in pdfs:
    filename = pdf_info['source_file']
    category = pdf_info['category']
    local_path = pdf_dir / filename

    print(f"\n処理中: {filename}")
    print(f"  カテゴリー: {category}")

    # ローカルファイルの存在確認
    if not local_path.exists():
        print(f"  Error: Local file not found: {local_path}")
        error_count += 1
        continue

    # 既にstorage_pathが設定されているか確認
    try:
        response = vector_store.client.table(vector_store.pdf_table)\
            .select('storage_path')\
            .eq('filename', filename)\
            .execute()

        if response.data and response.data[0].get('storage_path'):
            print(f"  Skip: Already in Storage: {response.data[0]['storage_path']}")
            skip_count += 1
            continue
    except Exception as e:
        # storage_path列が存在しない場合はスキップして続行
        if 'does not exist' in str(e):
            print(f"  Warning: storage_path column not found, skipping check")
        else:
            print(f"  Warning: storage_path check failed: {e}")

    # Supabase Storageにアップロード
    try:
        storage_path = vector_store.upload_pdf_to_storage(
            str(local_path), filename, category
        )
        print(f"  Success: Uploaded to Storage: {storage_path}")

        # registered_pdfsテーブルを更新
        try:
            vector_store.register_pdf(filename, category, storage_path)
            print(f"  Success: Database updated")
        except Exception as db_err:
            # storage_path列が存在しない場合でもアップロードは成功
            if 'does not exist' in str(db_err):
                print(f"  Warning: storage_path column not in DB, but file uploaded")
                print(f"  Info: Run add_storage_path_column.sql in Supabase SQL Editor")
            else:
                raise db_err

        success_count += 1

    except Exception as e:
        print(f"  Error: Upload failed: {e}")
        import traceback
        traceback.print_exc()
        error_count += 1

# 結果サマリー
print("\n" + "="*50)
print("=== Migration Complete ===")
print(f"Success: {success_count} files")
print(f"Skipped: {skip_count} files")
print(f"Errors: {error_count} files")
print(f"Total: {len(pdfs)} files")
print("="*50)

if error_count > 0:
    print("\nWarning: Some files had errors. Check the log above.")
    exit(1)
else:
    print("\nSuccess: All PDF files migrated to Supabase Storage!")
