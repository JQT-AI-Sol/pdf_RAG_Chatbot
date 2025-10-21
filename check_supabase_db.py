"""
SupabaseのDB内容を確認
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

print("=== Supabase DB内容確認 ===\n")

# 接続
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')
client = create_client(supabase_url, supabase_key)

# 1. registered_pdfs テーブル
print("--- registered_pdfs テーブル ---")
response = client.table('registered_pdfs').select('*').execute()
print(f"登録PDF数: {len(response.data)}")
for row in response.data:
    print(f"  - {row['filename']} (category: {row['category']})")

# 2. pdf_image_contents テーブル
print("\n--- pdf_image_contents テーブル ---")
response = client.table('pdf_image_contents').select('*').execute()
print(f"画像データ数: {len(response.data)}\n")

for i, row in enumerate(response.data, 1):
    content = row.get('content', '')
    print(f"{i}. ID: {row['id']}")
    print(f"   Source: {row['source_file']}")
    print(f"   Page: {row['page_number']}")
    print(f"   Type: {row['content_type']}")
    print(f"   Content length: {len(content)} chars")
    print(f"   Embedding dimension: {len(row.get('embedding', []))}")

    # 最初の500文字を表示
    print(f"   Content preview:")
    preview = content[:500].replace('\n', ' ')
    print(f"   {preview}...")

    # JSON形式か平文かを判定（より詳細に）
    has_json_keyword = 'json' in content.lower()
    starts_with_brace = content.strip().startswith('{')
    has_code_block = '```json' in content.lower() or '```' in content[:100]

    if has_code_block or (has_json_keyword and starts_with_brace):
        print(f"   [WARNING] JSON形式の可能性あり (json_keyword={has_json_keyword}, code_block={has_code_block})")
    else:
        print(f"   [OK] 平文テキスト形式")
    print()

# 3. pdf_text_chunks テーブル
print("\n--- pdf_text_chunks テーブル ---")
response = client.table('pdf_text_chunks').select('*').execute()
print(f"テキストチャンク数: {len(response.data)}")

print("\n=== 確認完了 ===")
