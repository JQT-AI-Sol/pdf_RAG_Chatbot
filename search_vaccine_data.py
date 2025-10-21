"""
予防接種関連のデータを検索
"""
import chromadb

# ChromaDBクライアント
client = chromadb.PersistentClient(path="./data/chroma_db")

# コレクション取得
image_collection = client.get_collection(name="pdf_image_contents")

# 全データを取得
all_data = image_collection.get()

print(f"Total images in DB: {len(all_data['ids'])}")
print("\n=== Searching for 予防接種 data ===\n")

found_count = 0
for i, (doc_id, metadata, document) in enumerate(zip(all_data['ids'], all_data['metadatas'], all_data['documents'])):
    source = metadata.get('source_file', '')

    if '予防接種' in source or 'full_page' in metadata.get('content_type', ''):
        found_count += 1
        print(f"{found_count}. ID: {doc_id}")
        print(f"   Source: {source}")
        print(f"   Page: {metadata.get('page_number', '?')}")
        print(f"   Type: {metadata.get('content_type', '?')}")
        print(f"   Description (first 300 chars):")
        print(f"   {document[:300]}...")
        print()

if found_count == 0:
    print("❌ 予防接種またはfull_pageのデータが見つかりませんでした")
else:
    print(f"✓ {found_count}件のデータが見つかりました")
