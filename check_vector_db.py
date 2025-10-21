"""
ベクトルDBの内容を確認するスクリプト
"""
import chromadb
from pathlib import Path

# ChromaDBクライアント
client = chromadb.PersistentClient(path="./data/chroma_db")

# コレクション取得
text_collection = client.get_collection(name="pdf_text_chunks")
image_collection = client.get_collection(name="pdf_image_contents")

print("=== Text Collection ===")
text_count = text_collection.count()
print(f"Total documents: {text_count}")

if text_count > 0:
    # 予防接種に関連するドキュメントを検索
    results = text_collection.get(limit=10)
    print(f"\nFirst 10 documents:")
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        source = metadata.get('source_file', 'Unknown')
        page = metadata.get('page_number', '?')
        print(f"\n{i+1}. [{source} - Page {page}]")
        print(f"   Content: {doc[:200]}...")

print("\n\n=== Image Collection ===")
image_count = image_collection.count()
print(f"Total images: {image_count}")

if image_count > 0:
    # 画像解析結果を取得
    results = image_collection.get(limit=10)
    print(f"\nFirst 10 image analyses:")
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        source = metadata.get('source_file', 'Unknown')
        page = metadata.get('page_number', '?')
        content_type = metadata.get('content_type', '?')
        print(f"\n{i+1}. [{source} - Page {page} - {content_type}]")
        print(f"   Description: {doc[:300]}...")
