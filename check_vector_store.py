"""
ベクトルストアに保存されているデータを確認するスクリプト
"""

import chromadb
from pathlib import Path

# ChromaDBクライアント初期化
client = chromadb.PersistentClient(path="./data/chroma_db")

# テキストコレクション
text_collection = client.get_collection("pdf_text_chunks")
print("=" * 80)
print("【テキストチャンク】")
print("=" * 80)
print(f"Total documents: {text_collection.count()}\n")

# 最初の3件のテキストチャンクを表示
results = text_collection.get(limit=3, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"\n--- Chunk {i} ---")
    print(f"File: {meta.get('source_file')}, Page: {meta.get('page_number')}")
    print(f"Content:\n{doc[:300]}...")  # 最初の300文字
    print()

# 画像コレクション
print("\n" + "=" * 80)
print("【画像コンテンツ（Vision AI解析結果）】")
print("=" * 80)
image_collection = client.get_collection("pdf_image_contents")
print(f"Total images: {image_collection.count()}\n")

# 全ての画像解析結果を表示
results = image_collection.get(include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"\n--- Image {i} ---")
    print(f"File: {meta.get('source_file')}, Page: {meta.get('page_number')}")
    print(f"Type: {meta.get('content_type')}")
    print(f"Image Path: {meta.get('image_path')}")
    print(f"Vision AI Description:\n{doc}")
    print()
