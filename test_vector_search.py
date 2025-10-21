"""
ベクトル検索のテスト - キャリアアップ助成金の質問で検索
"""
import os
import yaml
from dotenv import load_dotenv
from src.vector_store import VectorStore
from src.text_embedder import TextEmbedder

# 環境変数読み込み
load_dotenv()

# 設定読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# VectorStore初期化
vector_store = VectorStore(config)

# TextEmbedder初期化
embedder = TextEmbedder(config)

# テスト質問
question = """
常時雇用する労働者が25人であるサービス業の事業主（小規模企業に該当）が、「短時間労働者労働時間延長支援コース」を利用します。

対象労働者Aさんの週所定労働時間を3時間30分延長し、基本給を10%増額しました。
対象労働者Bさんの週所定労働時間を5時間延長しました（基本給の増額はなし）。

この事業主が1年目に受け取れる助成額は、AさんとBさんそれぞれいくらですか？
"""

print("=" * 80)
print("質問:", question)
print("=" * 80)

# カテゴリーフィルタ
category = "キャリアアップ助成金"

# エンベディング取得
print("\n[1] 質問のエンベディング取得中...")
question_embedding = embedder.embed_text(question)
print(f"   エンベディング次元数: {len(question_embedding)}")

# ベクトル検索実行
print("\n[2] ベクトル検索実行中...")
search_results = vector_store.search(
    query_embedding=question_embedding,
    top_k=5,
    category=category,
    search_type='both'
)

# テキストチャンク結果
text_chunks = search_results.get('text', [])
print(f"\n   取得されたテキストチャンク数: {len(text_chunks)}")
for i, chunk in enumerate(text_chunks, 1):
    print(f"\n   --- チャンク {i} ---")
    print(f"   RAW DATA: {chunk}")  # 生データを表示
    print(f"   Source: {chunk.get('source_file', 'N/A')}")
    print(f"   Page: {chunk.get('page_number', 'N/A')}")
    print(f"   Content Type: {chunk.get('content_type', 'N/A')}")
    print(f"   Content Length: {len(chunk.get('content', ''))} chars")
    print(f"   Content Preview (最初500文字):")
    content_preview = chunk.get('content', '')[:500]
    print(f"   {content_preview}")
    if len(chunk.get('content', '')) > 500:
        print(f"   ... (残り {len(chunk.get('content', '')) - 500} 文字)")

# 画像コンテンツ結果
image_data = search_results.get('images', [])
print(f"\n[3] 取得された画像コンテンツ数: {len(image_data)}")
for i, img in enumerate(image_data, 1):
    print(f"\n   --- 画像 {i} ---")
    print(f"   RAW DATA: {img}")  # 生データを表示
    print(f"   Source: {img.get('source_file', 'N/A')}")
    print(f"   Page: {img.get('page_number', 'N/A')}")
    print(f"   Content Type: {img.get('content_type', 'N/A')}")
    print(f"   Description Length: {len(img.get('description', ''))} chars")
    print(f"   Image Path: {img.get('path', 'N/A')}")
    print(f"   Description Preview (最初500文字):")
    desc_preview = img.get('description', '')[:500]
    print(f"   {desc_preview}")
    if len(img.get('description', '')) > 500:
        print(f"   ... (残り {len(img.get('description', '')) - 500} 文字)")

print("\n" + "=" * 80)
print("検索完了")
print("=" * 80)
