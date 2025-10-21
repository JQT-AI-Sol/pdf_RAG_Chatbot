"""
簡易ベクトル検索テスト
"""
import yaml
from dotenv import load_dotenv
from src.vector_store import VectorStore
from src.text_embedder import TextEmbedder

load_dotenv()

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

vector_store = VectorStore(config)
embedder = TextEmbedder(config)

question = "週所定労働時間を5時間延長した場合の助成額"

print(f"質問: {question}\n")

embedding = embedder.embed_text(question)
results = vector_store.search(
    query_embedding=embedding,
    top_k=3,
    category="キャリアアップ助成金",
    search_type='both'
)

print(f"テキストチャンク: {len(results['text'])}件")
for i, chunk in enumerate(results['text'], 1):
    print(f"\n[{i}] {chunk.get('source_file', 'N/A')} - Page {chunk.get('page_number', 'N/A')}")
    print(f"    Content ({len(chunk.get('content', ''))}文字): {chunk.get('content', '')[:200]}...")

print(f"\n画像: {len(results['images'])}件")
for i, img in enumerate(results['images'], 1):
    print(f"\n[{i}] {img.get('source_file', 'N/A')} - Page {img.get('page_number', 'N/A')}")
    print(f"    Type: {img.get('content_type', 'N/A')}")
    print(f"    Description ({len(img.get('description', ''))}文字): {img.get('description', '')[:200]}...")
