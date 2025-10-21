"""
検索テスト
"""
import yaml
from src.text_embedder import TextEmbedder
from src.vector_store import VectorStore

# 設定読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初期化
embedder = TextEmbedder(config)
vector_store = VectorStore(config)

# 質問をエンベディング
question = "給付の請求方法と必要書類について教えて"
print(f"質問: {question}\n")

query_embedding = embedder.embed_query(question)

# 検索実行
results = vector_store.search(
    query_embedding=query_embedding,
    category="予防接種後健康被害救済制度",
    top_k=5,
    search_type="both"
)

print("=== テキスト検索結果 ===")
for i, result in enumerate(results.get("text", []), 1):
    metadata = result["metadata"]
    doc = result["document"]
    print(f"\n{i}. [{metadata.get('source_file', 'Unknown')} - Page {metadata.get('page_number', '?')}]")
    print(f"   Distance: {result.get('distance', 'N/A')}")
    print(f"   Content: {doc[:200]}...")

print("\n\n=== 画像検索結果 ===")
for i, result in enumerate(results.get("images", []), 1):
    metadata = result["metadata"]
    doc = result["document"]
    print(f"\n{i}. [{metadata.get('source_file', 'Unknown')} - Page {metadata.get('page_number', '?')} - {metadata.get('content_type', '?')}]")
    print(f"   Distance: {result.get('distance', 'N/A')}")
    print(f"   Description: {doc[:300]}...")
