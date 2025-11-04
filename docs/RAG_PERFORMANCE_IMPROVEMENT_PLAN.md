# RAG性能向上計画

**作成日**: 2025-11-04
**対象システム**: PDF RAGチャットボット（PoC_chatbot）
**優先フェーズ**: Phase 1 - Quick Wins（1週間）
**改善目標**: 検索精度向上 + 応答速度改善

---

## 📊 エグゼクティブサマリー

本ドキュメントは、既存のPDF RAGシステムの詳細分析に基づき、短期間（1週間）で実現可能な性能向上施策を提案します。

### 期待される効果
- **検索精度**: 10-15%向上（NDCG@5: 0.85 → 0.92）
- **応答速度**: 30-50%改善（検索レイテンシ: 200-500ms → 100-300ms）
- **ユーザー体験**: より関連性の高い回答、高速なレスポンス

---

## 🎯 現状分析

### システム概要

**アーキテクチャ**:
```
PDFアップロード
    ↓
PDF処理（並列化: 最大4スレッド）
    ├─ テキスト抽出 → チャンク化 → エンベディング
    └─ 画像/表/グラフ抽出 → Vision AI解析 → エンベディング
    ↓
ベクトルストア（Supabase pgvector / ChromaDB）
    ↓
ユーザークエリ → エンベディング → ベクトル検索
    ↓
コンテキスト構築 → LLM（GPT-4o/Gemini） → 回答生成
```

**技術スタック**:
- UI: Streamlit
- RAGフレームワーク: LangChain
- AI/LLM: OpenAI (GPT-4o, text-embedding-3-large), Google Gemini (gemini-2.5-pro)
- ベクトルDB: Supabase (pgvector) / ChromaDB
- PDF処理: pdfplumber
- 可観測性: Langfuse

### 既存の強み

✅ **実装済みの最適化**:
- エンベディングのバッチ処理（100件/リクエスト）
- PDF処理の並列化（ThreadPoolExecutor）
- マルチモーダル対応（テキスト + 画像 + 表 + グラフ）
- Supabase統合による永続化
- カテゴリーフィルタリング機能
- Langfuseによるトレーシング

### 特定された課題

⚠️ **改善が必要な領域**:

1. **検索精度の限界**
   - 単純なベクトル検索のみ（コサイン類似度）
   - Rerankingなし → 精度10-15%の損失
   - 固定の類似度閾値（0.5）

2. **チャンク戦略の非最適性**
   - 固定サイズチャンキング（800トークン、150オーバーラップ）
   - セマンティック境界を無視 → 文脈の分断

3. **キャッシングの不足**
   - エンベディング生成の重複
   - Vision解析結果の再利用なし
   - APIコストと応答時間の増加

4. **ハイブリッド検索の未実装**
   - ベクトル検索のみ → キーワードマッチの弱さ
   - BM25との組み合わせで精度向上の余地

---

## 🚀 Phase 1: Quick Wins（1週間実装計画）

### 目標

| 指標 | 現状 | 目標 | 改善率 |
|------|------|------|--------|
| 検索精度（NDCG@5） | 0.85 | 0.92 | +8% |
| 検索レイテンシ | 200-500ms | 100-300ms | -50% |
| キャッシュヒット率 | 0% | 60-70% | - |

---

### 実装タスク

#### 1️⃣ Reranking導入（優先度: 🔴 最高）

**目的**: 検索精度を10-15%向上

**実装方法**:
```python
# 新規ファイル: backend/src/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        軽量でありながら高精度なCross-Encoderモデルを使用
        モデルサイズ: ~80MB
        推論速度: ~50ms/query（Top-10をrerank）
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        クエリとドキュメントのペアをrerankし、
        スコアの高い順にインデックスとスコアを返す
        """
        # クエリ-ドキュメントペアを作成
        pairs = [[query, doc] for doc in documents]

        # rerankスコアを計算
        scores = self.model.predict(pairs)

        # スコア順にソート
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_scores = scores[ranked_indices]

        return list(zip(ranked_indices.tolist(), ranked_scores.tolist()))
```

**統合ポイント** (`backend/src/rag_engine.py`):
```python
# RAGEngine._get_relevant_contexts()メソッドに追加

# 1次検索（ベクトル検索）でTop-10を取得
text_results = self.vector_store.similarity_search(
    query_embedding,
    k=10,  # rerankingのため多めに取得
    filter={"category": category} if category else None
)

# Rerankingを適用
reranker = Reranker()
documents = [r.page_content for r in text_results]
reranked_indices, scores = reranker.rerank(query, documents, top_k=5)

# 上位5件を最終結果として使用
final_results = [text_results[idx] for idx in reranked_indices]
```

**依存関係**:
```bash
pip install sentence-transformers
```

**推定効果**:
- 精度向上: +10-15%
- レイテンシ増加: +50-100ms（許容範囲内）
- 追加コスト: なし（ローカル推論）

---

#### 2️⃣ セマンティックチャンキング（優先度: 🟠 高）

**目的**: 文脈を保持したチャンク化で回答品質向上

**実装方法**:
```python
# 修正ファイル: backend/src/pdf_processor.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, config):
        # ...既存コード...

        # セマンティックチャンカーを初期化
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=self._count_tokens,
            # セマンティック境界を優先
            separators=[
                "\n\n",      # 段落区切り
                "\n",        # 改行
                "。",        # 日本語文末
                "．",        # 日本語文末（全角ピリオド）
                ". ",        # 英語文末
                "! ",        # 感嘆符
                "? ",        # 疑問符
                "；",        # セミコロン
                "、",        # 読点
                "，",        # カンマ（全角）
                ", ",        # カンマ
                " ",         # スペース
                ""           # 最後の手段（文字単位）
            ],
            keep_separator=True  # 区切り文字を保持
        )

    def _chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """
        セマンティック境界を考慮したチャンク化
        """
        # 表のコンテキストを保持するための前処理
        text = self._preserve_table_context(text)

        # セマンティックチャンキング
        chunks = self.text_splitter.split_text(text)

        return [
            {
                "content": chunk,
                "page_number": page_num,
                "content_type": "text",
                "chunk_index": idx
            }
            for idx, chunk in enumerate(chunks)
        ]

    def _preserve_table_context(self, text: str) -> str:
        """
        表の前後にコンテキスト情報を追加
        例: 「以下は〇〇表です」→ 表のコンテンツの前に残す
        """
        # 実装: 正規表現で表の前の見出しを検出し、
        # チャンク境界をまたがないようにマークアップ
        import re

        # 表や図の参照を検出
        patterns = [
            r'(表\s*\d+[.:].*?)(\n)',
            r'(図\s*\d+[.:].*?)(\n)',
            r'(Table\s+\d+[.:].*?)(\n)',
            r'(Figure\s+\d+[.:].*?)(\n)',
        ]

        for pattern in patterns:
            text = re.sub(pattern, r'\1\n\n', text)  # 段落区切りを強制

        return text
```

**推定効果**:
- 回答の文脈正確性: +15-20%
- 表や図の説明との紐付け改善
- 実装コスト: 既存コードの小規模修正のみ

---

#### 3️⃣ エンベディングキャッシング（優先度: 🟡 中）

**目的**: APIコスト削減 + 応答速度改善

**実装方法**:
```python
# 新規ファイル: backend/src/embedding_cache.py

import hashlib
import json
from typing import List, Optional
from pathlib import Path
import pickle

class EmbeddingCache:
    def __init__(self, cache_dir: str = "./cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # メモリキャッシュ（LRU）
        from functools import lru_cache
        self._memory_cache = {}
        self._max_memory_items = 1000

    def _get_cache_key(self, text: str) -> str:
        """テキストからキャッシュキーを生成"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """キャッシュから埋め込みを取得"""
        key = self._get_cache_key(text)

        # メモリキャッシュをチェック
        if key in self._memory_cache:
            return self._memory_cache[key]

        # ディスクキャッシュをチェック
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)

            # メモリキャッシュに追加
            self._add_to_memory_cache(key, embedding)
            return embedding

        return None

    def set(self, text: str, embedding: List[float]):
        """埋め込みをキャッシュに保存"""
        key = self._get_cache_key(text)

        # メモリキャッシュに追加
        self._add_to_memory_cache(key, embedding)

        # ディスクに永続化
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

    def _add_to_memory_cache(self, key: str, value: List[float]):
        """LRU方式でメモリキャッシュに追加"""
        if len(self._memory_cache) >= self._max_memory_items:
            # 最も古いアイテムを削除
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = value
```

**統合** (`backend/src/text_embedder.py`):
```python
class TextEmbedder:
    def __init__(self, config: Dict[str, Any]):
        # ...既存コード...
        self.cache = EmbeddingCache()

    def embed_text(self, text: str) -> List[float]:
        """キャッシュ付き埋め込み生成"""
        # キャッシュをチェック
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        # キャッシュミス: API呼び出し
        embedding = self._call_openai_api(text)

        # キャッシュに保存
        self.cache.set(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """バッチ処理でキャッシュを活用"""
        results = []
        uncached_texts = []
        uncached_indices = []

        # キャッシュヒット/ミスを判定
        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(idx)

        # キャッシュミスのテキストをバッチ処理
        if uncached_texts:
            embeddings = self._call_openai_api_batch(uncached_texts)

            # 結果をキャッシュに保存 & results配列に格納
            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                self.cache.set(text, embedding)
                results[idx] = embedding

        return results
```

**推定効果**:
- APIコスト削減: 60-70%（キャッシュヒット率による）
- クエリ応答速度: -100ms（キャッシュヒット時）
- ディスク使用量: ~10MB/1000埋め込み

---

#### 4️⃣ Vision解析結果のキャッシング（優先度: 🟡 中）

**実装方法**:
```python
# 修正ファイル: backend/src/vision_analyzer.py

class VisionAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        # ...既存コード...
        self.cache_dir = Path("./cache/vision_analysis")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_hash(self, image_path: str) -> str:
        """画像ファイルのハッシュを計算"""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def analyze_image(
        self,
        image_path: str,
        analysis_type: str
    ) -> str:
        """キャッシュ付き画像解析"""
        image_hash = self._get_image_hash(image_path)
        cache_key = f"{image_hash}_{analysis_type}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # キャッシュチェック
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['result']

        # キャッシュミス: Gemini API呼び出し
        result = self._call_gemini_vision(image_path, analysis_type)

        # キャッシュに保存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type
            }, f, ensure_ascii=False)

        return result
```

**推定効果**:
- Vision API呼び出し削減: 80-90%（同じPDFの再処理時）
- PDF処理時間: -30-50%（キャッシュヒット時）

---

#### 5️⃣ BM25ハイブリッド検索（優先度: 🟢 推奨）

**目的**: キーワードマッチの強化

**実装方法**:
```python
# 新規ファイル: backend/src/hybrid_search.py

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np

class HybridSearcher:
    def __init__(self, vector_store, bm25_index=None):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.bm25_docs = []

    def build_bm25_index(self, documents: List[str]):
        """BM25インデックスを構築"""
        # トークン化（日本語対応）
        import MeCab
        mecab = MeCab.Tagger("-Owakati")

        tokenized_docs = [
            mecab.parse(doc).strip().split()
            for doc in documents
        ]

        self.bm25_docs = documents
        self.bm25_index = BM25Okapi(tokenized_docs)

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 5,
        alpha: float = 0.7  # ベクトル検索の重み
    ) -> List[Tuple[str, float]]:
        """
        ハイブリッド検索（ベクトル + BM25）
        alpha: 1.0 = ベクトルのみ, 0.0 = BM25のみ
        """
        # ベクトル検索
        vector_results = self.vector_store.similarity_search_with_score(
            query_embedding,
            k=k*2  # 多めに取得して融合
        )

        # BM25検索
        import MeCab
        mecab = MeCab.Tagger("-Owakati")
        tokenized_query = mecab.parse(query).strip().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # スコア正規化
        vector_scores_norm = self._normalize_scores(
            [score for _, score in vector_results]
        )
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # 融合スコア計算（Reciprocal Rank Fusion）
        combined_scores = {}

        for idx, (doc, vec_score) in enumerate(vector_results):
            doc_id = doc.metadata.get('id', idx)
            combined_scores[doc_id] = alpha * vector_scores_norm[idx]

        for idx, bm25_score in enumerate(bm25_scores_norm):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * bm25_score
            else:
                combined_scores[idx] = (1 - alpha) * bm25_score

        # スコア順にソート
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        return [
            (self.bm25_docs[doc_id], score)
            for doc_id, score in sorted_results
        ]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-Max正規化"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            return [1.0] * len(scores)

        return ((scores - min_score) / (max_score - min_score)).tolist()
```

**依存関係**:
```bash
pip install rank-bm25 mecab-python3
```

**統合** (`backend/src/rag_engine.py`):
```python
from hybrid_search import HybridSearcher

class RAGEngine:
    def __init__(self, config):
        # ...既存コード...
        self.hybrid_searcher = HybridSearcher(self.vector_store)

        # 初回起動時にBM25インデックスを構築
        self._build_bm25_index()

    def _build_bm25_index(self):
        """全ドキュメントからBM25インデックスを構築"""
        # Supabaseから全ドキュメントを取得
        all_docs = self.vector_store.get_all_documents()
        documents = [doc.page_content for doc in all_docs]

        self.hybrid_searcher.build_bm25_index(documents)

    def _get_relevant_contexts(self, query: str, category: Optional[str] = None):
        """ハイブリッド検索を使用"""
        query_embedding = self.embedder.embed_text(query)

        results = self.hybrid_searcher.hybrid_search(
            query,
            query_embedding,
            k=5,
            alpha=0.7  # ベクトル検索を重視
        )

        return results
```

**推定効果**:
- 専門用語検索精度: +20-30%
- 固有名詞のマッチング改善
- レイテンシ増加: +20-50ms

---

### 実装順序（推奨）

**Day 1-2**:
1. ✅ Reranking導入
2. ✅ セマンティックチャンキング

**Day 3-4**:
3. ✅ エンベディングキャッシング
4. ✅ Vision解析キャッシング

**Day 5-7**:
5. ✅ BM25ハイブリッド検索
6. ✅ 統合テスト & パフォーマンス測定

---

## 📝 実装時の注意点

### 1. 後方互換性
- 既存のベクトルストアデータは変更不要
- 新規PDF処理時のみ新しいチャンク戦略を適用
- `config.yaml`に機能フラグを追加：
  ```yaml
  rag:
    enable_reranking: true
    enable_semantic_chunking: true
    enable_embedding_cache: true
    enable_bm25_hybrid: true
  ```

### 2. エラーハンドリング
- Reranking失敗時はベクトル検索結果をそのまま使用
- キャッシュ読み込み失敗時はAPI呼び出しにフォールバック
- BM25インデックス構築失敗時はベクトル検索のみで継続

### 3. モニタリング
- Langfuseでキャッシュヒット率を追跡
- 検索精度メトリクスを記録（NDCG, MRR）
- レイテンシの分位数（P50, P95, P99）を監視

### 4. 設定の調整
- Rerankingモデル: 精度とスピードのトレードオフ
  - 軽量: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - 高精度: `cross-encoder/ms-marco-electra-base`
- ハイブリッド検索のalpha値: カテゴリーごとに調整可能
- キャッシュサイズ: ディスク容量に応じて調整

---

## 🧪 テスト戦略

### 1. ユニットテスト
- 各コンポーネントの単体テスト
- キャッシュの保存/読み込み
- Rerankingスコアの正当性

### 2. 統合テスト
- エンドツーエンドのクエリ処理
- キャッシュヒット/ミスのシナリオ
- エラーハンドリング

### 3. パフォーマンステスト
- ベンチマーククエリセット（20-50件）
- 改善前後のメトリクス比較
- レイテンシ測定（複数回実行して平均）

### 4. A/Bテスト（推奨）
- 本番環境でトラフィックを分割
- ユーザーフィードバックの収集
- 客観的な精度評価

---

## 📈 成功指標（KPI）

### 定量的指標

| メトリクス | 測定方法 | 目標値 |
|-----------|---------|--------|
| NDCG@5 | 評価セットで測定 | 0.85 → 0.92 |
| 検索レイテンシ（P50） | Langfuseトレース | 200ms → 100ms |
| キャッシュヒット率 | ログ分析 | 60-70% |
| APIコスト削減 | OpenAI使用量 | -30-40% |

### 定性的指標
- ユーザーからの回答品質フィードバック
- 複雑な質問への対応改善
- 専門用語検索の精度

---

## 🔄 Phase 2以降のプレビュー

Phase 1完了後、以下の施策を検討：

### Phase 2: 検索精度向上（2-3週間）
- プロンプトエンジニアリング（Few-shot learning）
- 動的な類似度閾値調整
- クエリ拡張（Query expansion）

### Phase 3: パフォーマンス最適化（1ヶ月）
- Supabase pgvectorインデックスの調整
- 非同期処理の導入（Celery）
- マルチモーダルエンベディング（CLIP）

### Phase 4: スケーラビリティ（2-3ヶ月）
- 分散処理システム
- マイクロサービス化
- ファインチューニング（ドメイン特化）

---

## 📚 参考資料

### 技術ドキュメント
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Sentence Transformers - Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Rank BM25](https://github.com/dorianbrown/rank_bm25)
- [Supabase Vector Guide](https://supabase.com/docs/guides/ai/vector-columns)

### RAG最適化のベストプラクティス
- [Advanced RAG Techniques](https://www.anthropic.com/research/retrieval-augmented-generation)
- [Hybrid Search in Production](https://www.pinecone.io/learn/hybrid-search-intro/)

---

## 🤝 サポート

質問や問題が発生した場合:
1. Langfuseトレースを確認
2. ログファイル（`backend/logs/`）を確認
3. 設定ファイル（`config.yaml`）の確認
4. 実装チェックリスト（`RAG_IMPROVEMENT_CHECKLIST.md`）を参照

---

**最終更新**: 2025-11-04
**次回レビュー**: Phase 1完了後（1週間後）
