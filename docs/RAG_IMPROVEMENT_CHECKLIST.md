# RAG性能向上 Phase 1 実装チェックリスト

**対象フェーズ**: Phase 1 - Quick Wins（1週間）
**開始日**: ___________
**完了予定日**: ___________

---

## 📋 実装概要

| タスク | 優先度 | 推定時間 | 状態 |
|--------|--------|----------|------|
| 1. Reranking導入 | 🔴 最高 | 1-2日 | ⬜ 未着手 |
| 2. セマンティックチャンキング | 🟠 高 | 1日 | ⬜ 未着手 |
| 3. エンベディングキャッシング | 🟡 中 | 1日 | ⬜ 未着手 |
| 4. Vision解析キャッシング | 🟡 中 | 0.5日 | ⬜ 未着手 |
| 5. BM25ハイブリッド検索 | 🟢 推奨 | 1-2日 | ⬜ 未着手 |
| 6. 統合テスト & ベンチマーク | 🔵 必須 | 1日 | ⬜ 未着手 |

**状態記号**: ⬜ 未着手 / 🟡 進行中 / ✅ 完了 / ❌ ブロック

---

## 🎯 タスク 1: Reranking導入

### 準備

- [ ] 依存関係インストール
  ```bash
  cd backend
  pip install sentence-transformers==2.2.2
  ```

- [ ] モデルの事前ダウンロード（オプションだが推奨）
  ```bash
  python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
  ```
  - 推定時間: 2-5分
  - モデルサイズ: ~80MB

### 実装

- [ ] 新規ファイル作成: `backend/src/reranker.py`
  - [ ] `Reranker`クラスを実装
  - [ ] `__init__()`: モデル初期化
  - [ ] `rerank()`: クエリとドキュメントのrerankingロジック

<details>
<summary>📝 実装テンプレート</summary>

```python
# backend/src/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Cross-Encoderモデルを初期化

        Args:
            model_name: 使用するCross-Encoderモデル名
        """
        logger.info(f"Initializing Reranker with model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker initialized successfully")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        クエリとドキュメントのペアをrerankし、
        スコアの高い順にインデックスとスコアを返す

        Args:
            query: 検索クエリ
            documents: rerankするドキュメントのリスト
            top_k: 返却する上位K件

        Returns:
            (インデックス, スコア)のタプルのリスト
        """
        if not documents:
            return []

        # クエリ-ドキュメントペアを作成
        pairs = [[query, doc] for doc in documents]

        # rerankスコアを計算
        scores = self.model.predict(pairs)

        # スコア順にソート
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_scores = scores[ranked_indices]

        logger.debug(
            f"Reranking completed: {len(documents)} docs -> top {top_k}",
            extra={"scores": ranked_scores.tolist()}
        )

        return list(zip(ranked_indices.tolist(), ranked_scores.tolist()))
```
</details>

- [ ] `backend/src/rag_engine.py` を修正
  - [ ] `Reranker`をインポート
  - [ ] `__init__()`: Rerankerインスタンスを初期化
  - [ ] `_get_relevant_contexts()`: Rerankingロジックを統合

<details>
<summary>📝 統合コード例</summary>

```python
# backend/src/rag_engine.py の修正箇所

from src.reranker import Reranker

class RAGEngine:
    def __init__(self, config):
        # ...既存コード...

        # Reranker初期化
        if config["rag"].get("enable_reranking", True):
            reranking_model = config.get("reranking", {}).get(
                "model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self.reranker = Reranker(model_name=reranking_model)
        else:
            self.reranker = None

    def _get_relevant_contexts(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[Dict]:
        """コンテキスト取得（Reranking統合版）"""
        query_embedding = self.embedder.embed_text(query)

        # 1次検索: 多めに取得
        top_k_initial = self.config.get("reranking", {}).get("top_k_initial", 10)

        text_results = self.vector_store.similarity_search(
            query_embedding,
            k=top_k_initial,
            filter={"category": category} if category else None
        )

        # Rerankingを適用
        if self.reranker and text_results:
            documents = [r.page_content for r in text_results]
            top_k_final = self.config.get("reranking", {}).get("top_k_final", 5)

            reranked_indices, scores = self.reranker.rerank(
                query,
                documents,
                top_k=top_k_final
            )

            # 上位K件を選択
            text_results = [text_results[idx] for idx, _ in reranked_indices]

        return text_results
```
</details>

- [ ] `config.yaml` に設定を追加
  ```yaml
  rag:
    enable_reranking: true

  reranking:
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_initial: 10
    top_k_final: 5
  ```

### テスト

- [ ] ユニットテストを作成: `backend/tests/test_reranker.py`
  - [ ] `test_reranker_initialization`: モデルが正しく初期化されるか
  - [ ] `test_reranker_improves_ranking`: 関連性の高いドキュメントが上位に来るか
  - [ ] `test_reranker_empty_documents`: 空のドキュメントリストの処理

<details>
<summary>📝 テストコード例</summary>

```python
# backend/tests/test_reranker.py

import pytest
from src.reranker import Reranker

class TestReranker:
    @pytest.fixture
    def reranker(self):
        return Reranker()

    def test_reranker_initialization(self, reranker):
        """Rerankerが正しく初期化されることを確認"""
        assert reranker.model is not None

    def test_reranker_improves_ranking(self, reranker):
        """Rerankingで関連性の高いドキュメントが上位に来ることを確認"""
        query = "表の作成方法"
        documents = [
            "グラフを作成するには、データを選択してグラフメニューから作成します。",
            "表を作成するには、挿入メニューから表を選択し、行数と列数を指定します。",
            "図を挿入するには、挿入メニューから図を選択します。"
        ]

        reranked_indices, scores = reranker.rerank(query, documents, top_k=1)

        # 2番目のドキュメント（インデックス1）が最も関連性が高い
        assert reranked_indices[0] == 1
        assert scores[0] > 0.5  # スコアが十分高いことを確認

    def test_reranker_empty_documents(self, reranker):
        """空のドキュメントリストでもエラーが発生しないことを確認"""
        query = "テストクエリ"
        documents = []

        result = reranker.rerank(query, documents)

        assert result == []
```
</details>

- [ ] 統合テスト
  ```bash
  cd backend
  pytest tests/test_reranker.py -v
  ```

- [ ] 手動テスト: サンプルクエリで検索精度が向上することを確認

### 完了条件

- [ ] テストがすべてパスする
- [ ] Rerankingが有効な場合、検索結果の順位が変わる
- [ ] Rerankingを無効にしても既存機能が動作する
- [ ] レイテンシ増加が100ms以内

---

## 🎯 タスク 2: セマンティックチャンキング

### 準備

- [ ] LangChainの`RecursiveCharacterTextSplitter`を確認
  ```bash
  python -c "from langchain.text_splitter import RecursiveCharacterTextSplitter; print('OK')"
  ```

### 実装

- [ ] `backend/src/pdf_processor.py` を修正
  - [ ] `RecursiveCharacterTextSplitter`をインポート
  - [ ] `__init__()`: text_splitterを初期化
  - [ ] `_chunk_text()`: セマンティックチャンキングを実装
  - [ ] `_preserve_table_context()`: 表のコンテキスト保持ロジック（新規メソッド）

<details>
<summary>📝 実装コード例</summary>

```python
# backend/src/pdf_processor.py の修正箇所

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class PDFProcessor:
    def __init__(self, config):
        # ...既存コード...

        # セマンティックチャンカーを初期化
        if config["rag"].get("enable_semantic_chunking", True):
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunking"]["chunk_size"],
                chunk_overlap=config["chunking"]["chunk_overlap"],
                length_function=self._count_tokens,
                separators=config["chunking"]["separators"],
                keep_separator=True
            )
        else:
            self.text_splitter = None

    def _chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """セマンティック境界を考慮したチャンク化"""

        # 表のコンテキストを保持
        text = self._preserve_table_context(text)

        # セマンティックチャンキング
        if self.text_splitter:
            chunks = self.text_splitter.split_text(text)
        else:
            # フォールバック: 既存のチャンキング
            chunks = self._legacy_chunk_text(text)

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
        表の前後にコンテキスト情報を保持
        表や図の参照を検出し、段落区切りを強化
        """
        patterns = [
            r'(表\s*\d+[.:].*?)(\n)',
            r'(図\s*\d+[.:].*?)(\n)',
            r'(Table\s+\d+[.:].*?)(\n)',
            r'(Figure\s+\d+[.:].*?)(\n)',
        ]

        for pattern in patterns:
            text = re.sub(pattern, r'\1\n\n', text)

        return text

    def _legacy_chunk_text(self, text: str) -> List[str]:
        """既存のチャンキングロジック（後方互換性）"""
        # ...既存の実装...
        pass
```
</details>

- [ ] `config.yaml` に設定を追加
  ```yaml
  rag:
    enable_semantic_chunking: true

  chunking:
    chunk_size: 800
    chunk_overlap: 150
    separators:
      - "\n\n"
      - "\n"
      - "。"
      - "．"
      - ". "
      - "! "
      - "? "
      - "；"
      - "、"
      - ", "
      - " "
      - ""
  ```

### テスト

- [ ] ユニットテストを作成/更新: `backend/tests/test_pdf_processor.py`
  - [ ] `test_semantic_chunking_preserves_paragraphs`: 段落が保持されるか
  - [ ] `test_table_context_preservation`: 表の見出しがコンテキストに含まれるか
  - [ ] `test_chunking_respects_semantic_boundaries`: セマンティック境界で分割されるか

<details>
<summary>📝 テストコード例</summary>

```python
def test_table_context_preservation(processor):
    """表の見出しが表のコンテンツと同じチャンクに含まれることを確認"""
    text = """
    これは前のテキストです。

    表1: ユーザー情報
    名前 | 年齢 | 住所
    田中 | 30 | 東京
    鈴木 | 25 | 大阪

    これは後のテキストです。
    """

    chunks = processor._chunk_text(text, page_num=1)

    # 表の見出しと内容が同じチャンクに含まれることを確認
    table_chunks = [c for c in chunks if "表1" in c["content"]]
    assert len(table_chunks) > 0
    assert "ユーザー情報" in table_chunks[0]["content"]
    assert "田中" in table_chunks[0]["content"]
```
</details>

- [ ] 統合テスト
  ```bash
  pytest tests/test_pdf_processor.py::test_table_context_preservation -v
  ```

- [ ] 手動テスト: 表を含むPDFで、表の見出しとコンテンツが同じチャンクになることを確認

### 完了条件

- [ ] テストがすべてパスする
- [ ] 表の見出しとコンテンツが同じチャンクに含まれる
- [ ] チャンク境界が文の途中でない（日本語・英語両方）
- [ ] 既存のチャンク数と大きく変わらない（±20%以内）

---

## 🎯 タスク 3: エンベディングキャッシング

### 準備

- [ ] キャッシュディレクトリを作成
  ```bash
  mkdir backend\cache\embeddings
  ```

### 実装

- [ ] 新規ファイル作成: `backend/src/embedding_cache.py`
  - [ ] `EmbeddingCache`クラスを実装
  - [ ] `get()`: キャッシュ取得
  - [ ] `set()`: キャッシュ保存
  - [ ] `_get_cache_key()`: ハッシュ生成
  - [ ] メモリキャッシュ（LRU）とディスクキャッシュの両方を実装

<details>
<summary>📝 実装テンプレート</summary>

```python
# backend/src/embedding_cache.py

import hashlib
import pickle
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, cache_dir: str = "./cache/embeddings", max_memory_items: int = 1000):
        """
        エンベディングキャッシュを初期化

        Args:
            cache_dir: キャッシュディレクトリのパス
            max_memory_items: メモリキャッシュの最大アイテム数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._max_memory_items = max_memory_items
        logger.info(f"EmbeddingCache initialized: {cache_dir}")

    def _get_cache_key(self, text: str) -> str:
        """テキストからキャッシュキー（SHA256ハッシュ）を生成"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """キャッシュから埋め込みを取得"""
        key = self._get_cache_key(text)

        # メモリキャッシュをチェック
        if key in self._memory_cache:
            logger.debug(f"Memory cache hit: {key[:8]}...")
            return self._memory_cache[key]

        # ディスクキャッシュをチェック
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"Disk cache hit: {key[:8]}...")

                # メモリキャッシュに追加
                self._add_to_memory_cache(key, embedding)
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        logger.debug(f"Cache miss: {key[:8]}...")
        return None

    def set(self, text: str, embedding: List[float]):
        """埋め込みをキャッシュに保存"""
        key = self._get_cache_key(text)

        # メモリキャッシュに追加
        self._add_to_memory_cache(key, embedding)

        # ディスクに永続化
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"Cached embedding: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _add_to_memory_cache(self, key: str, value: List[float]):
        """LRU方式でメモリキャッシュに追加"""
        if len(self._memory_cache) >= self._max_memory_items:
            # 最も古いアイテムを削除
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = value
```
</details>

- [ ] `backend/src/text_embedder.py` を修正
  - [ ] `EmbeddingCache`をインポート
  - [ ] `__init__()`: キャッシュを初期化
  - [ ] `embed_text()`: キャッシュチェックを追加
  - [ ] `embed_batch()`: バッチ処理でキャッシュを活用

<details>
<summary>📝 統合コード例</summary>

```python
# backend/src/text_embedder.py の修正箇所

from src.embedding_cache import EmbeddingCache

class TextEmbedder:
    def __init__(self, config):
        # ...既存コード...

        # キャッシュ初期化
        if config["cache"]["embedding"].get("enabled", True):
            cache_dir = config["cache"]["embedding"].get("directory", "./cache/embeddings")
            max_items = config["cache"]["embedding"].get("max_memory_items", 1000)
            self.cache = EmbeddingCache(cache_dir=cache_dir, max_memory_items=max_items)
        else:
            self.cache = None

    def embed_text(self, text: str) -> List[float]:
        """キャッシュ付き埋め込み生成"""
        # キャッシュをチェック
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        # キャッシュミス: API呼び出し
        embedding = self._call_openai_api(text)

        # キャッシュに保存
        if self.cache:
            self.cache.set(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """バッチ処理でキャッシュを活用"""
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        # キャッシュヒット/ミスを判定
        if self.cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # キャッシュミスのテキストをバッチ処理
        if uncached_texts:
            embeddings = self._call_openai_api_batch(uncached_texts)

            # 結果をキャッシュに保存 & results配列に格納
            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                if self.cache:
                    self.cache.set(text, embedding)
                results[idx] = embedding

        return results
```
</details>

- [ ] `config.yaml` に設定を追加
  ```yaml
  cache:
    embedding:
      enabled: true
      directory: "./cache/embeddings"
      max_memory_items: 1000
  ```

### テスト

- [ ] ユニットテストを作成: `backend/tests/test_embedding_cache.py`
  - [ ] `test_cache_hit`: キャッシュヒットの動作
  - [ ] `test_cache_miss`: キャッシュミスの動作
  - [ ] `test_cache_persistence`: ディスクへの永続化

<details>
<summary>📝 テストコード例</summary>

```python
# backend/tests/test_embedding_cache.py

import pytest
from src.embedding_cache import EmbeddingCache
import tempfile
import shutil

class TestEmbeddingCache:
    @pytest.fixture
    def cache_dir(self):
        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # テスト後にクリーンアップ
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, cache_dir):
        return EmbeddingCache(cache_dir=cache_dir, max_memory_items=10)

    def test_cache_miss(self, cache):
        """キャッシュにないテキストはNoneが返されることを確認"""
        result = cache.get("存在しないテキスト")
        assert result is None

    def test_cache_hit(self, cache):
        """キャッシュに保存したテキストが取得できることを確認"""
        text = "テストテキスト"
        embedding = [0.1, 0.2, 0.3]

        cache.set(text, embedding)
        result = cache.get(text)

        assert result == embedding

    def test_cache_persistence(self, cache_dir):
        """ディスクに永続化されることを確認"""
        cache1 = EmbeddingCache(cache_dir=cache_dir)
        text = "永続化テスト"
        embedding = [0.5, 0.6, 0.7]

        cache1.set(text, embedding)

        # 新しいインスタンスを作成
        cache2 = EmbeddingCache(cache_dir=cache_dir)
        result = cache2.get(text)

        assert result == embedding
```
</details>

- [ ] 統合テスト
  ```bash
  pytest tests/test_embedding_cache.py -v
  ```

- [ ] キャッシュヒット率の測定
  ```python
  # backend/evaluation/measure_cache_hit_rate.py
  # 同じクエリを複数回実行してキャッシュヒット率を測定
  ```

### 完了条件

- [ ] テストがすべてパスする
- [ ] キャッシュヒット時、API呼び出しが発生しない
- [ ] キャッシュヒット率が60%以上（再処理時）
- [ ] ディスクキャッシュが永続化される

---

## 🎯 タスク 4: Vision解析キャッシング

### 準備

- [ ] キャッシュディレクトリを作成
  ```bash
  mkdir backend\cache\vision_analysis
  ```

### 実装

- [ ] `backend/src/vision_analyzer.py` を修正
  - [ ] `__init__()`: キャッシュディレクトリを設定
  - [ ] `_get_image_hash()`: 画像ハッシュ生成（新規メソッド）
  - [ ] `analyze_image()`: キャッシュチェックを追加

<details>
<summary>📝 実装コード例</summary>

```python
# backend/src/vision_analyzer.py の修正箇所

import hashlib
import json
from pathlib import Path
from datetime import datetime

class VisionAnalyzer:
    def __init__(self, config):
        # ...既存コード...

        # キャッシュ設定
        if config["cache"]["vision"].get("enabled", True):
            cache_dir = config["cache"]["vision"].get("directory", "./cache/vision_analysis")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _get_image_hash(self, image_path: str) -> str:
        """画像ファイルのSHA256ハッシュを計算"""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def analyze_image(
        self,
        image_path: str,
        analysis_type: str
    ) -> str:
        """キャッシュ付き画像解析"""

        # キャッシュチェック
        if self.cache_dir:
            image_hash = self._get_image_hash(image_path)
            cache_key = f"{image_hash}_{analysis_type}"
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    logger.debug(f"Vision cache hit: {cache_key[:8]}...")
                    return cached_data['result']
                except Exception as e:
                    logger.warning(f"Failed to load vision cache: {e}")

        # キャッシュミス: Gemini API呼び出し
        result = self._call_gemini_vision(image_path, analysis_type)

        # キャッシュに保存
        if self.cache_dir:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'result': result,
                        'timestamp': datetime.now().isoformat(),
                        'analysis_type': analysis_type,
                        'image_hash': image_hash
                    }, f, ensure_ascii=False, indent=2)
                logger.debug(f"Cached vision result: {cache_key[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to save vision cache: {e}")

        return result
```
</details>

- [ ] `config.yaml` に設定を追加
  ```yaml
  cache:
    vision:
      enabled: true
      directory: "./cache/vision_analysis"
      expiry_days: 30  # オプション
  ```

### テスト

- [ ] ユニットテストを作成/更新: `backend/tests/test_vision_analyzer.py`
  - [ ] `test_vision_cache_hit`: キャッシュヒットの動作
  - [ ] `test_vision_cache_miss`: キャッシュミスの動作
  - [ ] `test_different_analysis_types`: 解析タイプごとにキャッシュが分かれるか

- [ ] 統合テスト
  ```bash
  pytest tests/test_vision_analyzer.py -v
  ```

- [ ] 手動テスト: 同じPDFを2回処理し、2回目でVision API呼び出しが減ることを確認

### 完了条件

- [ ] テストがすべてパスする
- [ ] 同じ画像の再解析時、API呼び出しが発生しない
- [ ] 異なる解析タイプ（table/graph）で別々にキャッシュされる
- [ ] PDF処理時間が30-50%短縮される（再処理時）

---

## 🎯 タスク 5: BM25ハイブリッド検索

### 準備

- [ ] 依存関係インストール
  ```bash
  pip install rank-bm25==0.2.2
  pip install mecab-python3==1.0.6
  ```

- [ ] MeCabのインストール確認（Windows）
  ```bash
  python -c "import MeCab; print(MeCab.Tagger('-Owakati').parse('これはテストです'))"
  ```
  - エラーが出る場合: https://github.com/ikegami-yukino/mecab/releases からバイナリをインストール

### 実装

- [ ] 新規ファイル作成: `backend/src/hybrid_search.py`
  - [ ] `HybridSearcher`クラスを実装
  - [ ] `build_bm25_index()`: BM25インデックス構築
  - [ ] `hybrid_search()`: ハイブリッド検索ロジック
  - [ ] `_normalize_scores()`: スコア正規化

<details>
<summary>📝 実装テンプレート</summary>

```python
# backend/src/hybrid_search.py

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import numpy as np
import MeCab
import logging

logger = logging.getLogger(__name__)

class HybridSearcher:
    def __init__(self, vector_store, alpha: float = 0.7):
        """
        ハイブリッド検索の初期化

        Args:
            vector_store: ベクトルストアのインスタンス
            alpha: ベクトル検索の重み（0-1）
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.bm25_index = None
        self.bm25_docs = []
        self.mecab = MeCab.Tagger("-Owakati")
        logger.info(f"HybridSearcher initialized with alpha={alpha}")

    def build_bm25_index(self, documents: List[str]):
        """BM25インデックスを構築"""
        logger.info(f"Building BM25 index for {len(documents)} documents...")

        # トークン化
        tokenized_docs = [
            self.mecab.parse(doc).strip().split()
            for doc in documents
        ]

        self.bm25_docs = documents
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info("BM25 index built successfully")

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """
        ハイブリッド検索（ベクトル + BM25）

        Args:
            query: 検索クエリ
            query_embedding: クエリの埋め込み
            k: 返却する件数
            filter: カテゴリーフィルタ等

        Returns:
            (ドキュメント, スコア)のリスト
        """
        if not self.bm25_index:
            logger.warning("BM25 index not built, using vector search only")
            results = self.vector_store.similarity_search_with_score(
                query_embedding, k=k, filter=filter
            )
            return [(doc.page_content, score) for doc, score in results]

        # ベクトル検索
        vector_results = self.vector_store.similarity_search_with_score(
            query_embedding,
            k=k*2,  # 多めに取得して融合
            filter=filter
        )

        # BM25検索
        tokenized_query = self.mecab.parse(query).strip().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # スコア正規化
        vector_scores_norm = self._normalize_scores(
            [score for _, score in vector_results]
        )
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # 融合スコア計算
        combined_scores = {}

        for idx, (doc, _) in enumerate(vector_results):
            doc_id = doc.metadata.get('id', idx)
            doc_content = doc.page_content
            combined_scores[doc_content] = self.alpha * vector_scores_norm[idx]

        for idx, bm25_score in enumerate(bm25_scores_norm):
            doc_content = self.bm25_docs[idx]
            if doc_content in combined_scores:
                combined_scores[doc_content] += (1 - self.alpha) * bm25_score
            else:
                combined_scores[doc_content] = (1 - self.alpha) * bm25_score

        # スコア順にソート
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        logger.debug(f"Hybrid search completed: {len(sorted_results)} results")
        return sorted_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-Max正規化"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            return [1.0] * len(scores)

        return ((scores - min_score) / (max_score - min_score)).tolist()
```
</details>

- [ ] `backend/src/rag_engine.py` を修正
  - [ ] `HybridSearcher`をインポート
  - [ ] `__init__()`: HybridSearcherを初期化、BM25インデックスを構築
  - [ ] `_get_relevant_contexts()`: ハイブリッド検索を使用

<details>
<summary>📝 統合コード例</summary>

```python
# backend/src/rag_engine.py の修正箇所

from src.hybrid_search import HybridSearcher

class RAGEngine:
    def __init__(self, config):
        # ...既存コード...

        # ハイブリッド検索初期化
        if config["hybrid_search"].get("enabled", True):
            alpha = config["hybrid_search"].get("alpha", 0.7)
            self.hybrid_searcher = HybridSearcher(self.vector_store, alpha=alpha)

            # BM25インデックス構築
            self._build_bm25_index()
        else:
            self.hybrid_searcher = None

    def _build_bm25_index(self):
        """全ドキュメントからBM25インデックスを構築"""
        logger.info("Building BM25 index from all documents...")

        # Supabaseから全ドキュメントを取得
        all_docs = self.vector_store.get_all_documents()
        documents = [doc.page_content for doc in all_docs]

        self.hybrid_searcher.build_bm25_index(documents)

    def _get_relevant_contexts(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[Dict]:
        """ハイブリッド検索を使用"""
        query_embedding = self.embedder.embed_text(query)

        if self.hybrid_searcher:
            results = self.hybrid_searcher.hybrid_search(
                query,
                query_embedding,
                k=5,
                filter={"category": category} if category else None
            )
            # (doc, score)のタプルをDictに変換
            return [{"content": doc, "score": score} for doc, score in results]
        else:
            # フォールバック: ベクトル検索のみ
            results = self.vector_store.similarity_search(
                query_embedding,
                k=5,
                filter={"category": category} if category else None
            )
            return [{"content": r.page_content} for r in results]
```
</details>

- [ ] `config.yaml` に設定を追加
  ```yaml
  hybrid_search:
    enabled: true
    alpha: 0.7  # ベクトル検索の重み
    bm25_k1: 1.5
    bm25_b: 0.75
  ```

### テスト

- [ ] ユニットテストを作成: `backend/tests/test_hybrid_search.py`
  - [ ] `test_bm25_index_building`: インデックスが正しく構築されるか
  - [ ] `test_hybrid_search_keyword_query`: キーワードクエリで適切な結果が返るか
  - [ ] `test_hybrid_search_semantic_query`: セマンティッククエリで適切な結果が返るか

- [ ] 統合テスト
  ```bash
  pytest tests/test_hybrid_search.py -v
  ```

- [ ] 手動テスト: キーワードマッチと意味的マッチの両方で検索精度が向上することを確認

### 完了条件

- [ ] テストがすべてパスする
- [ ] BM25インデックスが正しく構築される
- [ ] キーワードクエリでBM25スコアが効く
- [ ] セマンティッククエリでベクトルスコアが効く
- [ ] ハイブリッド検索で両方のメリットが活かされる

---

## 🎯 タスク 6: 統合テスト & ベンチマーク

### 統合テスト

- [ ] エンドツーエンドテストを実行
  ```bash
  cd backend
  pytest tests/ -v --cov=src
  ```

- [ ] 全機能を有効にしてPDF処理テスト
  - [ ] サンプルPDFをアップロード
  - [ ] PDF処理が正常に完了する
  - [ ] 質問応答が正常に動作する
  - [ ] キャッシュが適切に動作する

- [ ] 機能フラグのテスト
  - [ ] 各機能を個別に有効/無効にして動作確認
  - [ ] すべて無効にしても既存機能が動作する

### ベンチマーク

- [ ] ベンチマーククエリセットを準備
  ```json
  // backend/evaluation/benchmark_queries.json
  [
    {"query": "表の作成方法", "category": "manual"},
    {"query": "グラフのデータ範囲変更", "category": "manual"},
    // ...20-50件
  ]
  ```

- [ ] ベンチマークスクリプトを実行
  ```bash
  cd backend/evaluation
  python benchmark.py
  ```

- [ ] メトリクスを測定
  - [ ] 検索レイテンシ（P50, P95, P99）
  - [ ] 検索精度（NDCG@5, MRR）
  - [ ] キャッシュヒット率
  - [ ] APIコスト削減率

- [ ] 改善前後の比較表を作成

### パフォーマンス目標の確認

| メトリクス | 改善前 | 目標値 | 実測値 | 達成 |
|-----------|--------|--------|--------|------|
| 検索レイテンシ（P50） | 200-500ms | 100-300ms | ___ms | [ ] |
| 検索精度（NDCG@5） | 0.85 | 0.92 | ___ | [ ] |
| キャッシュヒット率 | 0% | 60-70% | ___% | [ ] |
| APIコスト削減 | - | 30-40% | ___% | [ ] |

### ドキュメント更新

- [ ] README.md を更新
  - [ ] 新機能の説明を追加
  - [ ] 設定例を追加

- [ ] config.yaml にコメントを追加
  - [ ] 各設定項目の説明
  - [ ] 推奨値

- [ ] CHANGELOG.md を作成/更新
  ```markdown
  # Changelog

  ## [Phase 1] - 2025-XX-XX

  ### Added
  - Reranking導入（検索精度+10-15%）
  - セマンティックチャンキング
  - エンベディングキャッシング
  - Vision解析キャッシング
  - BM25ハイブリッド検索

  ### Performance
  - 検索レイテンシ: 200-500ms → 100-300ms
  - 検索精度（NDCG@5）: 0.85 → 0.92
  - キャッシュヒット率: 60-70%
  ```

### 完了条件

- [ ] すべてのユニットテストがパスする
- [ ] 統合テストがパスする
- [ ] パフォーマンス目標を達成している
- [ ] ドキュメントが更新されている
- [ ] コードがmasterブランチにマージされている

---

## 📊 最終レビュー

### コードレビュー

- [ ] コードスタイルが統一されている（Black, flake8）
- [ ] 型ヒントが適切に使用されている
- [ ] Docstringが適切に記述されている
- [ ] エラーハンドリングが適切
- [ ] ログが適切に記録されている

### セキュリティ

- [ ] APIキーがハードコードされていない
- [ ] 入力検証が適切
- [ ] キャッシュファイルのパーミッションが適切

### パフォーマンス

- [ ] メモリリークがない
- [ ] 不要な再計算がない
- [ ] 並列処理が適切に使用されている

### ユーザー体験

- [ ] エラーメッセージが分かりやすい
- [ ] レスポンスが高速
- [ ] 回答の質が向上している

---

## 🎉 Phase 1 完了！

すべてのタスクが完了したら:

1. [ ] 成果をチームに共有
2. [ ] ユーザーフィードバックを収集
3. [ ] Phase 2の計画を開始

**お疲れ様でした！** 🚀

---

**作成日**: 2025-11-04
**最終更新**: ___________
**担当者**: ___________
