# RAGシステム評価ツール

このディレクトリには、RAGシステムの性能を評価するためのツールとテストデータが含まれています。

## 📁 ファイル構成

```
evaluation/
├── README.md                          # 本ファイル（使い方ガイド）
├── test_questions.yaml                # 30問の評価用質問セット
├── evaluation_metrics.py              # メトリクス計算モジュール
├── run_evaluation.py                  # 評価実行スクリプト
├── evaluation_report_template.md      # レポートテンプレート
└── results/                           # 評価結果の保存先（自動生成）
    ├── evaluation_results_*.json      # 詳細結果
    ├── evaluation_summary_*.csv       # サマリー
    └── evaluation_statistics_*.json   # 統計情報
```

## 🎯 評価の目的

このツールは、以下の観点からRAGシステムを評価します:

1. **精度評価**
   - テキスト検索精度（目標: 85%以上）
   - 表検索精度（目標: 80%以上）
   - グラフ数値抽出精度（目標: 70%以上）

2. **性能評価**
   - 応答時間（目標: 15秒以内）
   - インデックス作成速度

3. **機能評価**
   - カテゴリーフィルタリング精度（目標: 100%）
   - ハイブリッド検索能力

## 📋 評価質問セット（30問）

質問は5つのカテゴリーに分類されています:

- **A. テキスト検索評価** (10問): 基本的な情報検索能力
- **B. 表検索評価** (5問): 表の検索と内容解析能力
- **C. グラフ検索・数値抽出評価** (5問): グラフからの数値データ抽出能力
- **D. ハイブリッド検索評価** (5問): テキストと画像の統合検索能力
- **E. カテゴリーフィルタリング評価** (5問): カテゴリー機能の正確性

詳細は `test_questions.yaml` を参照してください。

## 🚀 使い方

### 1. 前提条件

評価を実行する前に、以下を準備してください:

1. **RAGシステムが実装・起動していること**
   - `src/rag_engine.py` が正しく実装されている
   - 必要な依存パッケージがインストールされている

2. **テストデータ（PDF）がアップロードされていること**
   - 評価用のPDFファイルを準備
   - カテゴリー（「製品マニュアル」「技術仕様書」「運用レポート」）を設定
   - システムにアップロードしてインデックス化

3. **環境変数が設定されていること**
   - `.env` ファイルに `OPENAI_API_KEY` などを設定

### 2. 基本的な実行方法

```bash
# 全質問タイプを評価（推奨）
python evaluation/run_evaluation.py

# 特定の質問タイプのみ評価
python evaluation/run_evaluation.py --types text_search table_search

# カスタム設定で実行
python evaluation/run_evaluation.py --config config.yaml --questions evaluation/test_questions.yaml --output evaluation/results
```

### 3. コマンドラインオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--config` | 設定ファイルのパス | `config.yaml` |
| `--questions` | テスト質問ファイルのパス | `evaluation/test_questions.yaml` |
| `--output` | 結果出力ディレクトリ | `evaluation/results` |
| `--types` | 評価する質問タイプ（複数指定可） | 全タイプ |

**質問タイプの選択肢**:
- `text_search`: テキスト検索評価
- `table_search`: 表検索評価
- `graph_search`: グラフ検索評価
- `hybrid_search`: ハイブリッド検索評価
- `category_filtering`: カテゴリーフィルタリング評価

### 4. 実行例

```bash
# 例1: 全質問を評価
python evaluation/run_evaluation.py

# 例2: テキスト検索と表検索のみ評価
python evaluation/run_evaluation.py --types text_search table_search

# 例3: カスタム出力先を指定
python evaluation/run_evaluation.py --output my_evaluation_results

# 例4: カテゴリーフィルタリングのみを詳細にテスト
python evaluation/run_evaluation.py --types category_filtering
```

## 📊 評価結果の見方

評価実行後、以下のファイルが生成されます:

### 1. `evaluation_results_YYYYMMDD_HHMMSS.json`

各質問の詳細な評価結果（JSON形式）:

```json
[
  {
    "question_id": "T01",
    "question": "この製品の主な特徴は何ですか？",
    "category": "製品マニュアル",
    "answer": "...",
    "response_time": 12.5,
    "accuracy_score": 4.5,
    "reference_validity": 1.0,
    "keyword_match_score": 0.85,
    "sources": [...],
    "images": [...]
  },
  ...
]
```

### 2. `evaluation_summary_YYYYMMDD_HHMMSS.csv`

サマリー情報（CSV形式、Excelで開ける）:

| question_id | question | category | response_time | accuracy_score | ... |
|-------------|----------|----------|---------------|----------------|-----|
| T01 | この製品の... | 製品マニュアル | 12.50 | 4.5 | ... |

### 3. `evaluation_statistics_YYYYMMDD_HHMMSS.json`

統計情報と成功基準の達成状況:

```json
{
  "total_questions": 30,
  "accuracy": {
    "average_score": 4.2,
    "normalized_score": 0.84
  },
  "performance": {
    "average_response_time": 13.5,
    "max_response_time": 18.2
  },
  "success_criteria": {
    "text_accuracy": true,
    "table_accuracy": false,
    ...
  }
}
```

### 4. コンソール出力

評価実行中、進捗とサマリーがコンソールに表示されます:

```
================================================================================
評価サマリー
================================================================================

総質問数: 30

【精度評価】
  平均精度スコア: 4.20/5.0 (84.0%)
  参照元の妥当性: 95.0%
  画像検索精度: 80.0%
  数値抽出精度: 72.0%

【性能評価】
  平均応答時間: 13.50秒
  最大応答時間: 18.20秒
  最小応答時間: 8.30秒

【カテゴリーフィルタリング】
  精度: 100.0%

【成功基準の達成状況】
  テキスト検索精度 (≥85%): ✗ 未達成
  表検索精度 (≥80%): ✓ 達成
  グラフ数値抽出精度 (≥70%): ✓ 達成
  応答時間 (≤15秒): ✓ 達成
  カテゴリーフィルタリング (100%): ✓ 達成

  総合評価: ✗ 一部基準未達成
```

## 📈 評価メトリクスの詳細

### 1. 精度スコア（Accuracy Score）

**スコア範囲**: 1-5

| スコア | 説明 |
|--------|------|
| 5 | 完全に正確で完璧な回答 |
| 4 | ほぼ正確だが小さな不足あり |
| 3 | 部分的に正確だが重要な情報が欠落 |
| 2 | 不正確な情報が含まれる |
| 1 | 完全に不正確または無関連 |

**注意**: 初期スコアはキーワードマッチングベースで自動計算されますが、最終的には人間による検証が必要です。

### 2. 参照元の妥当性（Reference Validity）

**スコア範囲**: 0 or 1

- **1**: 正しい参照元（カテゴリー一致）
- **0**: 不適切な参照元（カテゴリー不一致）

### 3. 画像検索精度（Image Search Accuracy）

**スコア範囲**: 0, 0.5, 1.0

- **1.0**: 正しい表/グラフを表示
- **0.5**: 関連するが最適でない画像
- **0.0**: 不適切な画像または画像なし

### 4. 数値抽出精度（Numeric Extraction Accuracy）

**スコア範囲**: 0, 0.5, 1.0

- **1.0**: 数値が正確（±5%以内）
- **0.5**: 数値に誤差あり（±10%以内）
- **0.0**: 数値が大きく異なるor抽出失敗

## 🔧 カスタマイズ方法

### 1. 質問を追加/変更する

`test_questions.yaml` を編集:

```yaml
text_search:
  - id: "T11"  # 新しい質問ID
    question: "新しい質問内容"
    category: "製品マニュアル"
    expected_type: "text"
    evaluation_criteria:
      - "評価基準1"
      - "評価基準2"
    expected_keywords: ["キーワード1", "キーワード2"]
    accuracy_weight: 1.0
```

### 2. 評価基準を変更する

`test_questions.yaml` の末尾にある `evaluation_settings` を編集:

```yaml
evaluation_settings:
  success_criteria:
    text_accuracy_threshold: 0.90  # 85% → 90%に変更
    table_accuracy_threshold: 0.85  # 80% → 85%に変更
```

### 3. メトリクス計算ロジックをカスタマイズする

`evaluation_metrics.py` のメソッドを編集:

```python
def calculate_accuracy_score(self, answer, expected_keywords, evaluation_criteria):
    # カスタムロジックを実装
    ...
```

## 🐛 トラブルシューティング

### エラー: `ImportError: cannot import name 'RAGEngine'`

**原因**: RAGエンジンが実装されていない、またはパスが正しくない

**解決策**:
1. `src/rag_engine.py` が存在することを確認
2. `RAGEngine` クラスが正しく実装されているか確認
3. プロジェクトルートから実行しているか確認

### エラー: `FileNotFoundError: config.yaml not found`

**原因**: 設定ファイルが見つからない

**解決策**:
```bash
# プロジェクトルートから実行
cd C:\Users\r0u8b\DEV\PoC_chatbot
python evaluation/run_evaluation.py
```

### エラー: 評価中に質問がタイムアウトする

**原因**: RAGエンジンの処理が遅い、またはAPIレート制限

**解決策**:
1. `config.yaml` で処理を最適化（チャンクサイズ、検索数など）
2. APIキーのレート制限を確認
3. 一部の質問タイプのみを評価（`--types` オプション）

### 精度スコアが低すぎる/高すぎる

**原因**: 自動スコアリングは初期値のみで、人間による調整が必要

**解決策**:
1. JSON結果ファイルを開く
2. 各質問の回答を人間が確認
3. スコアを手動で調整
4. 調整後のデータで再計算

## 📝 評価レポートの作成

評価結果から正式なレポートを作成する手順:

1. **評価を実行**
   ```bash
   python evaluation/run_evaluation.py
   ```

2. **結果ファイルを確認**
   - `evaluation/results/` 内の最新ファイルを開く

3. **レポートテンプレートを使用**
   - `evaluation_report_template.md` をコピー
   - プレースホルダー（`{変数名}`）を実際の値で置換

4. **人間による検証を追加**
   - 各質問の回答を目視確認
   - 精度スコアを調整
   - 定性評価を追加
   - 改善推奨事項を記入

## 🎓 ベストプラクティス

### 評価を行う前に

1. **テストデータを準備**
   - 実際のユースケースに近いPDFを使用
   - 表・グラフを含むページが20-40%程度含まれることが理想

2. **システムを安定させる**
   - 評価前に数回テスト実行
   - エラーが出ないことを確認

3. **ベースラインを記録**
   - 初回評価結果を保存
   - 改善後の比較に使用

### 評価結果の解釈

1. **自動スコアは参考値**
   - キーワードマッチングベースの初期スコア
   - 必ず人間による検証を行う

2. **応答時間の変動に注意**
   - ネットワーク状況やAPI負荷で変動
   - 複数回実行して平均を取る

3. **カテゴリー別に分析**
   - どの機能が弱いか特定
   - 優先的に改善すべき箇所を判断

### 継続的な評価

1. **定期的に評価を実行**
   - コード変更後
   - モデル変更後
   - 設定調整後

2. **結果を記録**
   - タイムスタンプ付きで保存
   - 改善の経過を追跡

3. **問題のある質問を調査**
   - スコアが低い質問を特定
   - 原因を分析して改善

## 📚 参考資料

- **要件定義書**: `requirements_definition.md`（成功基準の詳細）
- **設定ファイル**: `config.yaml`（システムパラメータ）
- **質問セット**: `test_questions.yaml`（評価質問の詳細）
- **メトリクス計算**: `evaluation_metrics.py`（計算ロジックの実装）

## 💡 よくある質問（FAQ）

**Q: 評価にどれくらい時間がかかりますか？**
A: 30問で約7-10分程度（1問あたり15秒の目標値の場合）。実際の応答時間によって変動します。

**Q: テストデータのPDFはどこに配置すればいいですか？**
A: `data/uploaded_pdfs/` にアップロードし、Streamlit UIからインデックス化してください。

**Q: 一部の質問だけ評価したい場合は？**
A: `--types` オプションで質問タイプを指定するか、`test_questions.yaml` から不要な質問を削除してください。

**Q: 評価結果をグラフ化したい**
A: CSV結果ファイルをExcelやPythonのmatplotlibで可視化できます。

**Q: 成功基準を満たさなかった場合は？**
A: 統計情報ファイルで未達成の項目を確認し、該当する機能を重点的に改善してください。

---

**作成日**: 2025-01-21
**バージョン**: 1.0
**対象システム**: PDF RAGシステム（PoC版）
