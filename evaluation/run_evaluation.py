"""
RAGシステム評価実行スクリプト

test_questions.yamlから質問を読み込み、RAGシステムに問い合わせて
評価メトリクスを計算し、結果をJSON/CSV形式で保存します。
"""

import sys
import os
import time
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 評価モジュールをインポート
from evaluation.evaluation_metrics import EvaluationMetrics, EvaluationResult

# RAGエンジンをインポート
try:
    from src.rag_engine import RAGEngine
    from src.utils import load_config, load_environment, ensure_directories
    from src.vector_store import VectorStore
    from src.text_embedder import TextEmbedder
except ImportError as e:
    print(f"エラー: RAGエンジンのインポートに失敗しました: {e}")
    print("src/rag_engine.py、src/utils.py、src/vector_store.py、src/text_embedder.pyが存在することを確認してください。")
    sys.exit(1)


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAGシステムの評価を実行するクラス"""

    def __init__(self, config_path: str = "config.yaml", questions_path: str = "evaluation/test_questions.yaml"):
        """
        初期化

        Args:
            config_path: システム設定ファイルのパス
            questions_path: テスト質問ファイルのパス
        """
        self.config = load_config(config_path)
        self.questions_data = self._load_questions(questions_path)
        self.metrics = EvaluationMetrics()

        # 環境変数読み込みとディレクトリ確保
        load_environment()
        ensure_directories()

        # VectorStoreとEmbedderを初期化
        self.vector_store = VectorStore(self.config)
        self.embedder = TextEmbedder(self.config)

        # RAGエンジンを初期化
        try:
            self.rag_engine = RAGEngine(
                self.config,
                self.vector_store,
                self.embedder
            )
            logger.info("RAGエンジンの初期化に成功しました")
        except Exception as e:
            logger.error(f"RAGエンジンの初期化に失敗しました: {e}")
            raise

    def _load_questions(self, questions_path: str) -> Dict[str, Any]:
        """
        テスト質問ファイルを読み込む

        Args:
            questions_path: 質問ファイルのパス

        Returns:
            質問データ
        """
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"テスト質問を読み込みました: {questions_path}")
            return data
        except Exception as e:
            logger.error(f"質問ファイルの読み込みに失敗しました: {e}")
            raise

    def run_evaluation(self, question_types: List[str] = None) -> None:
        """
        評価を実行

        Args:
            question_types: 評価する質問タイプのリスト（None の場合は全て）
                           ["text_search", "table_search", "graph_search", "hybrid_search", "category_filtering"]
        """
        if question_types is None:
            question_types = ["text_search", "table_search", "graph_search", "hybrid_search", "category_filtering"]

        logger.info("評価を開始します...")

        for question_type in question_types:
            questions = self.questions_data.get(question_type, [])

            if not questions:
                logger.warning(f"質問タイプ '{question_type}' に質問が見つかりません")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"質問タイプ: {question_type} ({len(questions)}問)")
            logger.info(f"{'='*60}")

            for question_data in questions:
                self._evaluate_single_question(question_data, question_type)

        logger.info("\n評価が完了しました")

    def _evaluate_single_question(self, question_data: Dict[str, Any], question_type: str) -> None:
        """
        1つの質問を評価

        Args:
            question_data: 質問データ
            question_type: 質問タイプ
        """
        question_id = question_data.get("id", "unknown")
        question = question_data.get("question", "")
        category = question_data.get("category", "")

        logger.info(f"\n質問ID: {question_id}")
        logger.info(f"質問: {question}")
        logger.info(f"カテゴリー: {category}")

        try:
            # 質問を実行して応答時間を測定
            start_time = time.time()

            # RAGエンジンに質問を投げる
            # 注: 実際のRAGエンジンのAPIに合わせて調整してください
            response = self.rag_engine.query(
                question=question,
                category=category,
                chat_history=[]
            )

            end_time = time.time()
            response_time = end_time - start_time

            # 応答から情報を抽出
            answer = response.get("answer", "")
            sources_dict = response.get("sources", {})

            # sourcesが辞書形式の場合はリストに変換
            if isinstance(sources_dict, dict):
                sources = sources_dict.get("text", []) + sources_dict.get("images", [])
            else:
                sources = sources_dict if sources_dict else []

            images = response.get("images", [])

            logger.info(f"応答時間: {response_time:.2f}秒")
            logger.info(f"回答（先頭100文字）: {answer[:100]}...")

            # 評価メトリクスを計算
            result = self._calculate_metrics(
                question_id=question_id,
                question=question,
                category=category,
                answer=answer,
                sources=sources,
                images=images,
                response_time=response_time,
                question_data=question_data
            )

            # 結果を追加
            self.metrics.add_result(result)

        except Exception as e:
            logger.error(f"質問 {question_id} の評価中にエラーが発生しました: {e}")

            # エラー結果を記録
            error_result = EvaluationResult(
                question_id=question_id,
                question=question,
                category=category,
                answer="",
                response_time=0.0,
                error=str(e)
            )
            self.metrics.add_result(error_result)

    def _calculate_metrics(
        self,
        question_id: str,
        question: str,
        category: str,
        answer: str,
        sources: List[Dict[str, Any]],
        images: List[str],
        response_time: float,
        question_data: Dict[str, Any]
    ) -> EvaluationResult:
        """
        評価メトリクスを計算

        Args:
            question_id: 質問ID
            question: 質問文
            category: カテゴリー
            answer: 回答
            sources: 参照元リスト
            images: 画像パスリスト
            response_time: 応答時間
            question_data: 質問データ（評価基準を含む）

        Returns:
            評価結果
        """
        expected_keywords = question_data.get("expected_keywords", [])
        evaluation_criteria = question_data.get("evaluation_criteria", [])
        expected_type = question_data.get("expected_type", "")
        expected_image = question_data.get("expected_image", False)
        verify_category_filter = question_data.get("verify_category_filter", False)

        # 精度スコアを計算
        accuracy_score = self.metrics.calculate_accuracy_score(
            answer=answer,
            expected_keywords=expected_keywords,
            evaluation_criteria=evaluation_criteria
        )

        # 参照元の妥当性を評価
        reference_validity = self.metrics.calculate_reference_validity(
            sources=sources,
            expected_category=category
        )

        # 画像検索精度を評価
        image_search_accuracy = None
        if expected_image:
            image_search_accuracy = self.metrics.calculate_image_search_accuracy(
                images=images,
                expected_image=expected_image,
                expected_type=expected_type
            )

        # 数値抽出精度を評価（グラフ関連の質問のみ）
        numeric_extraction_accuracy = None
        if question_data.get("requires_numeric_data", False):
            numeric_extraction_accuracy = self.metrics.calculate_numeric_extraction_accuracy(
                answer=answer,
                ground_truth_numbers=None  # 正解データがある場合は指定
            )

        # キーワードマッチングスコアを計算
        keyword_match_score = self.metrics.calculate_keyword_match_score(
            answer=answer,
            expected_keywords=expected_keywords
        )

        # カテゴリーフィルタリングの検証
        category_filter_correct = None
        if verify_category_filter:
            category_filter_correct = self.metrics.verify_category_filter(
                sources=sources,
                expected_category=category
            )

        # 結果を作成
        result = EvaluationResult(
            question_id=question_id,
            question=question,
            category=category,
            answer=answer,
            response_time=response_time,
            accuracy_score=accuracy_score,
            reference_validity=reference_validity,
            image_search_accuracy=image_search_accuracy,
            numeric_extraction_accuracy=numeric_extraction_accuracy,
            keyword_match_score=keyword_match_score,
            category_filter_correct=category_filter_correct,
            sources=sources,
            images=images
        )

        return result

    def save_results(self, output_dir: str = "evaluation/results") -> None:
        """
        評価結果を保存

        Args:
            output_dir: 出力ディレクトリ
        """
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)

        # タイムスタンプを生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON形式で詳細結果を保存
        json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        self._save_json(json_path)

        # CSV形式でサマリーを保存
        csv_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
        self._save_csv(csv_path)

        # サマリー統計を保存
        summary_path = os.path.join(output_dir, f"evaluation_statistics_{timestamp}.json")
        self._save_summary_statistics(summary_path)

        logger.info(f"\n結果を保存しました:")
        logger.info(f"  - 詳細結果: {json_path}")
        logger.info(f"  - サマリー: {csv_path}")
        logger.info(f"  - 統計情報: {summary_path}")

    def _save_json(self, output_path: str) -> None:
        """JSON形式で結果を保存"""
        results_data = []

        for result in self.metrics.results:
            result_dict = {
                "question_id": result.question_id,
                "question": result.question,
                "category": result.category,
                "answer": result.answer,
                "response_time": result.response_time,
                "accuracy_score": result.accuracy_score,
                "reference_validity": result.reference_validity,
                "image_search_accuracy": result.image_search_accuracy,
                "numeric_extraction_accuracy": result.numeric_extraction_accuracy,
                "keyword_match_score": result.keyword_match_score,
                "category_filter_correct": result.category_filter_correct,
                "sources": result.sources,
                "images": result.images,
                "error": result.error
            }
            results_data.append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

    def _save_csv(self, output_path: str) -> None:
        """CSV形式で結果を保存"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "question_id", "question", "category", "response_time",
                "accuracy_score", "reference_validity", "image_search_accuracy",
                "numeric_extraction_accuracy", "keyword_match_score",
                "category_filter_correct", "has_error"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.metrics.results:
                writer.writerow({
                    "question_id": result.question_id,
                    "question": result.question,
                    "category": result.category,
                    "response_time": f"{result.response_time:.2f}",
                    "accuracy_score": result.accuracy_score if result.accuracy_score is not None else "",
                    "reference_validity": result.reference_validity if result.reference_validity is not None else "",
                    "image_search_accuracy": result.image_search_accuracy if result.image_search_accuracy is not None else "",
                    "numeric_extraction_accuracy": result.numeric_extraction_accuracy if result.numeric_extraction_accuracy is not None else "",
                    "keyword_match_score": f"{result.keyword_match_score:.2f}" if result.keyword_match_score is not None else "",
                    "category_filter_correct": result.category_filter_correct if result.category_filter_correct is not None else "",
                    "has_error": "Yes" if result.error else "No"
                })

    def _save_summary_statistics(self, output_path: str) -> None:
        """サマリー統計を保存"""
        summary = self.metrics.calculate_summary_statistics()

        # カテゴリー別統計を追加
        summary["category_specific"] = {
            "text_search": self.metrics.calculate_category_specific_stats("text"),
            "table_search": self.metrics.calculate_category_specific_stats("table"),
            "graph_search": self.metrics.calculate_category_specific_stats("graph"),
            "hybrid_search": self.metrics.calculate_category_specific_stats("hybrid"),
            "category_filtering": self.metrics.calculate_category_specific_stats("category")
        }

        # 成功基準のチェック結果を追加
        summary["success_criteria"] = self.metrics.check_success_criteria(summary)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        """サマリーをコンソールに出力"""
        summary = self.metrics.calculate_summary_statistics()

        print("\n" + "="*80)
        print("評価サマリー")
        print("="*80)

        print(f"\n総質問数: {summary['total_questions']}")

        print(f"\n【精度評価】")
        print(f"  平均精度スコア: {summary['accuracy']['average_score']:.2f}/5.0 ({summary['accuracy']['normalized_score']*100:.1f}%)")
        print(f"  参照元の妥当性: {summary['reference_validity']['average']*100:.1f}%")
        print(f"  画像検索精度: {summary['image_search']['average_accuracy']*100:.1f}%")
        print(f"  数値抽出精度: {summary['numeric_extraction']['average_accuracy']*100:.1f}%")

        print(f"\n【性能評価】")
        print(f"  平均応答時間: {summary['performance']['average_response_time']:.2f}秒")
        print(f"  最大応答時間: {summary['performance']['max_response_time']:.2f}秒")
        print(f"  最小応答時間: {summary['performance']['min_response_time']:.2f}秒")

        print(f"\n【カテゴリーフィルタリング】")
        print(f"  精度: {summary['category_filtering']['accuracy']*100:.1f}%")

        print(f"\n【エラー】")
        print(f"  エラー発生数: {summary['errors']['count']}")
        print(f"  エラー率: {summary['errors']['rate']*100:.1f}%")

        # 成功基準のチェック結果を表示
        success_criteria = self.metrics.check_success_criteria(summary)
        print(f"\n【成功基準の達成状況】")
        print(f"  テキスト検索精度 (≥85%): {'✓ 達成' if success_criteria['text_accuracy'] else '✗ 未達成'}")
        print(f"  表検索精度 (≥80%): {'✓ 達成' if success_criteria['table_accuracy'] else '✗ 未達成'}")
        print(f"  グラフ数値抽出精度 (≥70%): {'✓ 達成' if success_criteria['graph_numeric_accuracy'] else '✗ 未達成'}")
        print(f"  応答時間 (≤15秒): {'✓ 達成' if success_criteria['response_time'] else '✗ 未達成'}")
        print(f"  カテゴリーフィルタリング (100%): {'✓ 達成' if success_criteria['category_filtering'] else '✗ 未達成'}")
        print(f"\n  総合評価: {'✓ 全基準達成' if success_criteria['overall'] else '✗ 一部基準未達成'}")

        print("\n" + "="*80)


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="RAGシステムの評価を実行")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）"
    )
    parser.add_argument(
        "--questions",
        default="evaluation/test_questions.yaml",
        help="テスト質問ファイルのパス（デフォルト: evaluation/test_questions.yaml）"
    )
    parser.add_argument(
        "--output",
        default="evaluation/results",
        help="結果出力ディレクトリ（デフォルト: evaluation/results）"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["text_search", "table_search", "graph_search", "hybrid_search", "category_filtering"],
        help="評価する質問タイプ（デフォルト: 全タイプ）"
    )

    args = parser.parse_args()

    try:
        # 評価を実行
        evaluator = RAGEvaluator(
            config_path=args.config,
            questions_path=args.questions
        )

        evaluator.run_evaluation(question_types=args.types)

        # 結果を保存
        evaluator.save_results(output_dir=args.output)

        # サマリーを表示
        evaluator.print_summary()

    except Exception as e:
        logger.error(f"評価の実行中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
