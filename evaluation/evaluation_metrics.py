"""
RAGシステム評価用メトリクス計算モジュール

このモジュールは、RAGシステムの性能を定量的に評価するための
各種メトリクス計算機能を提供します。
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    question_id: str
    question: str
    category: str
    answer: str
    response_time: float

    # 精度評価
    accuracy_score: Optional[float] = None  # 1-5のスコア
    reference_validity: Optional[float] = None  # 0 or 1
    image_search_accuracy: Optional[float] = None  # 0, 0.5, or 1
    numeric_extraction_accuracy: Optional[float] = None  # 0, 0.5, or 1

    # カテゴリーフィルタリング
    category_filter_correct: Optional[bool] = None

    # キーワードマッチング
    keyword_match_score: Optional[float] = None

    # 参照元情報
    sources: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None

    # エラー情報
    error: Optional[str] = None


class EvaluationMetrics:
    """評価メトリクスを計算するクラス"""

    def __init__(self):
        """初期化"""
        self.results: List[EvaluationResult] = []

    def calculate_accuracy_score(
        self,
        answer: str,
        expected_keywords: List[str],
        evaluation_criteria: List[str]
    ) -> float:
        """
        回答の正確性スコアを計算（1-5のスコア）

        注: 完全な自動評価は困難なため、キーワードマッチングベースの
        初期スコアを算出。最終的には人間による評価が必要。

        Args:
            answer: システムの回答
            expected_keywords: 期待されるキーワードリスト
            evaluation_criteria: 評価基準リスト

        Returns:
            1-5のスコア（浮動小数点）
        """
        if not answer or not expected_keywords:
            return 1.0

        answer_lower = answer.lower()

        # キーワードマッチング率を計算
        matched_keywords = sum(
            1 for keyword in expected_keywords
            if keyword.lower() in answer_lower
        )

        keyword_match_rate = matched_keywords / len(expected_keywords)

        # キーワードマッチ率をスコアに変換（1-5）
        # これは初期スコアで、人間による調整が必要
        if keyword_match_rate >= 0.8:
            base_score = 5.0
        elif keyword_match_rate >= 0.6:
            base_score = 4.0
        elif keyword_match_rate >= 0.4:
            base_score = 3.0
        elif keyword_match_rate >= 0.2:
            base_score = 2.0
        else:
            base_score = 1.0

        # 回答の長さによる調整（あまりに短すぎる回答はペナルティ）
        if len(answer) < 50:
            base_score = max(1.0, base_score - 1.0)

        return base_score

    def calculate_reference_validity(
        self,
        sources: List[Dict[str, Any]],
        expected_category: str
    ) -> float:
        """
        参照元の適切性を評価（0 or 1）

        Args:
            sources: 参照元のリスト
            expected_category: 期待されるカテゴリー

        Returns:
            1: 正しい参照元、0: 不適切な参照元
        """
        if not sources:
            return 0.0

        # カテゴリーが「全カテゴリー」の場合は常に有効
        if expected_category == "全カテゴリー":
            return 1.0

        # 参照元のカテゴリーが期待されるものと一致するかチェック
        for source in sources:
            # metadata内のcategoryをチェック
            if isinstance(source, dict):
                metadata = source.get("metadata", {})
                source_category = metadata.get("category", "")
                if source_category == expected_category:
                    return 1.0

        return 0.0

    def calculate_image_search_accuracy(
        self,
        images: List[str],
        expected_image: bool,
        expected_type: str
    ) -> float:
        """
        画像検索精度を評価（0, 0.5, or 1）

        Args:
            images: 検索された画像パスのリスト
            expected_image: 画像が期待されるか
            expected_type: 期待される画像タイプ（table, graph, etc.）

        Returns:
            1.0: 正しい画像、0.5: 関連する画像、0.0: 不適切または画像なし
        """
        # 画像が期待されているのに見つからない場合
        if expected_image and not images:
            return 0.0

        # 画像が期待されていないのに画像がある場合
        if not expected_image and images:
            return 0.5

        # 画像が期待されていて、かつ画像がある場合
        if expected_image and images:
            # 注: 実際の画像内容の評価は人間が行う必要がある
            # ここでは画像が存在することのみを確認
            return 1.0  # 初期スコア、人間による調整が必要

        # その他の場合
        return 1.0

    def calculate_numeric_extraction_accuracy(
        self,
        answer: str,
        ground_truth_numbers: Optional[List[float]] = None
    ) -> float:
        """
        数値データ抽出精度を評価（0, 0.5, or 1）

        Args:
            answer: システムの回答
            ground_truth_numbers: 正解となる数値リスト（オプション）

        Returns:
            1.0: 正確、0.5: 誤差あり、0.0: 大きく異なるor抽出失敗
        """
        # 回答から数値を抽出
        numbers = self._extract_numbers_from_text(answer)

        if not numbers:
            return 0.0

        # 正解データがない場合は、数値が抽出できたことのみを評価
        if ground_truth_numbers is None:
            return 1.0  # 初期スコア、人間による検証が必要

        # 正解データがある場合は、誤差を計算
        if len(numbers) != len(ground_truth_numbers):
            return 0.5

        # 各数値の誤差を計算
        errors = []
        for extracted, truth in zip(numbers, ground_truth_numbers):
            if truth == 0:
                continue
            error_rate = abs(extracted - truth) / truth
            errors.append(error_rate)

        if not errors:
            return 0.5

        avg_error = sum(errors) / len(errors)

        if avg_error <= 0.05:  # ±5%以内
            return 1.0
        elif avg_error <= 0.10:  # ±10%以内
            return 0.5
        else:
            return 0.0

    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """
        テキストから数値を抽出

        Args:
            text: 入力テキスト

        Returns:
            抽出された数値のリスト
        """
        # カンマ区切りの数値にも対応
        pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                # カンマを除去して浮動小数点数に変換
                num = float(match.replace(',', ''))
                numbers.append(num)
            except ValueError:
                continue

        return numbers

    def calculate_keyword_match_score(
        self,
        answer: str,
        expected_keywords: List[str]
    ) -> float:
        """
        キーワードマッチングスコアを計算

        Args:
            answer: システムの回答
            expected_keywords: 期待されるキーワードリスト

        Returns:
            0-1のスコア
        """
        if not expected_keywords:
            return 1.0

        answer_lower = answer.lower()

        matched = sum(
            1 for keyword in expected_keywords
            if keyword.lower() in answer_lower
        )

        return matched / len(expected_keywords)

    def verify_category_filter(
        self,
        sources: List[Dict[str, Any]],
        expected_category: str
    ) -> bool:
        """
        カテゴリーフィルタリングが正しく機能しているか検証

        Args:
            sources: 参照元のリスト
            expected_category: 期待されるカテゴリー

        Returns:
            True: 正しくフィルタリングされている
            False: 他のカテゴリーが混入している
        """
        if not sources:
            return False

        # 「全カテゴリー」の場合は常にTrue
        if expected_category == "全カテゴリー":
            return True

        # すべての参照元が期待されるカテゴリーかチェック
        for source in sources:
            if isinstance(source, dict):
                metadata = source.get("metadata", {})
                source_category = metadata.get("category", "")
                if source_category != expected_category:
                    return False

        return True

    def add_result(self, result: EvaluationResult):
        """評価結果を追加"""
        self.results.append(result)

    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        全体のサマリー統計を計算

        Returns:
            サマリー統計の辞書
        """
        if not self.results:
            return {}

        # 各メトリクスの平均を計算
        accuracy_scores = [
            r.accuracy_score for r in self.results
            if r.accuracy_score is not None
        ]

        reference_validities = [
            r.reference_validity for r in self.results
            if r.reference_validity is not None
        ]

        image_accuracies = [
            r.image_search_accuracy for r in self.results
            if r.image_search_accuracy is not None
        ]

        numeric_accuracies = [
            r.numeric_extraction_accuracy for r in self.results
            if r.numeric_extraction_accuracy is not None
        ]

        keyword_scores = [
            r.keyword_match_score for r in self.results
            if r.keyword_match_score is not None
        ]

        response_times = [r.response_time for r in self.results]

        category_filter_results = [
            r.category_filter_correct for r in self.results
            if r.category_filter_correct is not None
        ]

        return {
            "total_questions": len(self.results),
            "accuracy": {
                "average_score": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
                "normalized_score": (sum(accuracy_scores) / len(accuracy_scores) / 5.0) if accuracy_scores else 0,  # 0-1に正規化
                "count": len(accuracy_scores)
            },
            "reference_validity": {
                "average": sum(reference_validities) / len(reference_validities) if reference_validities else 0,
                "count": len(reference_validities)
            },
            "image_search": {
                "average_accuracy": sum(image_accuracies) / len(image_accuracies) if image_accuracies else 0,
                "count": len(image_accuracies)
            },
            "numeric_extraction": {
                "average_accuracy": sum(numeric_accuracies) / len(numeric_accuracies) if numeric_accuracies else 0,
                "count": len(numeric_accuracies)
            },
            "keyword_matching": {
                "average_score": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0,
                "count": len(keyword_scores)
            },
            "performance": {
                "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0
            },
            "category_filtering": {
                "accuracy": sum(category_filter_results) / len(category_filter_results) if category_filter_results else 0,
                "count": len(category_filter_results)
            },
            "errors": {
                "count": sum(1 for r in self.results if r.error is not None),
                "rate": sum(1 for r in self.results if r.error is not None) / len(self.results) if self.results else 0
            }
        }

    def calculate_category_specific_stats(self, question_type: str) -> Dict[str, Any]:
        """
        カテゴリー別の統計を計算

        Args:
            question_type: 質問タイプ（text, table, graph, hybrid, category）

        Returns:
            該当タイプの統計
        """
        # 質問IDのプレフィックスでフィルタリング
        prefix_map = {
            "text": "T",
            "table": "TB",
            "graph": "G",
            "hybrid": "H",
            "category": "C"
        }

        prefix = prefix_map.get(question_type, "")
        filtered_results = [
            r for r in self.results
            if r.question_id.startswith(prefix)
        ]

        if not filtered_results:
            return {}

        # 該当タイプのメトリクスを計算
        accuracy_scores = [
            r.accuracy_score for r in filtered_results
            if r.accuracy_score is not None
        ]

        return {
            "question_type": question_type,
            "count": len(filtered_results),
            "average_accuracy": (sum(accuracy_scores) / len(accuracy_scores) / 5.0) if accuracy_scores else 0,
            "average_response_time": sum(r.response_time for r in filtered_results) / len(filtered_results)
        }

    def check_success_criteria(self, summary: Dict[str, Any]) -> Dict[str, bool]:
        """
        成功基準を満たしているかチェック

        Args:
            summary: サマリー統計

        Returns:
            各基準の達成状況
        """
        # テキスト検索精度: 85%以上
        text_stats = self.calculate_category_specific_stats("text")
        text_passed = text_stats.get("average_accuracy", 0) >= 0.85

        # 表検索精度: 80%以上
        table_stats = self.calculate_category_specific_stats("table")
        table_passed = table_stats.get("average_accuracy", 0) >= 0.80

        # グラフ数値抽出精度: 70%以上
        graph_passed = summary.get("numeric_extraction", {}).get("average_accuracy", 0) >= 0.70

        # 応答時間: 15秒以内
        performance_passed = summary.get("performance", {}).get("average_response_time", float('inf')) <= 15.0

        # カテゴリーフィルタリング: 100%
        category_passed = summary.get("category_filtering", {}).get("accuracy", 0) >= 1.0

        return {
            "text_accuracy": text_passed,
            "table_accuracy": table_passed,
            "graph_numeric_accuracy": graph_passed,
            "response_time": performance_passed,
            "category_filtering": category_passed,
            "overall": all([text_passed, table_passed, graph_passed, performance_passed, category_passed])
        }
