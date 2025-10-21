"""
既存のベクトルストアから実際のドキュメント内容に基づいた
テストケースを自動生成するスクリプト
"""

import sys
from pathlib import Path
import yaml
import json
from collections import defaultdict

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.utils import load_config


def analyze_documents(vector_store: VectorStore):
    """
    ベクトルストアの内容を分析

    Returns:
        dict: 分析結果
    """
    analysis = {
        'categories': defaultdict(int),
        'files': defaultdict(list),
        'content_types': defaultdict(int),
        'sample_texts': [],
        'sample_images': []
    }

    # テキストコレクションを分析
    text_results = vector_store.text_collection.get(include=['metadatas', 'documents'])

    for doc, meta in zip(text_results['documents'], text_results['metadatas']):
        category = meta.get('category', 'Unknown')
        file_name = meta.get('source_file', 'Unknown')

        analysis['categories'][category] += 1
        analysis['files'][category].append(file_name)

        # サンプルテキストを保存（最大20件）
        if len(analysis['sample_texts']) < 20:
            analysis['sample_texts'].append({
                'category': category,
                'file': file_name,
                'page': meta.get('page_number', '?'),
                'content': doc[:200]  # 先頭200文字
            })

    # 画像コレクションを分析
    image_results = vector_store.image_collection.get(include=['metadatas', 'documents'])

    for doc, meta in zip(image_results['documents'], image_results['metadatas']):
        content_type = meta.get('content_type', 'Unknown')
        category = meta.get('category', 'Unknown')

        analysis['content_types'][content_type] += 1

        # サンプル画像情報を保存（最大10件）
        if len(analysis['sample_images']) < 10:
            analysis['sample_images'].append({
                'category': category,
                'file': meta.get('source_file', 'Unknown'),
                'page': meta.get('page_number', '?'),
                'content_type': content_type,
                'description': doc[:200]  # 先頭200文字
            })

    # ファイルリストをユニーク化
    for category in analysis['files']:
        analysis['files'][category] = list(set(analysis['files'][category]))

    return analysis


def generate_test_questions(analysis: dict) -> dict:
    """
    分析結果に基づいてテストケースを生成

    Args:
        analysis: analyze_documents()の結果

    Returns:
        dict: test_questions.yaml形式のテストケース
    """
    categories = list(analysis['categories'].keys())

    # テキスト検索用の質問（実際のドキュメント内容に基づく）
    text_search_questions = []

    # サンプルテキストから質問を生成
    for i, sample in enumerate(analysis['sample_texts'][:10], 1):
        # 内容から重要なキーワードを抽出（簡易版）
        content = sample['content']
        keywords = extract_keywords(content)

        text_search_questions.append({
            'id': f'T{i:02d}',
            'question': generate_question_from_content(content),
            'category': sample['category'],
            'expected_type': 'text',
            'evaluation_criteria': [
                f'「{sample["file"]}」の情報が含まれているか',
                '具体的な説明が含まれているか'
            ],
            'expected_keywords': keywords[:5],  # 最大5個のキーワード
            'accuracy_weight': 1.0
        })

    # 表検索用の質問
    table_search_questions = []
    table_samples = [s for s in analysis['sample_images'] if 'table' in s.get('content_type', '').lower()]

    for i, sample in enumerate(table_samples[:5], 1):
        table_search_questions.append({
            'id': f'TB{i:02d}',
            'question': f'{sample["category"]}に関する表を見せてください',
            'category': sample['category'],
            'expected_type': 'table',
            'evaluation_criteria': [
                '表の画像が表示されているか',
                '表の内容が適切に解析されているか'
            ],
            'expected_image': True,
            'expected_keywords': ['表', sample['category']],
            'accuracy_weight': 1.0
        })

    # グラフ検索用の質問
    graph_search_questions = []
    graph_samples = [s for s in analysis['sample_images'] if 'graph' in s.get('description', '').lower()]

    for i, sample in enumerate(graph_samples[:5], 1):
        graph_search_questions.append({
            'id': f'G{i:02d}',
            'question': f'{sample["category"]}のグラフやデータを見せてください',
            'category': sample['category'],
            'expected_type': 'graph',
            'evaluation_criteria': [
                'グラフが表示されているか',
                'データの傾向が説明されているか'
            ],
            'expected_image': True,
            'expected_keywords': ['グラフ', 'データ', sample['category']],
            'requires_numeric_data': True,
            'accuracy_weight': 1.0
        })

    # ハイブリッド検索用の質問
    hybrid_search_questions = []

    for i, category in enumerate(categories[:5], 1):
        hybrid_search_questions.append({
            'id': f'H{i:02d}',
            'question': f'{category}に関する全体的な情報とデータを教えてください',
            'category': '全カテゴリー',
            'expected_type': 'hybrid',
            'evaluation_criteria': [
                'テキスト説明と図表の両方が含まれているか',
                '包括的な情報が提供されているか'
            ],
            'expected_image': True,
            'expected_keywords': [category, '情報', 'データ'],
            'accuracy_weight': 1.0
        })

    # カテゴリーフィルタリング用の質問
    category_filtering_questions = []

    for i, category in enumerate(categories, 1):
        category_filtering_questions.append({
            'id': f'C{i:02d}',
            'question': 'このカテゴリーの概要を教えてください',
            'category': category,
            'expected_type': 'text',
            'evaluation_criteria': [
                f'「{category}」カテゴリーのみから検索されているか',
                '他カテゴリーの情報が混入していないか',
                '参照元に正しいカテゴリーが表示されているか'
            ],
            'expected_keywords': [category],
            'verify_category_filter': True,
            'accuracy_weight': 1.0
        })

    # 「全カテゴリー」での検索も追加
    category_filtering_questions.append({
        'id': f'C{len(categories) + 1:02d}',
        'question': 'このシステムに登録されている全ての情報の概要を教えてください',
        'category': '全カテゴリー',
        'expected_type': 'text',
        'evaluation_criteria': [
            '複数カテゴリーから情報が統合されているか',
            '包括的な回答が提供されているか'
        ],
        'expected_keywords': ['情報', '概要'],
        'verify_category_filter': False,
        'accuracy_weight': 1.0
    })

    # YAML形式で出力
    test_data = {
        'text_search': text_search_questions,
        'table_search': table_search_questions,
        'graph_search': graph_search_questions,
        'hybrid_search': hybrid_search_questions,
        'category_filtering': category_filtering_questions,
        'evaluation_settings': {
            'accuracy_scores': {
                5: '完全に正確で完璧な回答',
                4: 'ほぼ正確だが小さな不足あり',
                3: '部分的に正確だが重要な情報が欠落',
                2: '不正確な情報が含まれる',
                1: '完全に不正確または無関連'
            },
            'reference_validity': {
                1: '正しい参照元を提示',
                0: '不適切な参照元'
            },
            'image_search_accuracy': {
                1.0: '正しい表/グラフを表示',
                0.5: '関連するが最適でない画像',
                0.0: '不適切な画像または画像なし'
            },
            'numeric_extraction_accuracy': {
                1.0: '数値が正確（±5%以内）',
                0.5: '数値に誤差あり（±10%以内）',
                0.0: '数値が大きく異なるor抽出失敗'
            },
            'performance_requirements': {
                'max_response_time_seconds': 15,
                'max_index_time_per_page_seconds': 20
            },
            'success_criteria': {
                'text_accuracy_threshold': 0.85,
                'table_accuracy_threshold': 0.80,
                'graph_numeric_accuracy_threshold': 0.70,
                'category_filter_accuracy': 1.00
            }
        }
    }

    return test_data


def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    テキストから重要なキーワードを抽出（簡易版）

    Args:
        text: 入力テキスト
        max_keywords: 最大キーワード数

    Returns:
        list: キーワードリスト
    """
    # 簡易的な実装：頻出する名詞的な表現を抽出
    # （本来はMeCabなどの形態素解析を使うべき）

    # 除外する一般的な語
    stop_words = {'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
                  'ある', 'いる', 'も', 'する', 'から', 'な', 'こ', 'として', 'い', 'や',
                  'れる', 'など', 'ない', 'この', 'ため', 'その', 'これ', 'それ', 'あり',
                  'より', 'また', '及び', 'において', 'による', 'ついて', 'おける', 'ます'}

    # 単語分割（簡易版）
    words = []
    for line in text.split('\n'):
        for word in line.split():
            word = word.strip('、。！？()（）「」『』【】')
            if len(word) > 1 and word not in stop_words:
                words.append(word)

    # 頻度カウント
    from collections import Counter
    word_freq = Counter(words)

    # 上位キーワードを返す
    return [word for word, _ in word_freq.most_common(max_keywords)]


def generate_question_from_content(content: str) -> str:
    """
    コンテンツから質問文を生成

    Args:
        content: ドキュメントの内容

    Returns:
        str: 質問文
    """
    # 簡易的な質問生成
    keywords = extract_keywords(content, max_keywords=3)

    if len(keywords) >= 2:
        return f'{keywords[0]}に関する{keywords[1]}について教えてください'
    elif len(keywords) >= 1:
        return f'{keywords[0]}について詳しく教えてください'
    else:
        return 'この内容について詳しく教えてください'


def main():
    """メイン処理"""
    print("="*80)
    print("テストケース自動生成ツール")
    print("="*80)
    print()

    # 設定読み込み
    print("[1/4] 設定を読み込み中...")
    config = load_config()

    # ベクトルストア初期化
    print("[2/4] ベクトルストアを読み込み中...")
    vector_store = VectorStore(config)

    print(f"  - テキストドキュメント数: {vector_store.text_collection.count()}")
    print(f"  - 画像ドキュメント数: {vector_store.image_collection.count()}")

    # ドキュメント分析
    print("\n[3/4] ドキュメントを分析中...")
    analysis = analyze_documents(vector_store)

    print(f"  - カテゴリー数: {len(analysis['categories'])}")
    for category, count in analysis['categories'].items():
        print(f"    - {category}: {count}件")

    print(f"  - ファイル数: {sum(len(files) for files in analysis['files'].values())}")
    print(f"  - 画像タイプ: {dict(analysis['content_types'])}")

    # テストケース生成
    print("\n[4/4] テストケースを生成中...")
    test_data = generate_test_questions(analysis)

    print(f"  - テキスト検索: {len(test_data['text_search'])}問")
    print(f"  - 表検索: {len(test_data['table_search'])}問")
    print(f"  - グラフ検索: {len(test_data['graph_search'])}問")
    print(f"  - ハイブリッド検索: {len(test_data['hybrid_search'])}問")
    print(f"  - カテゴリーフィルタリング: {len(test_data['category_filtering'])}問")
    print(f"  - 合計: {sum(len(v) for k, v in test_data.items() if k != 'evaluation_settings')}問")

    # YAML形式で保存
    output_path = 'evaluation/test_questions_generated.yaml'
    print(f"\n[完了] テストケースを保存中: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 分析結果もJSON形式で保存
    analysis_path = 'evaluation/document_analysis.json'
    print(f"[完了] 分析結果を保存中: {analysis_path}")

    # defaultdictを通常のdictに変換
    analysis_dict = {
        'categories': dict(analysis['categories']),
        'files': {k: v for k, v in analysis['files'].items()},
        'content_types': dict(analysis['content_types']),
        'sample_texts': analysis['sample_texts'],
        'sample_images': analysis['sample_images']
    }

    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_dict, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("テストケースの生成が完了しました！")
    print("="*80)
    print(f"\n生成されたファイル:")
    print(f"  1. {output_path}")
    print(f"  2. {analysis_path}")
    print(f"\n次のステップ:")
    print(f"  1. {output_path} を確認して質問を調整")
    print(f"  2. evaluation/run_evaluation.py --questions {output_path} で評価実行")


if __name__ == "__main__":
    main()
