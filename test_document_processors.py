"""
Document processors test script
各ドキュメントプロセッサの動作確認用テストスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.document_processor import DocumentProcessor
from src.txt_processor import TextFileProcessor
from src.word_processor import WordProcessor
from src.excel_processor import ExcelProcessor
from src.pptx_processor import PowerPointProcessor
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_text_processor():
    """テキストファイルプロセッサのテスト"""
    print("\n" + "="*80)
    print("[TEST] テキストファイルプロセッサのテスト")
    print("="*80)

    try:
        config = load_config()
        processor = TextFileProcessor(config)

        # サンプルテキストファイルを処理
        txt_path = "test_data/sample.txt"
        if not Path(txt_path).exists():
            print(f"[ERROR] テストファイルが見つかりません: {txt_path}")
            return False

        result = processor.process_text_file(txt_path, category="テスト")

        print(f"[OK] ファイル処理成功: {result['source_file']}")
        print(f"   - テキストチャンク数: {len(result['text_chunks'])}")
        print(f"   - エンコーディング: {result['metadata'].get('encoding', 'N/A')}")
        print(f"   - 文字数: {result['metadata'].get('character_count', 'N/A')}")
        print(f"   - 行数: {result['metadata'].get('line_count', 'N/A')}")

        if result['text_chunks']:
            print(f"\n最初のチャンク（抜粋）:")
            print(f"   {result['text_chunks'][0]['text'][:100]}...")

        return True

    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        logger.exception("Test failed")
        return False


def test_document_processor():
    """統合ドキュメントプロセッサのテスト"""
    print("\n" + "="*80)
    print("[TEST] 統合ドキュメントプロセッサのテスト")
    print("="*80)

    try:
        config = load_config()
        processor = DocumentProcessor(config)

        # サポートする拡張子を表示
        print(f"サポートする拡張子:")
        for ext, file_type in processor.SUPPORTED_EXTENSIONS.items():
            print(f"   {ext} → {file_type}")

        # テキストファイルを処理
        txt_path = "test_data/sample.txt"
        if Path(txt_path).exists():
            result = processor.process_document(txt_path, category="統合テスト")
            print(f"\n[OK] 統合処理成功: {result['source_file']}")
            print(f"   - ファイルタイプ: {result['metadata']['file_type']}")
            print(f"   - チャンク数: {len(result['text_chunks'])}")
        else:
            print(f"[WARN] テストファイルが見つかりません: {txt_path}")

        return True

    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        logger.exception("Test failed")
        return False


def test_file_type_detection():
    """ファイルタイプ検出のテスト"""
    print("\n" + "="*80)
    print("[TEST] ファイルタイプ検出のテスト")
    print("="*80)

    try:
        config = load_config()
        processor = DocumentProcessor(config)

        test_files = {
            "test.pdf": "pdf",
            "test.docx": "word",
            "test.doc": "word",
            "test.xlsx": "excel",
            "test.xls": "excel",
            "test.pptx": "powerpoint",
            "test.ppt": "powerpoint",
            "test.txt": "text",
            "test.unknown": None,
        }

        all_passed = True
        for filename, expected_type in test_files.items():
            detected_type = processor.get_file_type(filename)
            status = "[OK]" if detected_type == expected_type else "[ERROR]"
            print(f"{status} {filename:20} → {detected_type} (期待値: {expected_type})")
            if detected_type != expected_type:
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        logger.exception("Test failed")
        return False


def main():
    """メインテスト実行"""
    print("\n" + "="*80)
    print("[START] ドキュメントプロセッサテスト開始")
    print("="*80)

    results = {
        "ファイルタイプ検出": test_file_type_detection(),
        "テキストプロセッサ": test_text_processor(),
        "統合プロセッサ": test_document_processor(),
    }

    print("\n" + "="*80)
    print("[SUMMARY] テスト結果サマリー")
    print("="*80)

    for test_name, result in results.items():
        status = "[OK] 成功" if result else "[ERROR] 失敗"
        print(f"{test_name:30} {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("[OK] 全てのテストが成功しました！")
    else:
        print("[ERROR] 一部のテストが失敗しました。")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
