#!/usr/bin/env python
"""PDFとその関連データを削除するスクリプト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_manager import PDFManager
from src.vector_store import VectorStore
from src.category_manager import CategoryManager
import yaml
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 設定を読み込み
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 各コンポーネントを初期化
    vector_store = VectorStore(config)
    category_manager = CategoryManager()
    pdf_manager = PDFManager(vector_store, category_manager, config)

    # 削除対象のPDF
    target_file = "000868397.pdf"

    print(f"削除対象: {target_file}")
    print("削除を実行します...")

    # PDFを削除
    result = pdf_manager.delete_pdf(target_file)

    if result['success']:
        print("\n✅ 削除が完了しました！")
        print(f"- テキストチャンク削除: {result['vector_deleted']['text_deleted']}件")
        print(f"- 画像コンテンツ削除: {result['vector_deleted']['image_deleted']}件")
        print(f"- PDFファイル削除: {'成功' if result['pdf_file_deleted'] else '失敗'}")
        print(f"- 画像ディレクトリ削除: {'成功' if result['images_deleted'] else '失敗'}")
        print(f"- カテゴリー削除: {'成功' if result['category_deleted'] else 'スキップ（他のPDFが存在）'}")
        print(f"\nメッセージ: {result['message']}")
    else:
        print(f"\n❌ 削除に失敗しました: {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
