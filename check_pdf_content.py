"""
PDFファイルの内容を確認するスクリプト
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF

pdf_path = r"c:\Users\r0u8b\OneDrive\一時保管場所（削除してよいもの）\キャリアアップ助成金.pdf"

print(f"PDF Path: {pdf_path}")
print(f"Exists: {Path(pdf_path).exists()}\n")

if not Path(pdf_path).exists():
    print("PDFファイルが見つかりません")
    sys.exit(1)

# PDFを開く
doc = fitz.open(pdf_path)

print(f"総ページ数: {len(doc)}\n")

# Page 2を確認
if len(doc) >= 2:
    page = doc[1]  # 0-indexed

    print("=" * 80)
    print("Page 2 のテキスト抽出結果:")
    print("=" * 80)

    text = page.get_text()
    print(text)

    print("\n" + "=" * 80)
    print("Page 2 の画像数:")
    print("=" * 80)

    images = page.get_images()
    print(f"画像数: {len(images)}")

    print("\n" + "=" * 80)
    print("Page 2 の表検出:")
    print("=" * 80)

    # 表を検出
    tables = page.find_tables()
    print(f"検出された表の数: {len(tables)}")

    if len(tables) > 0:
        for i, table in enumerate(tables):
            print(f"\n--- 表 {i+1} ---")
            print(f"行数: {table.row_count}")
            print(f"列数: {table.col_count}")

            # 表データを抽出
            try:
                df = table.to_pandas()
                print("\n表の内容:")
                print(df.to_string())
            except Exception as e:
                print(f"表データの抽出エラー: {e}")

doc.close()
