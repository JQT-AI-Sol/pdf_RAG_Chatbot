"""
表抽出のデバッグスクリプト
"""
import pdfplumber
from pathlib import Path

pdf_path = "data/uploaded_pdfs/予防接種後健康被害救済制度.pdf"

print(f"Processing: {pdf_path}\n")

with pdfplumber.open(pdf_path) as pdf:
    # ページ2を確認（問題のページ）
    page = pdf.pages[1]  # 0-indexed, so page 2 is index 1

    print(f"Page 2 - Total pages: {len(pdf.pages)}\n")

    # 表を検出
    tables = page.find_tables()

    print(f"Found {len(tables)} tables on page 2\n")

    for idx, table in enumerate(tables):
        print(f"=== Table {idx + 1} ===")
        table_data = table.extract()

        if table_data:
            print(f"Rows: {len(table_data)}, Columns: {len(table_data[0]) if table_data else 0}")
            print("\nTable data:")
            for row_idx, row in enumerate(table_data[:5]):  # 最初の5行のみ表示
                print(f"Row {row_idx}: {row}")

            if len(table_data) > 5:
                print(f"... (showing first 5 of {len(table_data)} rows)")
        else:
            print("No data extracted")

        print()

    # ページ全体のテキストも確認
    print("=== Page 2 Text (first 500 chars) ===")
    text = page.extract_text()
    if text:
        print(text[:500])
    else:
        print("No text extracted")
