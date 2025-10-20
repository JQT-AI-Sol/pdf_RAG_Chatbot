"""
既存のデータをクリアするスクリプト
"""

import shutil
from pathlib import Path

# 削除するディレクトリ
dirs_to_clean = [
    "data/uploaded_pdfs",
    "data/extracted_images",
    "data/chroma_db",
]

# ファイル
files_to_clean = [
    "data/categories.json"
]

print("データをクリアします...\n")

# ディレクトリを削除
for dir_path in dirs_to_clean:
    p = Path(dir_path)
    if p.exists():
        shutil.rmtree(p)
        print(f"[OK] 削除: {dir_path}")
    else:
        print(f"[SKIP] スキップ: {dir_path} (存在しません)")

# ファイルを削除
for file_path in files_to_clean:
    p = Path(file_path)
    if p.exists():
        p.unlink()
        print(f"[OK] 削除: {file_path}")
    else:
        print(f"[SKIP] スキップ: {file_path} (存在しません)")

print("\nデータのクリアが完了しました！")
print("ブラウザでアプリをリロードして、再度PDFをアップロードしてください。")
