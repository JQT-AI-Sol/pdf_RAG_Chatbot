"""Clear Python cache files"""
import os
import shutil
from pathlib import Path

def clear_pycache():
    """__pycache__ディレクトリを削除"""
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = Path(root) / '__pycache__'
            try:
                shutil.rmtree(cache_dir)
                print(f"Deleted: {cache_dir}")
                count += 1
            except Exception as e:
                print(f"Failed to delete {cache_dir}: {e}")

    print(f"\nCleared {count} cache directories")

if __name__ == "__main__":
    clear_pycache()
