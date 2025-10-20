"""
Category management module
"""

import json
import logging
from pathlib import Path
from typing import List, Set


logger = logging.getLogger(__name__)


class CategoryManager:
    """ドキュメントカテゴリーを管理するクラス"""

    def __init__(self, storage_file: str = "./data/categories.json"):
        """
        初期化

        Args:
            storage_file: カテゴリー一覧を保存するJSONファイルのパス
        """
        self.storage_file = Path(storage_file)
        self.categories: Set[str] = set()
        self._load_categories()

    def _load_categories(self):
        """保存されているカテゴリーを読み込み"""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.categories = set(data.get('categories', []))
                logger.info(f"Loaded {len(self.categories)} categories")
            except Exception as e:
                logger.error(f"Error loading categories: {e}")
                self.categories = set()
        else:
            logger.info("No existing categories file found, starting fresh")
            self.categories = set()

    def _save_categories(self):
        """カテゴリーをファイルに保存"""
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'categories': sorted(list(self.categories))
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.categories)} categories")
        except Exception as e:
            logger.error(f"Error saving categories: {e}")

    def add_category(self, category: str) -> bool:
        """
        新しいカテゴリーを追加

        Args:
            category: カテゴリー名

        Returns:
            bool: 新規追加された場合True、既存の場合False
        """
        category = category.strip()
        if not category:
            logger.warning("Empty category name provided")
            return False

        is_new = category not in self.categories
        self.categories.add(category)

        if is_new:
            self._save_categories()
            logger.info(f"Added new category: {category}")

        return is_new

    def get_all_categories(self) -> List[str]:
        """
        すべてのカテゴリーを取得

        Returns:
            list: ソート済みカテゴリーリスト
        """
        return sorted(list(self.categories))

    def category_exists(self, category: str) -> bool:
        """
        カテゴリーが存在するか確認

        Args:
            category: カテゴリー名

        Returns:
            bool: 存在する場合True
        """
        return category.strip() in self.categories

    def remove_category(self, category: str) -> bool:
        """
        カテゴリーを削除

        Args:
            category: カテゴリー名

        Returns:
            bool: 削除された場合True
        """
        category = category.strip()
        if category in self.categories:
            self.categories.remove(category)
            self._save_categories()
            logger.info(f"Removed category: {category}")
            return True
        return False
