"""
PDF processing module - テキストと画像の抽出
"""

import logging
import pdfplumber
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
import io
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDFからテキストと画像を抽出するクラス"""

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: PDF処理設定
        """
        self.config = config
        self.pdf_config = config.get("pdf_processing", {})
        self.vision_config = config.get("vision", {})

        # tiktokenエンコーダーの初期化
        model_name = config.get("openai", {}).get("model_chat", "gpt-5")
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # モデル名が見つからない場合はcl100k_base（GPT-4系）を使用
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _process_single_page(self, pdf_path: str, page_num: int, source_file: str, category: str) -> Dict[str, Any]:
        """
        単一ページを処理（並列処理用ヘルパー）

        Args:
            pdf_path: PDFファイルのパス
            page_num: ページ番号（1-indexed）
            source_file: ソースファイル名
            category: カテゴリー

        Returns:
            dict: ページの処理結果
        """
        page_result = {
            "text_chunks": [],
            "images": [],
            "table_markdowns": []
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]  # 0-indexed

                # テキスト抽出
                text = page.extract_text()

                # テキストが抽出できない、または非常に少ない場合
                # → 画像ベースPDFまたは特殊エンコーディングの可能性
                # 閾値を50→20文字に緩和（表やグラフ中心のPDFでもテキスト抽出を試みる）
                if not text or len(text.strip()) < 20:
                    logger.warning(f"Page {page_num}: テキスト抽出失敗またはテキスト量が少ない。ページ全体を画像として保存します。")

                    # ページ全体を画像として保存
                    pdf_name = Path(pdf_path).stem
                    output_dir = Path(f"data/extracted_images/{pdf_name}")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # ページ全体を画像化
                    max_size = self.vision_config.get("max_image_size", 2000)
                    im = page.to_image(resolution=150)
                    pil_image = im.original

                    # 画像をリサイズ（大きすぎる場合）
                    if max(pil_image.size) > max_size:
                        ratio = max_size / max(pil_image.size)
                        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                    # 画像を保存
                    image_filename = f"page_{page_num}_full.png"
                    image_path = output_dir / image_filename
                    self.save_extracted_image(pil_image, str(image_path))

                    # メタデータを作成
                    image_info = {
                        "image_path": str(image_path),
                        "page_number": page_num,
                        "image_index": 0,
                        "width": pil_image.size[0],
                        "height": pil_image.size[1],
                        "source_file": source_file,
                        "category": category,
                        "content_type": "full_page",  # ページ全体の画像として明示
                    }
                    page_result["images"].append(image_info)

                    logger.info(f"Page {page_num}: 全体画像として保存しました ({image_path})")
                else:
                    # テキストが抽出できた場合は通常処理
                    page_result["text_chunks"].extend(
                        self._create_text_chunks(text, page_num, source_file, category)
                    )

                    # 画像と表を抽出
                    page_data = self._extract_images_from_page(page, page_num, pdf_path)

                    # 画像データを追加
                    page_result["images"].extend(page_data["images"])

                    # テーブルMarkdownを追加
                    for table_markdown in page_data["table_markdowns"]:
                        page_result["table_markdowns"].append({
                            "text": table_markdown["text"],
                            "page_number": table_markdown["page_number"],
                            "source_file": source_file,
                            "category": category,
                            "content_type": "table_markdown"
                        })

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            # ページ処理失敗時もエラーを投げずに空の結果を返す

        return page_result

    def process_pdf(self, pdf_path: str, category: str) -> Dict[str, Any]:
        """
        PDFを処理してテキストと画像を抽出（並列処理）

        Args:
            pdf_path: PDFファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果（テキストチャンク、画像情報など）
        """
        logger.info(f"Processing PDF: {pdf_path}")

        result = {
            "source_file": Path(pdf_path).name,
            "category": category,
            "text_chunks": [],
            "images": [],
            "metadata": {},
        }

        try:
            # ページ数を取得
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                result["metadata"]["page_count"] = page_count

            # 並列処理のワーカー数を設定
            max_workers = min(4, page_count)  # 最大4スレッド
            logger.info(f"Processing {page_count} pages with {max_workers} workers")

            # ページを並列処理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 各ページの処理をサブミット
                futures = {
                    executor.submit(self._process_single_page, pdf_path, page_num, result["source_file"], category): page_num
                    for page_num in range(1, page_count + 1)
                }

                # 結果を収集（ページ順序を保持）
                page_results = {}
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_result = future.result()
                        page_results[page_num] = page_result
                        logger.info(f"Completed processing page {page_num}/{page_count}")
                    except Exception as e:
                        logger.error(f"Error in page {page_num}: {e}")
                        page_results[page_num] = {"text_chunks": [], "images": [], "table_markdowns": []}

                # ページ順にソートして結果をマージ
                for page_num in sorted(page_results.keys()):
                    page_result = page_results[page_num]
                    result["text_chunks"].extend(page_result["text_chunks"])
                    result["images"].extend(page_result["images"])
                    result["text_chunks"].extend(page_result["table_markdowns"])

            logger.info(f"Extracted {len(result['text_chunks'])} text chunks and {len(result['images'])} images")

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

        return result

    def _create_text_chunks(self, text: str, page_num: int, source_file: str, category: str) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割（tiktoken使用）

        Args:
            text: ページから抽出したテキスト
            page_num: ページ番号
            source_file: ソースファイル名
            category: カテゴリー

        Returns:
            list: テキストチャンクのリスト
        """
        chunk_size = self.pdf_config.get("chunk_size", 800)
        chunk_overlap = self.pdf_config.get("chunk_overlap", 150)
        chunks = []

        # テキストをトークン化
        tokens = self.encoding.encode(text)

        # トークン数が少ない場合はそのまま返す
        if len(tokens) <= chunk_size:
            return [{"text": text.strip(), "page_number": page_num, "source_file": source_file, "category": category}]

        # チャンクに分割
        start = 0
        chunk_index = 0

        while start < len(tokens):
            # チャンクの終了位置を計算
            end = min(start + chunk_size, len(tokens))

            # トークンをデコードしてテキストに戻す
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # チャンクを追加
            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page_number": page_num,
                        "source_file": source_file,
                        "category": category,
                        "chunk_index": chunk_index,
                        "token_count": len(chunk_tokens),
                    }
                )
                chunk_index += 1

            # 次のチャンクの開始位置（オーバーラップを考慮）
            start = end - chunk_overlap if end < len(tokens) else end

        logger.debug(f"Created {len(chunks)} chunks from page {page_num} ({len(tokens)} tokens)")
        return chunks

    def _is_complex_table(self, table_data: List[List]) -> bool:
        """
        表の複雑度を判定

        Args:
            table_data: pdfplumberのtable.extract()で取得したデータ

        Returns:
            bool: 複雑な表の場合True
        """
        if not table_data:
            return False

        threshold = self.pdf_config.get("complex_table_threshold", 10)

        # 行数と列数を取得
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        total_cells = rows * cols

        # セル数が閾値を超えたら複雑と判定
        if total_cells > threshold:
            return True

        # 結合セル（None値）が多い場合も複雑と判定
        none_count = sum(1 for row in table_data for cell in row if cell is None)
        if none_count > total_cells * 0.2:  # 20%以上がNoneなら複雑
            return True

        return False

    def _table_to_markdown(self, table_data: List[List]) -> str:
        """
        表データをMarkdown形式に変換

        Args:
            table_data: pdfplumberのtable.extract()で取得したデータ

        Returns:
            str: Markdown形式の表
        """
        if not table_data:
            return ""

        # Noneを空文字列に置換
        cleaned_data = [[str(cell or "") for cell in row] for row in table_data]

        # Markdownテーブルを構築
        markdown_lines = []

        # ヘッダー行
        if cleaned_data:
            header = "| " + " | ".join(cleaned_data[0]) + " |"
            markdown_lines.append(header)

            # セパレーター
            separator = "| " + " | ".join(["---"] * len(cleaned_data[0])) + " |"
            markdown_lines.append(separator)

            # データ行
            for row in cleaned_data[1:]:
                data_row = "| " + " | ".join(row) + " |"
                markdown_lines.append(data_row)

        return "\n".join(markdown_lines)

    def _extract_images_from_page(self, page, page_num: int, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        ページから画像を抽出（表とその他の画像）

        Args:
            page: pdfplumberのページオブジェクト
            page_num: ページ番号
            pdf_path: PDFファイルのパス

        Returns:
            dict: {"images": [...], "table_markdowns": [...]}
        """
        images = []
        table_markdowns = []
        min_size = self.pdf_config.get("min_image_size", 100)
        max_size = self.vision_config.get("max_image_size", 2000)
        pdf_name = Path(pdf_path).stem
        output_dir = Path(f"data/extracted_images/{pdf_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 設定フラグを取得
        extract_tables = self.pdf_config.get("extract_tables_as_markdown", True)
        extract_images_flag = self.pdf_config.get("extract_images", True)

        try:
            # 1. 表を検出（最適化設定を適用）
            # 表抽出が無効の場合はスキップ
            if not extract_tables:
                logger.debug(f"Skipping table extraction on page {page_num} (disabled in config)")
                tables = []
            else:
                strategy = self.pdf_config.get("table_detection_strategy", "fast")

                if strategy == "fast":
                    # 高速モード: 精度を少し犠牲にして処理速度を優先
                    table_settings = {
                        "vertical_strategy": "lines",      # 垂直線のみ検出
                        "horizontal_strategy": "lines",    # 水平線のみ検出
                        "snap_tolerance": 5,               # スナップ許容値を緩和
                        "join_tolerance": 5,               # 結合許容値を緩和
                        "edge_min_length": 10,             # 最小エッジ長を設定（短い線を無視）
                        "min_words_vertical": 1,           # 最小単語数を削減
                        "min_words_horizontal": 1,
                    }
                    tables = page.find_tables(table_settings=table_settings)
                else:
                    # 高精度モード: デフォルト設定で詳細検出
                    tables = page.find_tables()

            if tables:
                logger.debug(f"Found {len(tables)} tables on page {page_num}")

                for table_idx, table in enumerate(tables):
                    try:
                        # 表データを抽出
                        table_data = table.extract()

                        # 表の複雑度を判定
                        is_complex = self._is_complex_table(table_data)

                        if is_complex:
                            # 複雑な表 → 画像として保存
                            logger.debug(f"Complex table detected on page {page_num}, saving as image")

                            # 表のバウンディングボックスを取得
                            bbox = table.bbox  # (x0, top, x1, bottom)
                            x0, y0, x1, y1 = bbox

                            # ページのサイズを取得
                            page_width = page.width
                            page_height = page.height

                            # バウンディングボックスをページ境界内にクリップ
                            x0 = max(0, x0)
                            y0 = max(0, y0)
                            x1 = min(page_width, x1)
                            y1 = min(page_height, y1)

                            clipped_bbox = (x0, y0, x1, y1)
                            width = x1 - x0
                            height = y1 - y0

                            # サイズフィルタリング
                            if width < min_size or height < min_size:
                                logger.debug(f"Skipping small table: {width}x{height}")
                                continue

                            # 表の領域を画像として切り取り
                            im = page.crop(clipped_bbox).to_image(resolution=150)
                            pil_image = im.original

                            # 画像をリサイズ（大きすぎる場合）
                            if max(pil_image.size) > max_size:
                                ratio = max_size / max(pil_image.size)
                                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                            # 画像を保存
                            image_filename = f"page_{page_num}_table_{table_idx}.png"
                            image_path = output_dir / image_filename
                            self.save_extracted_image(pil_image, str(image_path))

                            # メタデータを作成
                            image_info = {
                                "image_path": str(image_path),
                                "page_number": page_num,
                                "image_index": table_idx,
                                "width": pil_image.size[0],
                                "height": pil_image.size[1],
                                "source_file": Path(pdf_path).name,
                                "category": None,  # 後で設定される
                                "content_type": "table",  # 表として明示的にマーク
                            }
                            images.append(image_info)

                            logger.debug(f"Extracted complex table {table_idx} as image from page {page_num}")

                        else:
                            # シンプルな表 → Markdownに変換
                            logger.debug(f"Simple table detected on page {page_num}, converting to Markdown")
                            markdown = self._table_to_markdown(table_data)

                            if markdown:
                                table_markdowns.append({
                                    "text": f"\n[表 {table_idx + 1}]\n{markdown}\n",
                                    "page_number": page_num,
                                })
                                logger.debug(f"Converted simple table {table_idx} to Markdown on page {page_num}")

                    except Exception as e:
                        logger.warning(f"Failed to process table {table_idx} from page {page_num}: {e}")
                        continue

            # 2. その他の画像オブジェクトを抽出（グラフや図など）
            # 画像抽出が無効の場合はスキップ
            if not extract_images_flag:
                logger.debug(f"Skipping image extraction on page {page_num} (disabled in config)")
            elif hasattr(page, "images") and page.images:
                logger.debug(f"Found {len(page.images)} image objects on page {page_num}")
                margin = self.pdf_config.get("image_crop_margin", 50)  # デフォルト50px

                for img_idx, img in enumerate(page.images):
                    try:
                        # 画像のバウンディングボックス情報
                        x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                        width = x1 - x0
                        height = y1 - y0

                        # サイズフィルタリング
                        if width < min_size or height < min_size:
                            logger.debug(f"Skipping small image: {width}x{height}")
                            continue

                        # 表の領域と重複していないかチェック
                        is_in_table = False
                        for table in tables:
                            tx0, ty0, tx1, ty1 = table.bbox
                            # 画像が表の中にあるかチェック
                            if x0 >= tx0 and x1 <= tx1 and y0 >= ty0 and y1 <= ty1:
                                is_in_table = True
                                break

                        if is_in_table:
                            logger.debug(f"Skipping image {img_idx} (inside table)")
                            continue

                        # マージンを追加してバウンディングボックスを拡大
                        page_width = page.width
                        page_height = page.height

                        expanded_x0 = max(0, x0 - margin)
                        expanded_y0 = max(0, y0 - margin)
                        expanded_x1 = min(page_width, x1 + margin)
                        expanded_y1 = min(page_height, y1 + margin)

                        logger.debug(f"Expanding image bbox from ({x0},{y0},{x1},{y1}) to ({expanded_x0},{expanded_y0},{expanded_x1},{expanded_y1})")

                        # ページを画像として切り取り（拡大されたバウンディングボックス使用）
                        im = page.crop((expanded_x0, expanded_y0, expanded_x1, expanded_y1)).to_image(resolution=150)
                        pil_image = im.original

                        # 画像をリサイズ（大きすぎる場合）
                        if max(pil_image.size) > max_size:
                            ratio = max_size / max(pil_image.size)
                            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                        # 画像を保存
                        image_filename = f"page_{page_num}_img_{img_idx}.png"
                        image_path = output_dir / image_filename
                        self.save_extracted_image(pil_image, str(image_path))

                        # メタデータを作成
                        image_info = {
                            "image_path": str(image_path),
                            "page_number": page_num,
                            "image_index": img_idx,
                            "width": pil_image.size[0],
                            "height": pil_image.size[1],
                            "source_file": Path(pdf_path).name,
                            "category": None,  # 後で設定される
                            "content_type": "image",  # その他の画像
                        }
                        images.append(image_info)

                        logger.debug(f"Extracted image {img_idx} from page {page_num}")

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
                        continue

            logger.info(f"Extracted {len(images)} images and {len(table_markdowns)} markdown tables from page {page_num}")

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")

        return {"images": images, "table_markdowns": table_markdowns}

    def save_extracted_image(self, image: Image.Image, output_path: str) -> bool:
        """
        抽出した画像を保存

        Args:
            image: PIL Image
            output_path: 保存先パス

        Returns:
            bool: 成功時True
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format=self.vision_config.get("image_format", "PNG"))
            logger.info(f"Saved image: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
