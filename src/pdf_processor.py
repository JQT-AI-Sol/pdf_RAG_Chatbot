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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        self.rag_config = config.get("rag", {})
        self.chunking_config = config.get("chunking", {})

        # tiktokenエンコーダーの初期化
        model_name = config.get("openai", {}).get("model_chat", "gpt-5")
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # モデル名が見つからない場合はcl100k_base（GPT-4系）を使用
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # セマンティックチャンカーの初期化
        self.text_splitter = None
        if self.rag_config.get("enable_semantic_chunking", False):
            try:
                chunk_size = self.chunking_config.get("chunk_size", self.pdf_config.get("chunk_size", 800))
                chunk_overlap = self.chunking_config.get("chunk_overlap", self.pdf_config.get("chunk_overlap", 150))
                separators = self.chunking_config.get("separators", ["\n\n", "\n", "。", "．", ". ", "! ", "? ", "；", "、", ", ", " ", ""])

                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=lambda text: len(self.encoding.encode(text)),
                    separators=separators,
                    keep_separator=True
                )
                logger.info(f"Semantic chunking initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
            except Exception as e:
                logger.error(f"Failed to initialize semantic chunker: {e}")
                logger.warning("Falling back to legacy chunking")

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

                # 画像と表を先に抽出（表の領域を把握するため）
                page_data = self._extract_images_from_page(page, page_num, pdf_path)

                # 表の領域を除外したページテキストを抽出
                filtered_page = page
                table_bboxes = page_data.get("table_bboxes", [])

                if table_bboxes and self.pdf_config.get("exclude_tables_from_page_text", True):
                    # 表の領域をページから除外
                    try:
                        for bbox in table_bboxes:
                            filtered_page = filtered_page.outside_bbox(bbox)
                        text = filtered_page.extract_text() or ""
                        logger.debug(f"Page {page_num}: Excluded {len(table_bboxes)} table regions from page text")
                    except Exception as bbox_error:
                        # バウンディングボックスエラーの場合は表除外をスキップ
                        logger.warning(f"Page {page_num}: Failed to exclude table regions ({bbox_error}), extracting full page text")
                        text = page.extract_text() or ""
                else:
                    # 通常のテキスト抽出
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
                    # 表の領域を除外したテキストでチャンク化
                    page_result["text_chunks"].extend(
                        self._create_text_chunks(text, page_num, source_file, category)
                    )

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
                        # 10ページごと、または最終ページでログ出力
                        if page_num % 10 == 0 or page_num == page_count:
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

    def _preserve_table_context(self, text: str) -> str:
        """
        表の前後にコンテキスト情報を保持
        表や図の参照を検出し、段落区切りを強化

        Args:
            text: 処理対象のテキスト

        Returns:
            str: コンテキスト保持処理後のテキスト
        """
        patterns = [
            r'(表\s*\d+[.:].*?)(\n)',
            r'(図\s*\d+[.:].*?)(\n)',
            r'(Table\s+\d+[.:].*?)(\n)',
            r'(Figure\s+\d+[.:].*?)(\n)',
        ]

        for pattern in patterns:
            text = re.sub(pattern, r'\1\n\n', text)

        return text

    def _create_text_chunks(self, text: str, page_num: int, source_file: str, category: str) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割（セマンティックまたはレガシーチャンキング）

        Args:
            text: ページから抽出したテキスト
            page_num: ページ番号
            source_file: ソースファイル名
            category: カテゴリー

        Returns:
            list: テキストチャンクのリスト
        """
        # 表のコンテキストを保持
        text = self._preserve_table_context(text)

        # セマンティックチャンキング or レガシーチャンキング
        if self.text_splitter:
            # セマンティックチャンキング使用
            chunk_texts = self.text_splitter.split_text(text)
            chunks = []
            for idx, chunk_text in enumerate(chunk_texts):
                if chunk_text.strip():
                    token_count = len(self.encoding.encode(chunk_text))
                    chunks.append({
                        "text": chunk_text.strip(),
                        "page_number": page_num,
                        "source_file": source_file,
                        "category": category,
                        "chunk_index": idx,
                        "token_count": token_count,
                    })
            logger.debug(f"Created {len(chunks)} semantic chunks from page {page_num}")
            return chunks
        else:
            # レガシーチャンキング（既存実装）
            return self._legacy_create_text_chunks(text, page_num, source_file, category)

    def _legacy_create_text_chunks(self, text: str, page_num: int, source_file: str, category: str) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割（レガシー実装 - トークンベース）

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
        # 統計マトリクス（通常は結合セルなし）をMarkdown化するため、閾値を高めに設定
        merged_cell_threshold = self.pdf_config.get("merged_cell_threshold", 0.4)  # デフォルト40%
        none_count = sum(1 for row in table_data for cell in row if cell is None)
        if none_count > total_cells * merged_cell_threshold:
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
            dict: {"images": [...], "table_markdowns": [...], "table_bboxes": [...]}
        """
        images = []
        table_markdowns = []
        table_bboxes = []  # Initialize table_bboxes list
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
                                # 表の上にあるタイトルを抽出
                                table_title = self._extract_table_title(page, table.bbox)

                                # タイトルがある場合は追加
                                if table_title:
                                    table_text = f"\n{table_title}\n{markdown}\n"
                                else:
                                    table_text = f"\n[表 {table_idx + 1}]\n{markdown}\n"

                                table_markdowns.append({
                                    "text": table_text,
                                    "page_number": page_num,
                                })
                                # Markdown化された表のbboxを記録
                                table_bboxes.append(table.bbox)
                                logger.debug(f"Converted simple table {table_idx} to Markdown on page {page_num}, title: {table_title[:50] if table_title else 'None'}")

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

            # 3. ページ全体を画像化（フローチャート等のベクターグラフィックス対応）
            # まず、ページに図形（ベクターグラフィックス）があるかチェック
            graphics_count = 0
            if hasattr(page, 'curves') and page.curves:
                graphics_count += len(page.curves)
            if hasattr(page, 'rects') and page.rects:
                graphics_count += len(page.rects)
            if hasattr(page, 'lines') and page.lines:
                graphics_count += len(page.lines)

            has_many_graphics = graphics_count > 10  # フローチャート等の判定閾値

            # ページ全体を画像化する条件:
            # 1. 埋め込み画像が無い、かつ
            # 2. (複雑な表が無い、または、多くの図形要素がある場合)
            #    → 図形が多い場合は、表として誤検出されたフローチャート等の可能性
            has_complex_tables = any(t.get("text") for t in table_markdowns if t.get("text", "").strip())
            should_capture_full_page = len(images) == 0 and (not has_complex_tables or has_many_graphics)

            if should_capture_full_page:
                # ページに何らかのコンテンツがあるかチェック
                page_text = page.extract_text()
                has_content = page_text and len(page_text.strip()) > 50  # 最低50文字

                # コンテンツまたは図形があれば、ページ全体を画像化
                if has_content or has_many_graphics:
                    try:
                        reason = "high graphics density (flowchart/diagram)" if has_many_graphics else "text content"
                        logger.info(f"Capturing page {page_num} as full image due to {reason} (graphics_count={graphics_count})")

                        # ページ全体を画像化
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
                            "source_file": Path(pdf_path).name,
                            "category": None,  # 後で設定される
                            "content_type": "full_page",  # ページ全体
                        }
                        images.append(image_info)

                        logger.info(f"Captured full page {page_num} as image")
                    except Exception as e:
                        logger.warning(f"Failed to capture full page {page_num}: {e}")

            logger.info(f"Extracted {len(images)} images and {len(table_markdowns)} markdown tables from page {page_num}")

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")

        return {"images": images, "table_markdowns": table_markdowns, "table_bboxes": table_bboxes}

    def _extract_table_title(self, page, table_bbox, max_lines=3):
        """
        表の上にあるタイトルテキストを抽出

        Args:
            page: pdfplumber Page object
            table_bbox: 表のbounding box (x0, top, x1, bottom)
            max_lines: 抽出する最大行数

        Returns:
            str: 表のタイトル（なければ空文字列）
        """
        try:
            # 表の上端のy座標
            table_top = table_bbox[1]

            # ページの全単語を取得
            words = page.extract_words()

            if not words:
                return ""

            # 表の上にある単語をフィルタ（y座標が表のtop未満）
            # バッファ: 表の上100px以内のテキストのみ対象
            words_above = [
                w for w in words
                if w['bottom'] < table_top and w['bottom'] > table_top - 100
            ]

            if not words_above:
                return ""

            # y座標でグループ化（同じ行の単語をまとめる）
            lines = {}
            for word in words_above:
                # y座標を丸めて行をグループ化（±2pxの誤差を許容）
                line_y = round(word['top'] / 2) * 2
                if line_y not in lines:
                    lines[line_y] = []
                lines[line_y].append(word)

            # 表に最も近い行から順に並べる
            sorted_lines = sorted(lines.items(), key=lambda x: -x[0])  # y座標降順（下から上）

            # 最大max_lines行まで抽出
            title_lines = []
            for i, (y, line_words) in enumerate(sorted_lines[:max_lines]):
                # x座標でソート（左から右）
                line_words.sort(key=lambda w: w['x0'])
                line_text = ' '.join([w['text'] for w in line_words])

                # 「表」や「Table」で始まる行、または「単位」を含む行は優先的に採用
                if i == 0 or '表' in line_text or 'Table' in line_text or '単位' in line_text:
                    title_lines.insert(0, line_text)  # 上の行を先頭に追加

            return '\n'.join(reversed(title_lines))  # 上から下の順に戻す

        except Exception as e:
            logger.warning(f"Failed to extract table title: {e}")
            return ""

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
