"""
Text file processing module - プレーンテキストの抽出
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class TextFileProcessor:
    """テキストファイルからコンテンツを抽出するクラス"""

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: 処理設定
        """
        self.config = config
        self.pdf_config = config.get("pdf_processing", {})  # 同じ設定を使用
        self.rag_config = config.get("rag", {})
        self.chunking_config = config.get("chunking", {})

        # tiktokenエンコーダーの初期化
        model_name = config.get("openai", {}).get("model_chat", "gpt-4.1")
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
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
                logger.info(f"Semantic chunking initialized for text files (chunk_size={chunk_size}, overlap={chunk_overlap})")
            except Exception as e:
                logger.error(f"Failed to initialize semantic chunker: {e}")
                logger.warning("Falling back to legacy chunking")

    def process_text_file(self, txt_path: str, category: str) -> Dict[str, Any]:
        """
        テキストファイルを処理してコンテンツを抽出

        Args:
            txt_path: テキストファイルのパス
            category: ドキュメントカテゴリー

        Returns:
            dict: 抽出結果（テキストチャンクなど）
        """
        logger.info(f"Processing text file: {txt_path}")

        result = {
            "source_file": Path(txt_path).name,
            "category": category,
            "text_chunks": [],
            "images": [],  # テキストファイルには画像がない
            "metadata": {},
        }

        try:
            # テキストファイルを読み込み（複数のエンコーディングを試行）
            encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp', 'latin-1']
            text = None
            used_encoding = None

            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    used_encoding = encoding
                    logger.info(f"Successfully read text file with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to read with encoding {encoding}: {e}")
                    continue

            if text is None:
                error_msg = f"Failed to read text file with any supported encoding"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # メタデータ
            result["metadata"]["encoding"] = used_encoding
            result["metadata"]["character_count"] = len(text)
            result["metadata"]["line_count"] = text.count('\n') + 1

            # テキストをチャンクに分割
            if text.strip():
                result["text_chunks"] = self._create_text_chunks(
                    text,
                    section_num=1,  # テキストファイルは単一セクション
                    source_file=result["source_file"],
                    category=category
                )

            logger.info(f"Extracted {len(result['text_chunks'])} text chunks from text file")

        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise

        return result

    def _create_text_chunks(self, text: str, section_num: int, source_file: str, category: str) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割

        Args:
            text: テキスト
            section_num: セクション番号
            source_file: ソースファイル名
            category: カテゴリー

        Returns:
            list: テキストチャンクのリスト
        """
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
                        "page_number": section_num,  # ページ番号の代わりにセクション番号
                        "source_file": source_file,
                        "category": category,
                        "chunk_index": idx,
                        "token_count": token_count,
                    })
            logger.debug(f"Created {len(chunks)} semantic chunks from text file")
            return chunks
        else:
            # レガシーチャンキング
            return self._legacy_create_text_chunks(text, section_num, source_file, category)

    def _legacy_create_text_chunks(self, text: str, section_num: int, source_file: str, category: str) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割（レガシー実装）

        Args:
            text: テキスト
            section_num: セクション番号
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
            return [{
                "text": text.strip(),
                "page_number": section_num,
                "source_file": source_file,
                "category": category,
                "chunk_index": 0,
                "token_count": len(tokens)
            }]

        # チャンクに分割
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "page_number": section_num,
                    "source_file": source_file,
                    "category": category,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                })
                chunk_index += 1

            start = end - chunk_overlap if end < len(tokens) else end

        logger.debug(f"Created {len(chunks)} chunks from text file ({len(tokens)} tokens)")
        return chunks
