-- registered_pdfs テーブルに storage_path 列を追加

-- 列を追加
ALTER TABLE registered_pdfs
ADD COLUMN IF NOT EXISTS storage_path TEXT;

-- インデックスを追加（検索高速化）
CREATE INDEX IF NOT EXISTS idx_registered_pdfs_storage_path
ON registered_pdfs(storage_path);

-- 確認クエリ
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'registered_pdfs'
ORDER BY ordinal_position;
