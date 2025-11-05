-- registered_pdfsテーブルにstorage_path列を追加するSQL

-- storage_path列を追加（既に存在する場合はエラーを無視）
ALTER TABLE registered_pdfs
ADD COLUMN IF NOT EXISTS storage_path TEXT;

-- storage_path列にインデックスを追加（検索高速化）
CREATE INDEX IF NOT EXISTS idx_registered_pdfs_storage_path
ON registered_pdfs(storage_path);

-- 確認
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'registered_pdfs';
