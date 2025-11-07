-- Test query embedding against マッチングシステム data
-- Replace <QUERY_EMBEDDING> with the actual embedding from logs

-- 1. Check if query embedding can be compared with stored embeddings
SELECT
    id,
    source_file,
    page_number,
    substring(content, 1, 100) as content_preview,
    1 - (embedding <=> '<QUERY_EMBEDDING>'::vector) as similarity,
    embedding <=> '<QUERY_EMBEDDING>'::vector as distance
FROM pdf_text_chunks
WHERE category = 'マッチングシステム'
ORDER BY embedding <=> '<QUERY_EMBEDDING>'::vector
LIMIT 10;

-- 2. Check if threshold filtering works
SELECT
    id,
    source_file,
    similarity,
    distance,
    CASE
        WHEN distance < 0.5 THEN 'PASS (distance < 0.5)'
        ELSE 'FAIL (distance >= 0.5)'
    END as threshold_check
FROM (
    SELECT
        id,
        source_file,
        1 - (embedding <=> '<QUERY_EMBEDDING>'::vector) as similarity,
        embedding <=> '<QUERY_EMBEDDING>'::vector as distance
    FROM pdf_text_chunks
    WHERE category = 'マッチングシステム'
    ORDER BY embedding <=> '<QUERY_EMBEDDING>'::vector
    LIMIT 10
) results;

-- 3. Simulate the exact RPC function call
-- This shows what match_text_chunks SHOULD return
SELECT
    id,
    content,
    source_file,
    page_number,
    category,
    1 - (embedding <=> '<QUERY_EMBEDDING>'::vector) as similarity
FROM pdf_text_chunks
WHERE category = 'マッチングシステム'
    AND (embedding <=> '<QUERY_EMBEDDING>'::vector) < 0.5  -- threshold check
ORDER BY embedding <=> '<QUERY_EMBEDDING>'::vector
LIMIT 10;

-- Instructions:
-- 1. Run the app and search for "マッチングシステムの構成図はどうなってる？"
-- 2. Check logs for "🔍 DEBUG: First 10 elements of query embedding: [...]"
-- 3. Copy the FULL embedding array (all 3072 values) from logs
-- 4. Replace <QUERY_EMBEDDING> in this file with the actual values
-- 5. Run these queries in Supabase SQL Editor
