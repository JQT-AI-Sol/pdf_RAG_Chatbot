-- Check for category encoding or whitespace issues

-- 1. Check exact category values stored in database
SELECT DISTINCT
    category,
    length(category) as char_length,
    octet_length(category) as byte_length,
    encode(category::bytea, 'hex') as hex_encoding,
    ascii(substring(category, 1, 1)) as first_char_ascii,
    ascii(substring(category, length(category), 1)) as last_char_ascii
FROM pdf_text_chunks
WHERE category IS NOT NULL
ORDER BY category;

-- 2. Check for invisible characters or whitespace
SELECT
    category,
    category = 'マッチングシステム' as exact_match,
    category LIKE 'マッチングシステム%' as starts_with,
    category LIKE '%マッチングシステム' as ends_with,
    category LIKE '%マッチングシステム%' as contains,
    trim(category) = 'マッチングシステム' as matches_after_trim,
    COUNT(*) as chunk_count
FROM pdf_text_chunks
WHERE category LIKE '%マッチング%'
GROUP BY category;

-- 3. Compare with working PDF category
SELECT
    source_file,
    category,
    category = 'キャリアアップ助成金' as exact_match,
    COUNT(*) as chunk_count
FROM pdf_text_chunks
WHERE source_file LIKE '%キャリアアップ%'
GROUP BY source_file, category;

-- 4. List all categories with their chunk counts
SELECT
    category,
    COUNT(DISTINCT source_file) as file_count,
    COUNT(*) as chunk_count,
    MIN(source_file) as example_file
FROM pdf_text_chunks
WHERE category IS NOT NULL
GROUP BY category
ORDER BY category;
