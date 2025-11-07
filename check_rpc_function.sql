-- Check the RPC function definition for match_text_chunks

-- 1. Get the function definition
SELECT
    proname as function_name,
    pg_get_functiondef(oid) as function_definition
FROM pg_proc
WHERE proname = 'match_text_chunks';

-- 2. Get function parameters
SELECT
    proname as function_name,
    proargnames as parameter_names,
    proargtypes::regtype[] as parameter_types
FROM pg_proc
WHERE proname = 'match_text_chunks';

-- 3. Also check match_image_contents
SELECT
    proname as function_name,
    pg_get_functiondef(oid) as function_definition
FROM pg_proc
WHERE proname = 'match_image_contents';
