"""
Supabase Storage バケットの確認スクリプト
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')

client = create_client(supabase_url, supabase_key)

print("=== Supabase Storage Buckets ===\n")

try:
    buckets = client.storage.list_buckets()

    if buckets:
        for bucket in buckets:
            print(f"Bucket: {bucket.name}")
            print(f"  ID: {bucket.id}")
            print(f"  Public: {bucket.public}")
            print(f"  Created: {bucket.created_at}")
            print()
    else:
        print("No buckets found")

    # pdf-files バケットの確認
    pdf_bucket = [b for b in buckets if b.name == 'pdf-files']

    if pdf_bucket:
        print("OK: pdf-files bucket exists!")
        print(f"   Public: {pdf_bucket[0].public}")
        print(f"   (Should be False for private bucket)")
    else:
        print("ERROR: pdf-files bucket NOT found")
        print("   Please create it in Supabase Dashboard")

except Exception as e:
    print(f"Error: {e}")
