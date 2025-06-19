#!/usr/bin/env python3
"""Debug script to test URL loading individually."""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()

def test_url_loading():
    """Test loading each URL individually to see which ones fail."""
    
    # Read URLs from file
    with open('urls.txt', 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    print(f"Testing {len(urls)} URLs:")
    print("=" * 60)
    
    successful_urls = []
    failed_urls = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{i}. Testing: {url}")
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            if docs:
                content_length = len(docs[0].page_content) if docs[0].page_content else 0
                print(f"   ✅ SUCCESS: Loaded {len(docs)} documents, {content_length} characters")
                successful_urls.append(url)
            else:
                print(f"   ⚠️  WARNING: No content loaded")
                failed_urls.append((url, "No content"))
                
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            failed_urls.append((url, str(e)))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Successful: {len(successful_urls)}")
    print(f"❌ Failed: {len(failed_urls)}")
    
    if successful_urls:
        print(f"\nSuccessful URLs:")
        for url in successful_urls:
            print(f"  - {url}")
    
    if failed_urls:
        print(f"\nFailed URLs:")
        for url, error in failed_urls:
            print(f"  - {url}")
            print(f"    Error: {error}")

if __name__ == "__main__":
    test_url_loading()
