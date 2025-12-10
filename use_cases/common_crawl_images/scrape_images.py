import os
import daft
import io
import boto3
from urllib.parse import urlparse

# Constants
# Using a recent Common Crawl snapshot
WAT_PATHS_URL = "s3://commoncrawl/crawl-data/CC-MAIN-2023-50/wat.paths.gz"
TARGET_IMAGE_COUNT = 2_000_000
# Default bucket if not provided
S3_BUCKET = os.environ.get("S3_BUCKET", "daft-public-data") 
S3_PREFIX = os.environ.get("S3_PREFIX", "common-crawl-images/")

# Social media domains to prioritize
SOCIAL_DOMAINS = [
    "twitter.com", "x.com", "instagram.com", "facebook.com", 
    "reddit.com", "linkedin.com", "tiktok.com", "pinterest.com",
    "flickr.com", "tumblr.com", "snapchat.com", "weibo.com"
]

def get_s3_client():
    return boto3.client("s3")

def is_social_domain(url):
    try:
        domain = urlparse(url).netloc
        # Remove 'www.' if present
        if domain.startswith("www."):
            domain = domain[4:]
        
        for social in SOCIAL_DOMAINS:
            if social in domain:
                return True
        return False
    except:
        return False

# UDF to download and upload image to S3
@daft.udf(return_dtype=daft.DataType.string())
def download_and_upload(url: str, original_url: str) -> str:
    # This is a placeholder for the actual download/upload logic
    # We will implement this with proper error handling and boto3
    return f"s3://{S3_BUCKET}/{S3_PREFIX}placeholder.jpg"

import json
from bs4 import BeautifulSoup

from urllib.parse import urlparse, urljoin

@daft.udf(return_dtype=daft.DataType.list(daft.DataType.string()))
def extract_links(warc_contents: daft.Series, base_urls: daft.Series) -> list[list[str]]:
    results = []
    # Iterate over both content and base_url
    for content, base_url in zip(warc_contents.to_pylist(), base_urls.to_pylist()):
        if content is None:
            results.append([])
            continue
        try:
            soup = BeautifulSoup(content, "html.parser")
            image_urls = []
            for img in soup.find_all("img"):
                src = img.get("src")
                if src:
                    # Resolve relative URLs
                    if base_url:
                        src = urljoin(base_url, src)
                    image_urls.append(src)
            results.append(image_urls)
        except Exception:
            results.append([])
    return results

def main():
    print(f"Starting Common Crawl Image Scraper")
    print(f"Target Bucket: {S3_BUCKET}")
    
    # Authenticate with AWS if credentials are present
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        print("Using AWS Credentials")
        io_config = daft.io.IOConfig(
            s3=daft.io.S3Config(
                region_name="us-east-1",
                requester_pays=True,
                key_id=os.environ["AWS_ACCESS_KEY_ID"],
                access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                anonymous=False,
            )
        )
        in_aws = True
    else:
        print("Using Anonymous Access")
        io_config = daft.io.IOConfig(
            s3=daft.io.S3Config(
                region_name="us-east-1",
                anonymous=True
            )
        )
        in_aws = False

    # Use a recent crawl ID
    CRAWL_ID = "CC-MAIN-2024-42"
    
    print(f"Reading Common Crawl WARC files from: {CRAWL_ID}")
    
    df = daft.datasets.common_crawl(
        CRAWL_ID,
        content="warc", 
        num_files=20, # Increase to find images
        in_aws=in_aws,
        io_config=io_config
    )
    
    # Filter for 'response' records which contain the HTML payload
    df = df.where(daft.col("WARC-Type") == "response")
    
    # Extract the content (already bytes/string in warc_content)
    # We decode to string for BS4
    df = df.with_column("html_str", daft.col("warc_content").try_decode("utf-8"))
    
    # Filter out null content
    df = df.where(daft.col("html_str").not_null())
    
    # Extract links using UDF, passing WARC-Target-URI as base_url
    df = df.with_column("image_urls", extract_links(daft.col("html_str"), daft.col("WARC-Target-URI")))
    
    # Explode to get one row per image URL
    df = df.explode("image_urls")
    
    # Filter out nulls
    df = df.where(daft.col("image_urls").not_null())
    
    # Filter for social domains
    @daft.udf(return_dtype=daft.DataType.bool())
    def filter_social(urls: daft.Series) -> list[bool]:
        return [is_social_domain(url) for url in urls.to_pylist()]
        
    df = df.with_column("is_social", filter_social(daft.col("image_urls")))
    
    # Apply global limit to stop once we have enough images
    # This answers "kill the job once the full number is found"
    df = df.limit(TARGET_IMAGE_COUNT)
    
    # Progress Logging UDF
    # We use a stateful class or just a simple print in a UDF
    # Since Daft runs in parallel, this will be approximate and interleaved
    @daft.udf(return_dtype=daft.DataType.int64())
    def log_progress(urls: daft.Series) -> list[int]:
        # We can't easily maintain a global count across processes without external state (like Redis or a file)
        # But we can print "Found X images in this batch"
        count = len(urls.to_pylist())
        if count > 0:
            print(f"[Progress] Found {count} images in current batch")
        return [count] * len(urls)

    # We can attach this to the dataframe to force execution/logging
    df = df.with_column("log", log_progress(daft.col("image_urls")))
    
    print("Extracted Image URLs (Preview):")
    df.select("image_urls", "is_social").show(10)
    
    # Debug: Check Schema to find WARC-Target-URI
    print("Debug: Schema")
    print(df.schema())
    
    # Extract links using UDF
    # df = df.with_column("image_urls", extract_links(daft.col("html_str")))
    
    # Download and Upload UDF
    import requests
    import boto3
    import io
    import time
    
    # Global counter for this process (will be separate per worker)
    processed_count = 0
    
    @daft.func.batch(return_dtype=daft.DataType.string())
    def download_and_upload(urls: daft.Series) -> list[str]:
        # We can log progress here too
        nonlocal processed_count
        s3 = boto3.client("s3")
        results = []
        for url in urls.to_pylist():
            if not url:
                results.append(None)
                continue
            try:
                # Log every 100 images
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"[Worker Progress] Processed {processed_count} images...")

                # Download
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    # Upload to S3
                    key = f"{S3_PREFIX}{os.path.basename(urlparse(url).path)}"
                    # Ensure unique key if needed, but for now simple basename
                    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=resp.content)
                    results.append(f"s3://{S3_BUCKET}/{key}")
                else:
                    results.append(None)
            except Exception as e:
                # print(f"Error processing {url}: {e}")
                results.append(None)
        return results

    # Apply download and upload
    # We only want to download if we are actually running the full pipeline
    # For this demo, let's limit to a few images
    
    # For the real run, we would use the full df
    df_final = df.with_column("s3_uri", download_and_upload(daft.col("image_urls")))
    
    # Write to Parquet
    df_final.select("image_urls", "s3_uri", "is_social").write_parquet(f"s3://{S3_BUCKET}/common-crawl-images/metadata/")

if __name__ == "__main__":
    main()
