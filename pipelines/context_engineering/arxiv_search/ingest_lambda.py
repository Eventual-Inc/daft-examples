# /// script
# description = "Ingest Arxiv Papers to S3 (Lambda-ready)"
# requires-python = ">=3.12, <3.13"
# dependencies = ["boto3", "feedparser", "python-dotenv"]
# ///

import json
import os
from datetime import datetime

import boto3
import feedparser
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "daft-arxiv-demo")
S3_PREFIX = os.getenv("S3_PREFIX", "raw/")
ARXIV_CATEGORY = os.getenv("ARXIV_CATEGORY", "cs.AI")


def fetch_papers():
    """Fetch recent papers from Arxiv API."""
    # Query: Category = cs.AI, Sorted by Submitted Date, Descending
    url = f"http://export.arxiv.org/api/query?search_query=cat:{ARXIV_CATEGORY}&sortBy=submittedDate&sortOrder=descending&max_results=50"
    print(f"Fetching from {url}...")
    feed = feedparser.parse(url)
    return feed.entries


def save_to_s3(entry):
    """Save a single paper entry to S3 as JSON."""
    s3 = boto3.client("s3")

    # Create a unique key based on arxiv ID (e.g., http://arxiv.org/abs/2310.12345 -> 2310.12345)
    paper_id = entry.id.split("/")[-1]

    # Partition by Date of ingestion
    date_str = datetime.now().strftime("%Y-%m-%d")
    key = f"{S3_PREFIX}{date_str}/{paper_id}.json"

    data = {
        "id": paper_id,
        "title": entry.title.replace("\n", " "),
        "summary": entry.summary.replace("\n", " "),
        "authors": [a.name for a in entry.authors],
        "published": entry.published,
        "link": entry.link,
        "category": ARXIV_CATEGORY,
        "ingested_at": datetime.now().isoformat(),
    }

    print(f"Saving {paper_id} to s3://{S3_BUCKET}/{key}")

    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(data),
            ContentType="application/json",
        )
    except Exception as e:
        print(f"Failed to upload {paper_id}: {e}")
        # For local testing without S3 credentials, we might want to dump to a file
        if os.getenv("LOCAL_MODE"):
            local_path = f".data/{key}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved locally to {local_path}")


def lambda_handler(event, context):
    """AWS Lambda Handler."""
    print("Starting ingestion...")
    entries = fetch_papers()
    print(f"Found {len(entries)} papers.")

    for entry in entries:
        save_to_s3(entry)

    return {"statusCode": 200, "body": json.dumps(f"Ingested {len(entries)} papers")}


if __name__ == "__main__":
    # Simulate Lambda execution
    lambda_handler(None, None)
