# /// script
# description = "Search Arxiv Papers using Turbopuffer"
# requires-python = ">=3.12, <3.13"
# dependencies = ["openai", "turbopuffer", "python-dotenv"]
# ///

import os
import time

import turbopuffer as tp
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

tp.api_key = os.getenv("TURBOPUFFER_API_KEY")
client = OpenAI()


def search(query_text, top_k=5):
    print(f"Embedding query: '{query_text}'...")
    resp = client.embeddings.create(input=query_text, model="text-embedding-3-small")
    query_vector = resp.data[0].embedding

    print("Querying Turbopuffer...")
    ns = tp.Namespace("arxiv-papers")
    results = ns.query(
        vector=query_vector,
        top_k=top_k,
        include_attributes=["title", "summary", "link", "authors"],
    )

    return results


if __name__ == "__main__":
    if not os.getenv("TURBOPUFFER_API_KEY"):
        print("Please set TURBOPUFFER_API_KEY to run search.")
        exit(1)

    print("--- Arxiv Semantic Search ---")
    while True:
        try:
            q = input("\nEnter search query (or 'q' to quit): ")
            if q.lower() in ["q", "quit", "exit"]:
                break

            start = time.time()
            results = search(q)
            duration = time.time() - start

            print(f"\nFound {len(results)} results in {duration:.2f}s:\n")
            for res in results:
                attrs = res.attributes

                print(f"Title: {attrs['title']}")
                print(f"Summary: {attrs['summary']}")
                print(f"Authors: {attrs.get('authors', 'Unknown')}")
                print(f"Link: {attrs['link']}")
                print("-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
