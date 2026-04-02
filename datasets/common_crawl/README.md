# Common Crawl Examples with `daft.datasets.common_crawl()`

This folder contains patterns for working with Common Crawl data using Daft's `daft.datasets.common_crawl()` API.

## Quick Start

```python
import daft

# Load plain text for LLM training
df = daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",
    content="text",
    num_files=1,
    in_aws=False
)

df.show(5)
```

## Examples

### Basic Examples

| File | Description | Runtime |
|------|-------------|---------|
| **[basic_warc.py](./basic_warc.py)** | Load raw WARC data with full HTTP responses | ~30s |
| **[basic_wet.py](./basic_wet.py)** | Load plain text (WET) for text analysis | ~30s |
| **[basic_wat.py](./basic_wat.py)** | Load metadata (WAT) for content statistics | ~30s |

### Advanced Use Cases

| File | Description | Runtime |
|------|-------------|---------|
| **[content_analysis.py](./content_analysis.py)** | Analyze MIME types and content distribution | ~1min |
| **[text_deduplication.py](./text_deduplication.py)** | Paragraph-level deduplication using MinHash + LSH | ~5min |
| **[chunk_embed.py](./chunk_embed.py)** | Chunk text with spaCy and generate embeddings | ~3min |

## Content Types

### WARC (Raw)
- **Aliases:** `"raw"`, `"warc"`
- **Size:** Largest (includes full HTTP headers/body)
- **Use cases:** Web archiving, HTTP analysis, full protocol inspection
- **Example:** `content="raw"`

### WET (Text)
- **Aliases:** `"text"`, `"wet"`
- **Size:** Smallest (plain text only)
- **Use cases:** LLM training, text mining, NLP tasks
- **Example:** `content="text"`

### WAT (Metadata)
- **Aliases:** `"metadata"`, `"wat"`
- **Size:** Medium (metadata without content)
- **Use cases:** Content analysis, statistics, URL filtering
- **Example:** `content="metadata"`

## Parameters

### Required Parameters

```python
daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",  # Crawl identifier
    in_aws=False,              # True if running in AWS, False otherwise
)
```

### Optional Parameters

```python
daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",
    content="text",            # Content type: raw/warc, text/wet, metadata/wat
    segment="1754151279521.11", # Specific segment (100 per crawl)
    num_files=1,               # Limit files for testing
    io_config=IOConfig(...),   # AWS credentials
    in_aws=False,
)
```

## Authentication

⚠️ **AWS Credentials Required**: All examples require AWS credentials to access Common Crawl index files, even when using `in_aws=False`.

```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### Inside AWS (Recommended for Production)

Run from us-east-1 for best performance (10-100x faster):

```python
from daft.io import IOConfig, S3Config

io_config = IOConfig(
    s3=S3Config(
        region_name="us-east-1",
        requester_pays=True,
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
)

df = daft.datasets.common_crawl(
    "CC-MAIN-2025-33",
    in_aws=True,
    io_config=io_config
)
```

### Outside AWS (Development)

Access via HTTPS (slower but no inter-region fees):

```python
from daft.io import IOConfig, S3Config

io_config = IOConfig(
    s3=S3Config(
        region_name="us-east-1",
        requester_pays=True,
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
)

df = daft.datasets.common_crawl(
    "CC-MAIN-2025-33",
    in_aws=False,
    io_config=io_config
)
```

## Schema

### WARC Schema (content="raw")

```
WARC-Record-ID: String              # Unique record identifier
WARC-Target-URI: String             # Crawled URL
WARC-Type: String                   # Record type (response, request, etc.)
WARC-Date: Timestamp                # Crawl timestamp
WARC-Identified-Payload-Type: String # MIME type
warc_content: Binary                # Full HTTP response
warc_headers: String                # WARC headers (JSON)
```

### WET Schema (content="text")

```
WARC-Record-ID: String              # Unique record identifier
WARC-Target-URI: String             # Source URL
WARC-Type: String                   # "conversion"
WARC-Date: Timestamp                # Crawl timestamp
warc_content: Binary                # Plain text (decode with try_decode)
```

### WAT Schema (content="metadata")

```
WARC-Record-ID: String              # Unique record identifier
WARC-Target-URI: String             # Crawled URL
WARC-Date: Timestamp                # Crawl timestamp
WARC-Identified-Payload-Type: String # MIME type
warc_content: Binary                # Metadata JSON
```

## Common Patterns

### Decode Text Content

```python
from daft import col

df.with_column("text", col("warc_content").try_decode("utf-8"))
```

### Filter by MIME Type

```python
df.where(col("WARC-Identified-Payload-Type") == "text/html")
```

### Content Size Analysis

```python
df.with_column("size", col("warc_content").length()).select("size").describe()
```

### Extract Paragraphs

```python
@daft.func()
def split_paragraphs(text: str) -> list[str]:
    return text.split("\n\n")

df.with_column("paragraphs", split_paragraphs(col("text"))).explode("paragraphs")
```

## Performance Tips

1. **Start with `num_files=1`** for testing
2. **Use `in_aws=True`** from EC2/Lambda for 10-100x faster access
3. **Run in us-east-1** to avoid inter-region transfer fees
4. **Choose smallest content type** needed:
   - Text mining → use WET
   - Metadata analysis → use WAT
   - Full HTTP analysis → use WARC
5. **Enable dynamic batching** for optimal throughput:
   ```python
   daft.set_execution_config(enable_dynamic_batching=True)
   ```

## Deduplication Strategy

Common Crawl contains duplicates. For LLM training datasets:

1. **Paragraph-level dedup** - Use MinHash + LSH (see `text_deduplication.py`)
2. **URL-based dedup** - Group by `WARC-Target-URI`
3. **Content-hash dedup** - Hash normalized text

Example:
```python
# Simple URL-based dedup
df.groupby("WARC-Target-URI").first()
```

## Resources

- **[Common Crawl Docs](https://docs.daft.ai/en/stable/datasets/common-crawl/)** - Official Daft documentation
- **[Common Crawl Tutorial](https://docs.daft.ai/en/stable/examples/common-crawl-daft-tutorial/)** - End-to-end tutorial
- **[Common Crawl Website](https://commoncrawl.org/)** - Dataset information and index
- **[AWS Open Data](https://registry.opendata.aws/commoncrawl/)** - S3 bucket details

## Common Issues

### Issue: Slow download speeds
**Solution:** Set `in_aws=True` and run from EC2 in us-east-1

### Issue: Out of memory
**Solution:** Use `num_files=1` or add `.limit()` to test queries first

### Issue: Permission denied
**Solution:** Verify AWS credentials and set `requester_pays=True` in IOConfig

### Issue: Empty results
**Solution:** Check crawl ID exists at https://commoncrawl.org/get-started
