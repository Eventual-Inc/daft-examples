# Daft Public Datasets

This folder contains examples for accessing public datasets available in Daft's S3 buckets (`daft-public-datasets` and `daft-public-data`).

## Available Datasets

### 1. Common Crawl (`daft.datasets.common_crawl()`)

**Source:** `daft.datasets` API
**Location:** Common Crawl S3 (requester-pays)
**Size:** 250+ billion web pages (18 years)
**Auth:** AWS credentials required

Access the world's largest web archive for LLM training, text mining, and web analysis.

**Quick Example:**
```python
import daft

df = daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",
    content="text",
    num_files=1,
    in_aws=False,
    io_config=IOConfig(...)
)
df.show(5)
```

**See:** [`common_crawl/`](./common_crawl/)

---

### 2. Reddit IRL

**Source:** `s3://daft-public-datasets/reddit-irl/`
**Size:** 10M+ comments, thousands of images
**Auth:** Public (no credentials needed)

Reddit comments, posts, images, and pre-computed embeddings for social media analytics.

**Quick Example:**
```python
import daft

df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet")
df.show(5)
```

**See:** [`reddit_irl/`](./reddit_irl/)

---

### 3. TPC-H Lineitem

**Source:** `s3://daft-public-datasets/tpch-lineitem/`
**Size:** 10GB (10K files, ~60M rows)
**Auth:** Public (no credentials needed)

Industry-standard SQL benchmark dataset for testing distributed query performance.

**Quick Example:**
```python
import daft

df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")
df.show(5)
```

**See:** [`tpch/`](./tpch/)

---

### 4. Open Images

**Source:** `s3://daft-public-data/open-images/`
**Size:** 41K+ validation images
**Auth:** Public (no credentials needed)

Google Open Images validation set for computer vision tasks.

**Quick Example:**
```python
import daft
from daft.functions import file

df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")
df = df.with_column("image", file(daft.col("path")))
df.show(5)
```

**See:** [`open_images/`](./open_images/)

---

### 5. LAION Samples

**Source:** `s3://daft-public-data/laion-sample-images/`, `s3://daft-public-data/tutorials/laion-parquet/`
**Size:** Sample of LAION-5B dataset
**Auth:** Public (no credentials needed)

Image-text pairs with CLIP similarity scores for vision-language model training.

**Quick Example:**
```python
import daft

df = daft.read_parquet("s3://daft-public-data/tutorials/laion-parquet/train-*.parquet")
df.show(5)
```

**See:** [`laion/`](./laion/)

## Dataset Comparison

| Dataset | Size | Type | Use Cases | Auth Required |
|---------|------|------|-----------|---------------|
| **Common Crawl** | 250B+ pages | Text, HTML, Metadata | LLM training, web mining | ✅ AWS |
| **Reddit IRL** | 10M+ comments | Text, Images | Social analytics, NLP | ❌ Public |
| **TPC-H** | 60M rows, 10GB | Structured CSV | SQL benchmarks | ❌ Public |
| **Open Images** | 41K images | Images | Computer vision | ❌ Public |
| **LAION** | Sample dataset | Image-text pairs | Multimodal models | ❌ Public |

## Quick Start by Use Case

### Text Analysis / LLM Training
- **Common Crawl (WET)** - Massive web text corpus
- **Reddit IRL** - Social media conversations

### Computer Vision
- **Open Images** - Pre-labeled images for classification
- **LAION** - Image-text pairs for generative models

### Structured Data / SQL
- **TPC-H** - Benchmark queries and performance testing

### Multimodal AI
- **LAION** - Image + caption pairs for CLIP-style training
- **Reddit IRL** - Social posts with images

## Authentication

⚠️ **AWS credentials are currently required** for all Common Crawl access, even with `in_aws=False`. Set these environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

**Inside AWS (us-east-1 recommended for best performance):**
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

**Outside AWS (HTTPS access):**
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
    io_config=io_config  # Still needed for index file access
)
```

## Examples by Dataset

### Common Crawl
- `basic_warc.py` - Raw HTTP responses
- `basic_wet.py` - Plain text extracts
- `basic_wat.py` - Metadata only
- `content_analysis.py` - MIME type analysis
- `text_deduplication.py` - MinHash deduplication
- `chunk_embed.py` - Chunking + embeddings

### Reddit IRL
- `basic_comments.py` - Load and analyze comments
- `basic_images.py` - Posts with images
- `embeddings.py` - Pre-computed embeddings

### TPC-H
- `basic_query.py` - Standard SQL queries
- `performance_test.py` - Benchmark Daft
- `sql_queries.py` - TPC-H query templates

### Open Images
- `basic_images.py` - Load and inspect images
- `image_processing.py` - Resize and transform
- `vision_models.py` - GPT-4 Vision inference

### LAION
- `basic_metadata.py` - Explore metadata
- `image_text_pairs.py` - Image-caption pairs
- `clip_training.py` - Prepare for CLIP training

## Performance Tips

### General Best Practices
1. **Start with `.limit()`** - Test queries on small samples first
2. **Enable dynamic batching** - `daft.set_execution_config(enable_dynamic_batching=True)`
3. **Filter early** - Push filters down to reduce data scanned
4. **Use appropriate file formats** - Parquet > CSV for structured data

### Dataset-Specific Tips

**Common Crawl:**
- Run in us-east-1 for faster access
- Use `in_aws=True` on EC2/Lambda
- Choose smallest content type needed (WET < WAT < WARC)

**Reddit IRL:**
- Filter by subreddit early
- Use partition columns in filters
- Sample for ML model development

**TPC-H:**
- Leverage parallel CSV reading (10K files)
- Enable projection pushdown
- Use date filters for better performance

**Open Images / LAION:**
- Limit during development (images are large)
- Resize images early before expensive operations
- Batch process for efficiency

## Hackathon-Ready Examples

These datasets are ideal for hackathons and demos:

### 5-Minute Demos
- **Reddit IRL comments** - Sentiment analysis, trend detection
- **Open Images** - Image classification with GPT-4 Vision
- **TPC-H queries** - SQL performance showcase

### 15-Minute Demos
- **LAION pairs** - Build semantic image search
- **Reddit IRL images** - Multimodal content moderation
- **TPC-H + SQL** - Complex analytics dashboard

### Production Examples
- **Common Crawl** - LLM pre-training pipeline
- **Reddit IRL + embeddings** - Social recommendation system
- **LAION** - Train custom CLIP models

## Authentication Summary

| Dataset | Bucket | Auth Required | Credentials |
|---------|--------|---------------|-------------|
| Common Crawl | Common Crawl S3 | ✅ Yes | AWS access keys |
| Reddit IRL | daft-public-datasets | ❌ No | Public read |
| TPC-H | daft-public-datasets | ❌ No | Public read |
| Open Images | daft-public-data | ❌ No | Public read |
| LAION | daft-public-data | ❌ No | Public read |

## Resources

### Documentation
- [Daft Datasets API](https://docs.daft.ai/en/stable/api/datasets/)
- [Common Crawl Docs](https://docs.daft.ai/en/stable/datasets/common-crawl/)
- [Daft I/O Guide](https://docs.daft.ai/en/stable/user-guide/io/)

### External Resources
- [Common Crawl Website](https://commoncrawl.org/)
- [TPC-H Benchmark](http://www.tpc.org/tpch/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)
- [LAION](https://laion.ai/)
