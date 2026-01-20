# Daft Public Datasets - Complete Summary

## Overview

Comprehensive examples for 5 public datasets available in Daft's S3 buckets, organized by use case with 25 files total (21 Python examples + 4 documentation files).

## New Folder Structure

```
usage_patterns/datasets/
├── README.md                              # Main overview with all datasets
├── MIGRATION_SUMMARY.md                   # Common Crawl migration guide
├── DATASET_SUMMARY.md                     # This file
│
├── common_crawl/                          # daft.datasets.common_crawl()
│   ├── README.md
│   ├── basic_warc.py
│   ├── basic_wet.py
│   ├── basic_wat.py
│   ├── content_analysis.py
│   ├── text_deduplication.py
│   └── chunk_embed.py
│
├── reddit_irl/                            # s3://daft-public-datasets/reddit-irl/
│   ├── README.md
│   ├── basic_comments.py
│   ├── basic_images.py
│   └── embeddings.py
│
├── tpch/                                  # s3://daft-public-datasets/tpch-lineitem/
│   ├── README.md
│   ├── basic_query.py
│   ├── performance_test.py
│   └── sql_queries.py
│
├── open_images/                           # s3://daft-public-data/open-images/
│   ├── README.md
│   ├── basic_images.py
│   ├── image_processing.py
│   └── vision_models.py
│
└── laion/                                 # s3://daft-public-data/laion-*
    ├── README.md
    ├── basic_metadata.py
    ├── image_text_pairs.py
    └── clip_training.py
```

## Dataset Details

### 1. Common Crawl
**Focus:** `daft.datasets` API for web-scale text data

**Files:** 6 examples + 1 README
- `basic_warc.py` - Full HTTP responses with headers
- `basic_wet.py` - Plain text extracts (smallest)
- `basic_wat.py` - Metadata only (medium)
- `content_analysis.py` - MIME type distribution analysis
- `text_deduplication.py` - MinHash + LSH paragraph deduplication
- `chunk_embed.py` - spaCy chunking + sentence-transformers embeddings

**Key Features:**
- Three content types: WARC/WET/WAT
- 250B+ web pages spanning 18 years
- Requires AWS credentials (requester-pays bucket)
- Ideal for LLM training and web mining

**Authentication:** ✅ Required (AWS credentials)

---

### 2. Reddit IRL
**Focus:** Social media analytics with text + images

**Files:** 3 examples + 1 README
- `basic_comments.py` - Load and analyze 10M+ comments
- `basic_images.py` - Reddit posts with images
- `embeddings.py` - Pre-computed Qwen3 embeddings

**Key Features:**
- 10M+ Reddit comments
- Thousands of images from Reddit posts
- Pre-computed embeddings for semantic search
- Multiple subreddits included
- Partitioned data structure

**Authentication:** ❌ Public (no credentials needed)

**Use Cases:**
- Sentiment analysis
- Trend detection
- Social recommendation systems
- Content moderation

---

### 3. TPC-H Lineitem
**Focus:** SQL benchmark and performance testing

**Files:** 3 examples + 1 README
- `basic_query.py` - TPC-H Query 1 (pricing summary)
- `performance_test.py` - Benchmark Daft throughput
- `sql_queries.py` - Multiple TPC-H queries with Daft SQL

**Key Features:**
- 10,000 CSV files (~1MB each)
- 60M rows, ~10GB total
- Industry-standard benchmark
- Tests parallel CSV reading, aggregations, joins

**Authentication:** ❌ Public (no credentials needed)

**Use Cases:**
- SQL query performance testing
- Distributed processing benchmarks
- Demo complex aggregations
- Showcase Daft SQL

---

### 4. Open Images
**Focus:** Computer vision with validation images

**Files:** 3 examples + 1 README
- `basic_images.py` - Load and inspect 41K images
- `image_processing.py` - Resize, crop, transform images
- `vision_models.py` - GPT-4 Vision inference for captions/objects

**Key Features:**
- 41,620 validation images
- Google Open Images subset
- JPEG format
- Diverse real-world content

**Authentication:** ❌ Public (no credentials needed)

**Use Cases:**
- Image classification
- Object detection
- Vision model validation
- Image captioning with GPT-4V

---

### 5. LAION Samples
**Focus:** Image-text pairs for multimodal AI

**Files:** 3 examples + 1 README
- `basic_metadata.py` - Explore LAION metadata with CLIP scores
- `image_text_pairs.py` - Work with image-caption pairs
- `clip_training.py` - Prepare data for CLIP-style training

**Key Features:**
- Sample of LAION-5B dataset
- Image URLs + text captions
- CLIP similarity scores (quality metric)
- Image dimensions and metadata

**Authentication:** ❌ Public (no credentials needed)

**Use Cases:**
- CLIP/vision-language training
- Image captioning models
- Text-to-image generation prep
- Semantic search systems

---

## Code Quality Standards

All examples follow consistent patterns:

### 1. Print Statement Policy
✅ **Kept:** Structural section headers
```python
print("\n=== Sample Results ===")
```

❌ **Removed:** Status messages, warnings, success confirmations

### 2. Script Metadata (PEP 723)
All examples include uv-compatible metadata:
```python
# /// script
# description = "Brief description"
# dependencies = ["daft[aws]", "package2"]
# ///
```

### 3. Authentication Patterns
```python
# Public datasets (Reddit, TPC-H, Open Images, LAION)
df = daft.read_parquet("s3://daft-public-datasets/...")

# Common Crawl (requires credentials)
from daft.io import IOConfig, S3Config
io_config = IOConfig(s3=S3Config(...))
df = daft.datasets.common_crawl(..., io_config=io_config)
```

### 4. Output Format
- Use `.show()` for inspection
- Include `.describe()` for statistics
- Show sample data with `max_col_width` parameter
- No excessive logging or debug output

## Quick Start Guide

### For Hackathons (No AWS Credentials)

**5-Minute Demos:**
```bash
# Reddit sentiment analysis
uv run usage_patterns/datasets/reddit_irl/basic_comments.py

# Open Images vision
uv run usage_patterns/datasets/open_images/vision_models.py

# TPC-H SQL
uv run usage_patterns/datasets/tpch/basic_query.py
```

**15-Minute Demos:**
```bash
# LAION semantic search
uv run usage_patterns/datasets/laion/image_text_pairs.py

# Reddit multimodal
uv run usage_patterns/datasets/reddit_irl/basic_images.py
```

### For Production (With AWS)

**Common Crawl LLM Training:**
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
uv run usage_patterns/datasets/common_crawl/basic_wet.py
```

## Authentication Matrix

| Dataset | Bucket | Credentials | Cost |
|---------|--------|-------------|------|
| **Common Crawl** | Common Crawl S3 | AWS keys required | Data transfer (requester-pays) |
| **Reddit IRL** | daft-public-datasets | Public read | Free |
| **TPC-H** | daft-public-datasets | Public read | Free |
| **Open Images** | daft-public-data | Public read | Free |
| **LAION** | daft-public-data | Public read | Free |

## Use Case Matrix

| Use Case | Dataset | Example File | Runtime |
|----------|---------|--------------|---------|
| **LLM Training** | Common Crawl WET | `basic_wet.py` | Varies |
| **Sentiment Analysis** | Reddit IRL | `basic_comments.py` | ~30s |
| **SQL Benchmarks** | TPC-H | `performance_test.py` | ~2min |
| **Image Classification** | Open Images | `vision_models.py` | ~1min |
| **CLIP Training** | LAION | `clip_training.py` | ~1min |
| **Text Mining** | Common Crawl | `content_analysis.py` | Varies |
| **Social Analytics** | Reddit IRL | All examples | ~30s |
| **Vision Models** | Open Images | `vision_models.py` | ~1min |
| **Multimodal Search** | LAION | `image_text_pairs.py` | ~1min |
| **Deduplication** | Common Crawl | `text_deduplication.py` | ~5min |

## Performance Characteristics

### Dataset Sizes

| Dataset | Files | Rows/Images | Total Size | Load Time* |
|---------|-------|-------------|------------|------------|
| Common Crawl | Varies | 250B+ pages | Petabytes | Depends on filters |
| Reddit IRL | ~100s | 10M+ rows | ~50GB | ~2min full scan |
| TPC-H | 10,000 | 60M rows | 10GB | ~1min full scan |
| Open Images | 41,620 | 41K images | ~15GB | ~5min to load all |
| LAION | Sample | Varies | ~1GB metadata | ~10s |

*Approximate times for reading full dataset

### Recommended Limits

For development/testing:
- **Common Crawl:** `num_files=1` (varies by content type)
- **Reddit IRL:** `.limit(1000)` for comments
- **TPC-H:** Use full dataset (optimized for parallel processing)
- **Open Images:** `.limit(100)` for images
- **LAION:** `.limit(100)` for metadata

## Integration with Other Tools

### LangChain Integration
```python
# Reddit IRL for RAG
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet")
# → Chunk → Embed → Vector store → LangChain retriever
```

### Databricks Integration
```python
# TPC-H for Unity Catalog
df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/**/*.csv")
df.write_deltalake("dbfs:/mnt/tpch/lineitem")
```

### HuggingFace Integration
```python
# LAION for dataset hub
df = daft.read_parquet("s3://daft-public-data/tutorials/laion-parquet/*.parquet")
# → Process → Push to HuggingFace datasets
```

## Documentation Quality

Each dataset folder includes:

1. **Comprehensive README** with:
   - Dataset description and structure
   - Complete schema documentation
   - Quick start examples
   - Common use cases with code
   - Performance tips
   - Authentication requirements
   - External resource links

2. **Runnable Examples** with:
   - PEP 723 metadata
   - Clear section headers
   - Sample output expectations
   - Inline comments for complex operations

3. **Progressive Complexity**:
   - Basic examples (load + inspect)
   - Intermediate (transformations)
   - Advanced (ML pipelines, optimization)

## Testing Strategy

**Manual Testing Required:**
- Common Crawl examples need AWS credentials
- Vision models require OpenAI API key
- All other examples work without credentials

**Recommended Testing Order:**
1. Reddit IRL (fastest, no auth)
2. TPC-H (moderate speed, no auth)
3. Open Images (slow, large files)
4. LAION (fast metadata)
5. Common Crawl (slowest, needs AWS)

## Next Steps for Users

### Beginners
Start with Reddit IRL or TPC-H - public, fast, straightforward:
```bash
uv run usage_patterns/datasets/reddit_irl/basic_comments.py
```

### Intermediate
Explore Open Images with vision models:
```bash
export OPENAI_API_KEY="your-key"
uv run usage_patterns/datasets/open_images/vision_models.py
```

### Advanced
Build LLM training pipeline with Common Crawl:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
uv run usage_patterns/datasets/common_crawl/text_deduplication.py
```

## Related Documentation

- **Main README:** `usage_patterns/datasets/README.md`
- **Migration Guide:** `usage_patterns/datasets/MIGRATION_SUMMARY.md`
- **Daft Docs:** [docs.daft.ai](https://docs.daft.ai)
- **Use Cases:** `/use_cases/` folder for production examples

## Contribution Guidelines

When adding new datasets:
1. Follow existing folder structure
2. Include comprehensive README
3. Create 3+ progressive examples
4. Follow print statement policy
5. Add PEP 723 metadata
6. Update main README with dataset entry
7. Test examples (or document testing requirements)
