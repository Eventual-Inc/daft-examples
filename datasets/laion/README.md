# LAION Dataset Samples

LAION (Large-scale Artificial Intelligence Open Network) image-text pairs for training generative models and vision-language tasks.

## Dataset Location

**S3 Buckets:**
- **Parquet:** `s3://daft-public-data/tutorials/laion-parquet/`
- **Images:** `s3://daft-public-data/laion-sample-images/`

**Authentication:** Public bucket (no AWS credentials required)

## Dataset Structure

```
laion-sample-images/              # Sample images
└── *.jpg

tutorials/laion-parquet/          # Metadata in parquet
└── train-00000-of-00001-*.parquet
```

## Schema

### Parquet Metadata
```
URL: String                      # Image URL
TEXT: String                     # Text description/caption
WIDTH: Float64                   # Image width (pixels)
HEIGHT: Float64                  # Image height (pixels)
AESTHETIC_SCORE: Float32         # Aesthetic quality score (0-10)
hash: Int64                      # Image hash
__index_level_0__: Int64         # Index
```

### Image Files
When loaded with `daft.from_glob_path()`:
```
path: String                     # S3 path to image
size: Int64                      # File size in bytes
image: Image                     # daft.Image object (when decoded)
```

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| **[basic_metadata.py](./basic_metadata.py)** | Load LAION metadata | Dataset exploration |
| **[image_text_pairs.py](./image_text_pairs.py)** | Work with image-caption pairs | Multimodal training |
| **[clip_training.py](./clip_training.py)** | Prepare data for CLIP training | Vision-language models |

## Quick Start

### Load Metadata
```python
import daft

df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
)
df.show(5)
```

### Load Images
```python
import daft
from daft.functions import file

df = daft.from_glob_path("s3://daft-public-data/laion-sample-images/*.jpg")
df = df.with_column("image", file(daft.col("path")))
df.show(5)
```

## Common Use Cases

### 1. Explore Metadata
```python
import daft
from daft import col

df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
)

# Caption length analysis
df = df.with_column("caption_length", col("TEXT").length())
df.select("caption_length").describe()

# High quality pairs (high aesthetic score)
high_quality = df.where(col("AESTHETIC_SCORE") > 6.0)
high_quality.select("TEXT", "AESTHETIC_SCORE", "URL").show(10)
```

### 2. Filter by Image Dimensions
```python
# Get square images suitable for training
square_images = df.where(
    (col("WIDTH") == col("HEIGHT")) &
    (col("WIDTH") >= 512)
)
square_images.select("WIDTH", "HEIGHT", "TEXT").show(10)

# Common aspect ratios
df_ratios = df.with_column("aspect_ratio", col("WIDTH") / col("HEIGHT"))
df_ratios.select("aspect_ratio").describe()
```

### 3. Image-Text Embedding
```python
from daft.functions import embed_text, file

# Load parquet with URLs
df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
).limit(10)

# Embed captions
df = df.with_column(
    "text_embedding",
    embed_text(col("TEXT"), model="text-embedding-3-small", provider="openai")
)

# Note: Image embedding would require downloading images from URLs
df.select("TEXT", "text_embedding").show(5)
```

### 4. Caption Analysis
```python
# Most common words in captions
df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
)

# Normalize and analyze captions
df = df.with_column(
    "normalized",
    col("TEXT").normalize(lowercase=True, remove_punct=True)
)

# Sample captions
df.select("TEXT", "AESTHETIC_SCORE").show(10)
```

## Dataset Statistics

### Parquet Metadata
- **Rows:** Sample of full LAION dataset
- **Columns:** URL, caption, CLIP similarity, dimensions
- **Quality:** Pre-filtered by CLIP similarity score

### Sample Images
- **Format:** JPEG
- **Count:** Sample subset
- **Resolution:** Varies (typically 256-1024px)

## Use Cases

### Vision-Language Models
- **CLIP training**: Contrastive learning from image-text pairs
- **Image captioning**: Train models to generate descriptions
- **Text-to-image**: Foundation for generative models (Stable Diffusion, DALL-E)

### Data Quality
- **Caption quality**: Filter by CLIP similarity score
- **Resolution filtering**: Select appropriate dimensions
- **Aspect ratio**: Find images matching target dimensions

### Semantic Search
- **Text → Image**: Find images matching text query
- **Image → Image**: Find visually similar images
- **Multimodal search**: Combined text and visual similarity

## Aesthetic Score

The `AESTHETIC_SCORE` column contains quality scores (0-10):
- **>6.5**: High quality, aesthetically pleasing images
- **6.0-6.5**: Good quality
- **<6.0**: Lower quality

Filter by aesthetic score for better training data:
```python
high_quality = df.where(col("AESTHETIC_SCORE") > 6.0)
```

## Performance Tips

1. **Filter early**: Use similarity and dimension filters before expensive operations
2. **Sample for development**: Use `.limit()` or `.sample_fraction()` for testing
3. **Batch processing**: Process in batches for memory efficiency
4. **Download strategy**: URLs point to external images - plan download strategy carefully

## Training Pipeline Example

```python
import daft
from daft import col
from daft.functions import embed_text

# Load high-quality pairs
df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
)

# Filter for quality and dimensions
df = df.where(
    (col("AESTHETIC_SCORE") > 6.0) &
    (col("WIDTH") >= 256) &
    (col("HEIGHT") >= 256)
)

# Generate text embeddings
df = df.with_column(
    "text_embedding",
    embed_text(col("TEXT"), model="text-embedding-3-small", provider="openai")
)

# Write processed data
df.write_parquet("./laion_processed/")
```

## Full LAION Dataset

This is a **sample** of the full LAION-5B dataset:
- **Full dataset:** 5+ billion image-text pairs
- **Access:** [laion.ai](https://laion.ai/)
- **Formats:** Parquet metadata + image URLs
- **License:** CC-BY-4.0 (metadata), images have individual licenses

## Related Datasets

- **LAION-400M**: 400M English image-text pairs
- **LAION-5B**: 5B multilingual pairs
- **LAION-COCO**: COCO-style captions
- **LAION-Aesthetics**: Aesthetically filtered subset

## Resources

- [LAION Website](https://laion.ai/)
- [LAION-5B Paper](https://arxiv.org/abs/2210.08402)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion](https://stability.ai/) (trained on LAION)
