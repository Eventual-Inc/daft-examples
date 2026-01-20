# Google Open Images Validation Dataset

A subset of Google's Open Images dataset containing validation images for computer vision tasks.

## Dataset Location

**S3 Bucket:** `s3://daft-public-data/open-images/validation-images/`

**Authentication:** Public bucket (no AWS credentials required)

## Dataset Structure

```
open-images/
└── validation-images/
    └── *.jpg                # JPEG images
```

## Schema

When loaded with `daft.from_glob_path()`:
```
path: String                  # S3 path to image
size: Int64                   # File size in bytes
image: Image                  # daft.Image object (when decoded)
```

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| **[basic_images.py](./basic_images.py)** | Load and inspect images | Image exploration |
| **[image_processing.py](./image_processing.py)** | Resize and transform images | Image preprocessing |
| **[vision_models.py](./vision_models.py)** | Run vision models on images | Object detection, classification |

## Quick Start

```python
import daft
from daft.functions import file

# Load image paths
df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")

# Decode images
df = df.with_column("image", file(daft.col("path")))

df.show(5)
```

## Common Use Cases

### 1. Load and Decode Images
```python
import daft
from daft.functions import file

df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")
df = df.with_column("image", file(daft.col("path")))
df.show(5)
```

### 2. Image Preprocessing
```python
from daft.functions import image_resize

df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")
df = df.with_column("image", file(daft.col("path")))
df = df.with_column("resized", image_resize(daft.col("image"), width=224, height=224))
df.show(5)
```

### 3. Vision Model Inference
```python
from daft.functions import prompt

df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg").limit(10)
df = df.with_column("image", file(daft.col("path")))
df = df.with_column(
    "description",
    prompt(
        daft.col("image"),
        system_message="Describe this image in one sentence",
        model="gpt-4o-mini",
        provider="openai"
    )
)
df.show(5)
```

### 4. Batch Image Statistics
```python
from daft.functions import image_metadata

df = daft.from_glob_path("s3://daft-public-data/open-images/validation-images/*.jpg")
df = df.with_column("image", file(daft.col("path")))
df = df.with_column("metadata", image_metadata(daft.col("image")))

# Extract dimensions
df = df.with_column("width", daft.col("metadata")["width"])
df = df.with_column("height", daft.col("metadata")["height"])

df.select("width", "height").describe()
```

## Dataset Statistics

- **Images:** ~41,620 validation images
- **Format:** JPEG
- **Size:** Varies (typically 200KB-2MB per image)
- **Content:** Diverse real-world images from Open Images dataset
- **Use case:** Computer vision validation/testing

## Performance Tips

1. **Use `.limit()` during development**: Images are large - test with small batches
2. **Lazy evaluation**: Daft doesn't load images until needed
3. **Batch processing**: Use UDFs for efficient batch inference
4. **Resize early**: Downscale images before expensive operations
5. **Filter by size**: Use file size to filter out very large images

## Image Processing Functions

Daft provides native image operations:

- **`file()`** - Load image from path
- **`image_resize()`** - Resize to target dimensions
- **`image_crop()`** - Crop image region
- **`image_metadata()`** - Extract width, height, channels

## Vision Model Integration

### OpenAI GPT-4 Vision
```python
from daft.functions import prompt

df = df.with_column(
    "caption",
    prompt(daft.col("image"), system_message="Caption this image", model="gpt-4o", provider="openai")
)
```

### Claude Vision
```python
df = df.with_column(
    "analysis",
    prompt(daft.col("image"), system_message="Analyze this image", model="claude-3-5-sonnet-20241022", provider="anthropic")
)
```

### Local Models (HuggingFace Transformers)
```python
from daft.functions import run_inference

df = df.with_column(
    "embeddings",
    run_inference(
        daft.col("image"),
        model="openai/clip-vit-base-patch32",
        provider="transformers"
    )
)
```

## Use Cases

- **Image classification**: Categorize images by content
- **Object detection**: Find objects within images
- **Image captioning**: Generate text descriptions
- **Semantic search**: Find similar images via embeddings
- **Data quality**: Filter corrupt or inappropriate images
- **Model validation**: Test CV models on standard dataset

## Related Datasets

- **Open Images Full Dataset**: [openimages.org](https://storage.googleapis.com/openimages/web/index.html)
- **ImageNet**: Classification benchmark
- **COCO**: Detection and segmentation
- **LAION**: Large-scale image-text pairs

## Resources

- [Daft Image Documentation](https://docs.daft.ai/en/stable/api/dataframe.html#images)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
