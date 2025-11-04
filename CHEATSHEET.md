# Daft Cheatsheet

* Daft is a Python DataFrame library optimized for multimodal AI workloads, enabling scalable inference on text, images, audio, video, and more.

* It combines DataFrame operations with AI-specific functions like embedding, prompting, classification, and transcription.

* Daft supports distributed execution and integrates seamlessly with providers like OpenAI, Hugging Face Transformers, and more.

* Daft files typically have a .py extension and can be run as scripts or in notebooks.

* Run a Daft script with: `$ python <filename.py>` or in a Jupyter notebook.

#### By default, Daft operates lazily and doesn't require imports beyond `import daft`.

## Create and Execute a Program

1. Open a terminal or Jupyter notebook.
2. Create the program: e.g., `nano daft_ai_example.py` or use an editor.
3. Write the program and save it.
4. Run: `python daft_ai_example.py` or execute in a notebook cell.

<br>

### Basic Data Types in Daft

Daft supports multimodal data types natively for AI tasks.

| Data Type | Description |
| --------- | ----------- |
| Int64 | Integer values [0, 1, -2, 3] |
| Float64 | Floating point values [0.1, 4.532, -5.092] |
| Utf8 | Strings [abc, AbC, A@B, sd!] |
| Boolean | Boolean Values [True, False] |
| Binary | Binary data (e.g., image bytes, audio files) |
| Image | Image data (supports modes like RGB, RGBA; variable or fixed shape) |
| Audio | Audio data (waveform arrays with sample rates) |
| Embedding | Vector embeddings (e.g., Float32 arrays from models) |
| Struct | Structured data (e.g., dictionaries or Pydantic models for AI outputs) |
| List | Lists of values (e.g., sentence chunks, word timestamps) |

<br>

## Key Concepts

- **Lazy Execution**: Operations are planned but not executed until `.collect()` or `.show()` is called.
- **Providers**: Configure AI model providers (e.g., OpenAI, Transformers) for inference.
- **UDFs**: User-defined functions for custom AI logic, decorated with `@daft.func` or `@daft.cls`.
- **Modalities**: Built-in support for text, images, audio, video, embeddings, and structured outputs.
- **IO**: Read from local paths, S3, Hugging Face, or glob patterns for multimodal data.

<br>

## AI Functions

Daft provides high-level AI functions for multimodal inference.

| Function | Description | Example Use Case |
|----------|-------------|-----------------|
| embed_text | Generate vector embeddings from text | Semantic search on documents |
| embed_image | Generate vector embeddings from images | Image similarity search |
| classify_text | Zero-shot text classification | Sentiment analysis on reviews |
| classify_image | Zero-shot image classification | Object detection in photos |
| prompt | Generate text completions or structured outputs from LLMs | Chatbots, content generation |
| decode_image | Decode binary image data | Preprocessing images from bytes |
| resize (Image) | Resize images | Standardizing inputs for models |
| unnest | Expand structured or list columns | Flattening transcription segments |

<br>

## Installation

Install Daft and dependencies for AI workloads:

```bash
pip install daft[ai]  # Includes common AI dependencies like transformers, openai
```

For specific modalities (e.g., audio/video), add extras like `pip install daft[audio,video]`.

<br>

## Create Your First DataFrame

Daft DataFrames are created from Python dictionaries or data sources.

```python
import daft

df = daft.from_pydict({
    "text": ["Hello Daft!", "Multimodal AI is powerful."],
    "labels": [["positive", "neutral"], ["positive", "negative"]]
})
df.show()
```

Output:

```
╭───────────────────────────┬────────────────────────╮
│ text                      ┆ labels                 │
│ ---                       ┆ ---                    │
│ Utf8                      ┆ List[Utf8]             │
╞═══════════════════════════╪════════════════════════╡
│ Hello Daft!               ┆ ["positive", "neutral"]│
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Multimodal AI is powerful.┆ ["positive", "negative"]│
╰───────────────────────────┴────────────────────────╯
```

<br>

## Reading Multimodal Data

Daft supports reading from local files, S3, Hugging Face, or glob patterns. Focus on AI inputs like images, audio, video, PDFs.

- Images: `daft.from_glob_path("path/to/images/*.jpg")`
- Audio/Video: Use `daft.from_glob_path` and decode with UDFs.
- PDFs/Text: Custom UDFs for extraction (e.g., pymupdf for PDFs).

Example: Reading and embedding PDF text.

```python
import daft
from daft.functions import embed_text

uri = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
df = (
    daft.from_glob_path(uri)
    .limit(1)
    .with_column("documents", daft.col("path").url.download())
    # Custom UDF for PDF extraction (see examples/embed_text/embed_pdf_text.py)
    .with_column("pages", extract_pdf_text(daft.col("documents")))
    .explode("pages")
    .with_column("text", daft.col("pages").struct.get("text"))
    .with_column("embeddings", embed_text(daft.col("text"), provider="openai", model="text-embedding-3-small"))
)
df.show()
```

<br>

## Execute and View Data

Daft is lazy; materialize with `.collect()` or `.show()`.

```python
df.show(3)  # View first 3 rows
df.collect()  # Materialize entire DataFrame
```

<br>

## Select Columns

Select specific columns for AI pipelines.

```python
df.select("text", "embeddings").show()
```

<br>

## Filter Rows

Filter data for targeted AI processing (e.g., non-empty text).

```python
df.where(daft.col("text").str.len() > 10).show()
```

<br>

## Transform Columns with Expressions

Use expressions for preprocessing (e.g., normalize text).

```python
df = df.with_column("text_normalized", daft.col("text").normalize(nfd_unicode=True, white_space=True))
df.show()
```

<br>

## Transform with Custom Logic (UDFs)

Define UDFs for AI tasks like transcription or embedding.

Example: Faster Whisper transcription (from transcribe/transcribe_faster_whisper.py).

```python
@daft.cls()
class FasterWhisperTranscriber:
    def __init__(self, model="distil-large-v3"):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self.model = WhisperModel(model)
        self.pipe = BatchedInferencePipeline(self.model)

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File):
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(str(tmp.name), vad_filter=True)
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])
            return {"transcript": text, "segments": segments, "info": asdict(info)}

fwt = FasterWhisperTranscriber()
df = daft.from_glob_path("path/to/audio/*.mp3")
df = df.with_column("audio_file", daft.col("path").file())
df = df.with_column("result", fwt.transcribe(daft.col("audio_file")))
df = df.explode("result.segments").select("path", "result.transcript", daft.unnest(daft.col("result.segments")))
df.show()
```

<br>

## Embedding Text

Generate embeddings for semantic tasks.

```python
df = df.with_column("embeddings", embed_text(daft.col("text"), provider="openai", model="text-embedding-3-small"))
```

<br>

## Embedding Images/Video

Process and embed visual data.

Example: Embed video frames (from embed_images/embed_video_frames.py).

```python
df = daft.from_glob_path("path/to/videos/*.mp4")
df = df.with_column("frames", extract_video_frames(daft.col("path")))  # Custom UDF
df = df.explode("frames")
df = df.with_column("embeddings", embed_image(daft.col("frames").decode_image().resize(224, 224), provider="transformers", model="apple/aimv2-large-patch14-224-lit"))
```

<br>

## Prompting LLMs

Generate text or structured outputs.

Example: Structured prompting (from prompt/prompt_structured_outputs.py).

```python
from pydantic import BaseModel
class Anime(BaseModel):
    show: str
    character: str
    explanation: str

df = daft.from_pydict({"quote": ["I am going to be the king of the pirates!"]})
df = df.with_column("response", prompt(daft.col("quote"), system_message="Classify anime quote", return_format=Anime, provider="openai", model="gpt-3.5-turbo"))
df = df.select("quote", daft.unnest(daft.col("response")))
df.show()
```

<br>

## Classification

Zero-shot classification for text/images.

Example: Classify text (from classify_text/classify_text.py).

```python
df = df.with_column("label", classify_text(daft.col("text"), labels=["positive", "negative"], provider="transformers", model="tabularisai/multilingual-sentiment-analysis"))
```

<br>

## Transcription and Audio Processing

Transcribe audio with VAD.

See Faster Whisper example above.

<br>

## Group and Aggregate

Aggregate AI results (e.g., average embeddings per category).

```python
df.groupby("label").agg(daft.col("embeddings").mean().alias("avg_embedding")).show()
```

<br>

## Sort Data

Sort by AI-derived metrics (e.g., confidence).

```python
df.sort(daft.col("confidence"), desc=True).show()
```

<br>

## Writing Data

Export results to Parquet, CSV, etc.

```python
df.write_parquet("output/path")
```

<br>

## Providers and Sessions

- Set a global provider by name or attach custom providers per session
- Works with OpenAI, OpenRouter (OpenAI-compatible), Transformers, Sentence-Transformers

```python
import os
import daft
from daft.ai.openai.provider import OpenAIProvider

# Global provider (uses OPENAI_API_KEY)
daft.set_provider(
    "openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Attach a custom OpenRouter provider
openrouter = OpenAIProvider(
    name="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
daft.attach_provider(openrouter)
daft.set_provider("OpenRouter")
```

---

## Working with Text

### Text Embeddings

- With Sentence-Transformers (local/CPU/GPU):
```python
import daft
from daft.functions import embed_text

df = daft.from_pydict({"text": ["Hello World", "Daft is fast!"]})
df = df.with_column(
    "embeddings",
    embed_text(
        daft.col("text"),
        provider="sentence_transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
    ),
)
df.show()
```

- With OpenAI (managed API):
```python
df = daft.from_pydict({"text": ["Hello World"]})
df = df.with_column(
    "embeddings",
    embed_text(daft.col("text"), provider="openai", model="text-embedding-3-small"),
)
df.show()
```

### Zero-shot Text Classification

```python
import daft
from daft.functions import classify_text

df = daft.from_pydict({"text": ["Daft is wicked fast!"]})
df = df.with_column(
    "label",
    classify_text(
        daft.col("text"),
        labels=["Positive", "Negative"],
        provider="transformers",
        model="tabularisai/multilingual-sentiment-analysis",
    ),
)
df.show()
```

### Prompting LLMs

- Simple prompt with OpenAI (global provider):
```python
import daft
from daft.functions.ai import prompt

df = daft.from_pydict({
    "quote": [
        "I am going to be the king of the pirates!",
        "I'm going to be the next Hokage!",
    ]
})

df = df.with_column(
    "response",
    prompt(
        daft.col("quote"),
        system_message="Classify the anime and explain.",
        provider="openai",
        model="gpt-5-nano",
    ),
)
df.show(format="fancy", max_width=120)
```

- Structured outputs with Pydantic and custom provider (OpenRouter):
```python
import os
import daft
from pydantic import BaseModel, Field
from daft.functions.ai import prompt
from daft.functions import unnest
from daft.ai.openai.provider import OpenAIProvider

class Anime(BaseModel):
    show: str = Field(description="Anime show name")
    character: str = Field(description="Character name")
    explanation: str = Field(description="Why the quote matters")

openrouter = OpenAIProvider(
    name="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

df = daft.from_pydict({
    "quote": [
        "I am going to be the king of the pirates!",
        "I'm going to be the next Hokage!",
    ],
})

df = (
    df.with_column(
        "response",
        prompt(
            daft.col("quote"),
            system_message="Classify the anime from the quote.",
            return_format=Anime,
            provider=openrouter,
            model="nvidia/nemotron-nano-9b-v2:free",
        ),
    )
    .select("quote", unnest(daft.col("response")))
)

df.show(format="fancy", max_width=120)
```

---

## Working with Images

### Load and Preprocess Images

```python
import daft
from daft.functions import decode_image

# Discover images from HuggingFace
df = (
    daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")
    .with_column("image_bytes", daft.col("path").url.download())
    .with_column("image", decode_image(daft.col("image_bytes")))
    .with_column("image_rgb_288", daft.col("image").convert_image("RGB").resize(288, 288))
)
```

### Image Embeddings

```python
from daft.functions import embed_image

df = df.with_column(
    "image_embeddings",
    embed_image(
        daft.col("image_rgb_288"),
        provider="transformers",
        model="apple/aimv2-large-patch14-224-lit",
    ),
)

df.show()
```

### Image Classification

```python
from daft.functions.ai import classify_image

df = df.with_column(
    "image_label",
    classify_image(
        daft.col("image_rgb_288"),
        labels=[],
        provider="transformers",
        model="google/vit-base-patch16-224",
    ),
)

df.show()
```

---

## Working with Video

### Read and Embed Video Frames

```python
import daft
from daft.functions import embed_image

VIDEOS = [
    "https://www.youtube.com/watch?v=y5hs7q_LaLM",
]
MODEL = "apple/aimv2-large-patch14-224-lit"
H, W = 288, 288

# Read frames (automatically downloads and decodes)
df_frames = daft.read_video_frames(VIDEOS, image_height=H, image_width=W).limit(50)

# Embed frames
df_emb = df_frames.with_column(
    f"img_embeddings_{MODEL}",
    embed_image(daft.col("data"), provider="transformers", model_name=MODEL),
).collect()

df_emb.show()
```

### Shot Boundary Detection (Window Functions)

```python
import numpy as np
import daft
from daft import col, Window, DataType as dt
from daft.functions import embed_image

@daft.func(return_dtype=dt.float32())
def l2_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

VIDEOS = ["https://www.youtube.com/watch?v=y5hs7q_LaLM"]
MODEL = "apple/aimv2-large-patch14-224-lit"
H, W = 288, 288

# Frames and embeddings
df_frames = daft.read_video_frames(VIDEOS, image_height=H, image_width=W).limit(50)
df_emb = df_frames.with_column(
    "img_emb",
    embed_image(col("data"), provider="transformers", model_name=MODEL),
)

# Windows
w = Window().partition_by("path").order_by("frame_time")
w_cut = w.range_between(-0.1, Window.current_row)
w_dissolve = w.range_between(-1.0, Window.current_row)

# Distances and boundaries
df_shots = (
    df_emb.with_column("l2", l2_distance(col("img_emb"), col("img_emb").lag(1).over(w)))
    .with_column("l2_cut", col("l2").mean().over(w_cut))
    .with_column("l2_dissolve", col("l2").mean().over(w_dissolve))
    .with_column("is_cut_boundary", col("l2") >= 0.1)
    .with_column("is_dissolve_boundary", col("l2") >= 1.0)
)

# Frames at detected cuts
df_shots.where(col("is_cut_boundary")).select("data").show()
```

---

## Working with Audio and Transcription

### Read and Resample Audio

```python
import daft
from daft import col
from daft.functions import file
from io.read_audio_file import read_audio, write_audio_to_mp3, sanitize_filename

SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
DEST_URI = ".data/audio/"
TARGET_SR = 16000

df = (
    daft.from_glob_path(SOURCE_URI)
    .with_column("audio_file", file(col("path")))
    .with_column("audio", read_audio(col("audio_file")))
    .with_column("filename_sanitized", sanitize_filename(col("path").split("/").list.get(-1).split(".").list.get(0)))
    .with_column(
        "resampled_path",
        write_audio_to_mp3(
            audio=col("audio").struct.get("audio_array"),
            destination=daft.functions.format("{}{}.mp3", daft.lit(DEST_URI), col("filename_sanitized")),
            sample_rate=TARGET_SR,
        ),
    )
)

df.show()
```

### Faster-Whisper Transcription (with VAD)

```python
from dataclasses import asdict
import daft
from daft import col
from daft.functions import file, unnest
from faster_whisper import WhisperModel, BatchedInferencePipeline
from transcribe.faster_whisper_schema import TranscriptionResult

@daft.cls()
class FasterWhisperTranscriber:
    def __init__(self, model="distil-large-v3", compute_type="float32", device="auto"):
        self.model = WhisperModel(model, compute_type=compute_type, device=device)
        self.pipe = BatchedInferencePipeline(self.model)

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File):
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(
                str(tmp.name), vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True, batch_size=16,
            )
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])
            return {"transcript": text, "segments": segments, "info": asdict(info)}

SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
fwt = FasterWhisperTranscriber()

df_transcript = (
    daft.from_glob_path(SOURCE_URI)
    .with_column("audio_file", file(col("path")))
    .with_column("result", fwt.transcribe(col("audio_file")))
    .select("path", unnest(col("result")))
    .explode("segments")
    .select("path", "info", "transcript", unnest(col("segments")))
).collect()

df_transcript.show(format="fancy", max_width=40)
```

### OpenAI Whisper Transcription (Async)

```python
import daft
from daft import col
from daft.functions import file
from openai import AsyncOpenAI

@daft.cls()
class OpenAITranscription:
    def __init__(self):
        self.client = AsyncOpenAI()

    @daft.method(unnest=True)
    async def transcribe(self, audio_file: daft.File) -> {
        "transcript": str,
        "segments": list[{"seg_text": str, "seg_start": float, "seg_end": float}],
    }:
        with audio_file.to_tempfile() as tmpfile:
            tx = await self.client.audio.transcriptions.create(
                model="whisper-1", file=str(tmpfile.name), response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segs = [{"seg_text": t.text, "seg_start": t.start, "seg_end": t.end} for t in tx.segments]
        return {"transcript": " ".join([t.text for t in tx.segments]), "segments": segs}

SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
oai_tx = OpenAITranscription()

df_transcripts = (
    daft.from_glob_path(SOURCE_URI)
    .with_column("transcript_segments", oai_tx.transcribe(file(col("path"))))
)
```

---

## PDFs and Web-Scale Text

### Read PDFs and Extract Text + Embeddings

```python
import daft
from daft import col
from daft.functions import embed_text, unnest
from embed_text.embed_pdf_text import extract_pdf_text, SpaCyChunkText

URI = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"

# Discover + download PDFs
df = (
    daft.from_glob_path(URI)
    .limit(1)
    .with_column("documents", col("path").url.download())
    .with_column("pages", extract_pdf_text(col("documents")))
    .explode("pages").select(col("path"), unnest(col("pages")))
    .with_column("text_normalized", col("text").normalize(nfd_unicode=True, white_space=True))
    .with_column("sentences", SpaCyChunkText(col("text_normalized")))
    .explode("sentences").select(col("path"), col("page_number"), unnest(col("sentences")))
    .where(col("sent_end") - col("sent_start") > 1)
    .with_column(
        "text_embed_minilm",
        embed_text(col("sent_text"), provider="openai", model="text-embedding-3-small"),
    )
)

df.show()
```

### Common Crawl: Chunk + Embed

```python
import daft
from daft import col
from daft.functions import embed_text, decode
from commoncrawl.chunk_embed import SpacyChunker

# Read preprocessed text
df_wet = daft.datasets.common_crawl("CC-MAIN-2025-33", content="text", num_files=2)

# Chunk and embed
df_embed = (
    df_wet.with_column("spacy_results", SpacyChunker.with_init_args(model="en_core_web_sm")(decode(col("warc_content"), "utf-8")))
    .explode("spacy_results")
    .with_column(
        "embed_minilm",
        embed_text(col("spacy_results")["sent_text"], model="sentence-transformers/all-MiniLM-L6-v2", provider="sentence_transformers"),
    )
)

df_embed.show()
```

---

## UDFs for Custom Logic

- Stateful class UDFs via `@daft.cls()` and `@daft.method()`
- Return types via Python typing, Pydantic, or explicit `DataType`
- Use `unnest=True` to expand struct results into columns

```python
import daft
import typing
import pydantic

class Result(pydantic.BaseModel):
    x3: int
    y3: int
    note: str

@daft.cls()
class Example:
    def __init__(self, x1: int, y1: int):
        self.x1 = x1
        self.y1 = y1

    @daft.method()
    def compute(self, x2: int, y2: int, note: str) -> Result:
        return Result(x3=self.x1 + x2, y3=self.y1 - y2, note=note)

exa = Example(x1=1, y1=2)
df = daft.from_pydict({"x": [1,2,3], "y": [4,5,6]})
df = df.with_column("result", exa.compute(daft.col("x"), daft.col("y"), daft.lit("ok")))
df.show()
```

---

## I/O and Display Tips

- Discover: `daft.from_glob_path("/path/*.ext")`, `daft.read_parquet`, `daft.read_video_frames`, `daft.read_huggingface`
- Download from URLs: `col("path").url.download()`
- Explode nested lists: `.explode("col")`; unnest structs: `select(unnest(col("struct_col")))`
- Materialize: `.collect()`; Inspect: `.show()`
- Persist: `.write_parquet(".data/...")`, `.write_lance(".data/...")`

---

## Notes and Best Practices

- Use provider closest to your deployment constraints: Transformers/Sentence-Transformers for local inference; OpenAI/OpenRouter for hosted models
- Normalize/resize images to match model requirements
- Use UDFs for preprocessing (tokenization, sentence chunking, metrics)
- For long documents: chunk then embed; avoid embedding entire documents at once
- Leverage Window functions for temporal video analytics


### What's Next?

Explore Daft examples for full workflows:
- Embed PDF text (embed_text/embed_pdf_text.py)
- Transcribe audio (transcribe/transcribe_faster_whisper.py)
- Structured prompting (prompt/prompt_structured_outputs.py)

For more, see [Daft Documentation](https://www.getdaft.io/) and [Examples Repo](https://github.com/Eventual-Inc/daft-examples).