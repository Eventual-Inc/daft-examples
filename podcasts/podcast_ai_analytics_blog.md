# How to Transcribe, Translate, and Embed Podcasts in under 50 lines

## Introducing Podcast AI Analytics

I love long-form podcasts. They give guests and hosts the chance to dive into complex topics and invite listeners to feel a part of a conversation without the constraints of sound bites or character limits. I'm not the only one. Millions of people around the world tune in to talking heads, not just for the celeb gossip, but for the access to the nuanced and sometimes secret information shared among thought leaders.

I usually discover top podcasts from a clips I see on socials. These clips are the default means for content managers to attract more listeners, serving as a critical marketing tool for businesses and influencers alike. It used to be the case that these clips were curated manually by the podcasting team, but with the significant advances in voice ai in recent years, the entire process is now easily automated.

Most voice ai pipelines includes stages for transcription, summarization, translation, and embeddings. Each of these stages help to enrich the original content from each episode and are generally stored together in a structured way. In general podcast AI analytics pipelines looks something like this:

1. Ingest a directory of podcast audio files.
2. Transcribe episodes into text.
3. Generate summaries of episodes.
4. Generate embeddings of transcripts or summaries.
5. Translate the transcripts to another language for audience localization.
6. Persist results.

Traditional analytics engines have ignored multimodal data types like audio, and for good reason. Reading and processing audio data is fundamentally different from traditional columnar table workloads.

At Eventual, we envision a world where all multimodal ai workloads are trivial to implement and run at scale. We're especially focused on simplifying canonical workloads like long-form audio analytics to help developers focus more on challenges in their domain and less on core structural data processing challenges like working with audio, building for parallelism, managing memory, and other pernicious bottlenecks that plague modern ML pipelines.

### Why Audio is harder than traditional data

If you’ve ever tried to process a folder of raw .mp3 files, you already know the truth: **audio is deceptively hard**. Let’s start with the obvious. Before you can do *anything* useful whether it be transcription, translation, or even a simple word-count, you have to decode, chunk, and buffer it in memory-hungry batches. That means the default data processing model for reading audio is streaming - leveraging generators to yield packets/frames one at a time. One hour of 48 kHz/24-bit stereo audio balloons to **518 MB** of raw samples. That’s larger than the entire works of Shakespeare in plain text, and you haven’t even called your first model yet.
Most machine learning models require mono audio sampled at 16 KHz, usually packaged in 1-D numpy or pytorch arrays. Doing this once is trivial but once you need to scale to distributed computing, there's little support.

### The Traditional Python Script Approach

In this method, developers typically use a single Python environment with libraries such as Librosa for audio manipulation, SpeechRecognition or Whisper (from OpenAI) for transcription, Transformers (from Hugging Face) for translation and summarization, and models like Sentence-BERT or Audio Spectrogram Transformer for embeddings. Processing is often sequential, executed via scripts on local hardware or a modest server.

#### Pain Points by Task

- **Transcription**: Requires loading audio files into memory for feature extraction (e.g., MFCCs or spectrograms), which can fail for long files due to out-of-memory errors.
- **Translation**: Post-transcription text must be fed into NLP models, but handling multilingual audio adds preprocessing steps (e.g., language detection with Langdetect), increasing script complexity and potential for encoding errors in text pipelines.
- **Summarization**: Involves chaining transcription/translation outputs into LLMs, where intermediate data storage in variables or files creates bottlenecks; large batches can exceed memory, forcing manual chunking and error-prone recombination.
- **Embedding**: Generating vector representations from audio or transcribed text demands high computational resources; libraries like Torchaudio may overload CPU/GPU, resulting in slow iteration times and challenges in hyperparameter tuning without distributed computing.

#### Pain Points Across Operations

- **Reading Audio Data**: Libraries like SoundFile or PyDub handle file I/O sequentially, leading to delays for large directories; no built-in support for streaming terabyte-scale data from storage.
- **Parallelism**: Limited to multiprocessing or threading modules, which are error-prone (e.g., GIL limitations in CPython) and require manual process management, often resulting in race conditions or inefficient resource utilization.
- **Memory Management**: Audio data (e.g., WAV files at 44.1 kHz) consumes significant RAM; developers must implement custom generators or lazy loading, but overflows are common without vigilant monitoring.
- **Pipelining**: Stages are hardcoded in scripts, making modifications tedious; errors in one stage (e.g., failed transcription) halt the entire process, with no automatic retry mechanisms.
- **Storing and Querying Data**: Outputs are saved to local files (e.g., JSON/CSV) or databases like SQLite, but querying large datasets involves custom scripts, leading to performance degradation and data inconsistency issues.

Overall, this approach is straightforward for prototyping but scales poorly, with developers facing steep debugging times, lack of fault tolerance, and manual orchestration, often requiring rewrites as data grows.

## How Daft makes it trivial to process audio at scale

Now, imagine if you produce and distribute hundreds of hours of audio content each month. Reading a single multi-hour lossless audio file can easily peak memory into the hundreds of megabytes, which means just reading 100 podcasts requires 10 GB! That doesn't account for any of the data your producing downstream and your workload quickly falls into the "My workload doesn't fit into memory" category.

With Daft, you can run this entire pipeline in just a few lines of code. More importantly the same code scales from laptop to cluster so you never have to rewrite your pipeline. Daft's expressive API handles data, memory, and storage. That means podcasters can focus on their content, not plumbing together frameworks, while still unlocking advanced use cases like multilingual subtitles and Q&A over episodes.

Daft has excellent support for [performing inference on open source models](https://docs.daft.ai/en/stable/examples/). To keep things simple, we'll use OpenAI to run our transcription, summaries, and embeddings. This is where most developers start anyways and helps get us running quickly.

The simplest transcription example from OpenAI looks like:

```python
from openai import OpenAI

file = "./folder/my_podcast.mp3"
client = OpenAI() # Assume we set our OPENAI_API_KEY env var

with open(file, "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=f,
    )
```

In Daft, all we do is wrap this in a row-wise user defined function to give us massive parallelism and scalability from laptop to cluster.

```python
import daft
from openai import OpenAI

@daft.func()
def transcribe(file: str) -> str:
    """
    Transcribes an audio file using openai whisper.
    """
    client = OpenAI()
    with open(file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
        )
    return transcription.text
```

Here, the `@daft.func()` decorator is turning our wrapper function into a massively parallel distributed user-defined-function. When we invoke the udf on our mp3 files, `transcribe` will read a single file from our provided podcast mp3 path, and send the transcription request to OpenAI, returning the text results.

```python
df = (
    # List our files
    daft.from_pydict({
        "paths": [
            "./folder/my_podcast_ep1.mp3",
            "./folder/my_podcast_ep2.mp3",
            "./folder/my_podcast_ep3.mp3",
        ]
    })
    .with_column("transcripts", transcribe(daft.col("path")))
)
```

To invoke our udf, we pass it directly as the primary transformation inside a `with_column()` statement, defining the `"transcripts"` column as the result for all rows.

From here, we can leverage daft's functions for embedding and text generation. We can simply continue the pipeline with daft's built-in `llm_generate` and `embed_text` functions who both support huggingface transformers and openai providers.

```python
from daft.functions import llm_generate, embed_text, format
...

df = (
    df
    # Summarize the transcript
    .with_column(
        "transcript_summary",
        llm_generate(
            format("Summarize the following podcast transcript: {}", daft.col("transcript")), 
            model="gpt-5-nano", 
            provider="openai"
        )
    )

    # Translate Segment Subtitles to Spanish for Localization
    .with_column(
        "transcript_spanish", 
        llm_generate(
            format("Translate the following text to Spanish: {}", daft.col("transcript")), 
            model="gpt-5-nano", 
            provider="openai"
        )
    )
    
    # Embed the transcript and summary
    .with_column(
        "transcript_embeddings", 
        embed_text(
            col("transcript"), 
            model="text-embedding-ada-002", 
            provider="openai"
        )
    )
    .with_column(
        "transcript_summary_embeddings",
        embed_text(
            col("transcript_summary"), 
            model="text-embedding-ada-002", 
            provider="openai"
        )
    )
)
```

To the average python programmer, this syntax may feel a bit alien, so lets take a step back and break down what's happening internally.

What we are looking at is *dataframe* syntax. We aren't manipulating variables, instead we're defining how each column updates the next at the row level. Dataframe syntax is useful for a number of reasons. 

First, it enables developers to perform traditional tabular operations within a managed data model. That means it a lot harder to mess up your data structures since the engine takes care of it for you.

Second, defining your code at this level lets you abstract the complexity of orchestrating your processing for distributed parallelism - maximum CPU and GPU core utilization are active by default.

Third, Daft's execution engine is lazy. That means each operation we apply isn't materialized until we invoke a collection. This is because Daft runs on a push-based processing model, enabling the engine to optimize each operation by planning everything from query through the logic and finally writing to disk. Most importantly, lazy-evaluation helps foces the engine to decouple the tranformations from the load, enabling you, the developer, to focus just on semantic operations instead of worrying if you'r pipeline will work for 10 GB or 10 TB.

### Persisting Results to Disk

Finishing up our pipeline, we have two lines that write our results to lance and then query them back for inspection:

```python
save_uri = "folder/location/"

# Write to Lance (LanceDB table format)
df.write_lance(save_uri)

# Query the data and display the results
daft.read_lance(save_uri).show()
```

This is where daft truly shines. In it's early days, Daft was idealized as the worlds fastest distrbuted query engine, combining big data technology with a syntax that scales from laptop to cluster. Daft's record-setting performance on distributed [S3 reads and writes](https://www.daft.ai/blog/announcing-daft-02) is thanks to how it's maximizes throughput at the I/O level. This breakneck performance is complimented by a wide offering of cloud native integrations for Hive, Apache Iceberg, Delta Lake, and Unity Catalog.

Daft's native integration with [Apache Arrow](https://arrow.apache.org/rust/arrow/index.html) gets you efficient data representation and processing, minimizing CPU and memory overhead through vectorized operations and and SIMD (Single Instruction, Multiple Data) optimizations. A key strength of Apache Arrow is its standardized memory layout, which permits zero-copy data exchange across programming languages without serialization or deserialization overhead.

As Daft has matured and artificial intelligence adoption shifted from text to multimodal it quickly became clear that daft was primed to simplify multimodal ai workloads.

### Inspecting Results

When we finally inspect our data we get:

```bash
╭───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬──────────────────────────┬───────────────────────────╮
│ path                      ┆ transcript                ┆ transcript_summary        ┆ transcript_spanish        ┆ transcript_embeddings    ┆ transcript_summary_embedd │
│ ---                       ┆ ---                       ┆ ---                       ┆ ---                       ┆ ---                      ┆ ings                      │
│ Utf8                      ┆ Utf8                      ┆ Utf8                      ┆ Utf8                      ┆ Embedding[Float32; 1536] ┆ ---                       │
│                           ┆                           ┆                           ┆                           ┆                          ┆ Embedding[Float32; 1536]  │
│                           ┆                           ┆                           ┆                           ┆                          ┆                           │
╞═══════════════════════════╪═══════════════════════════╪═══════════════════════════╪═══════════════════════════╪══════════════════════════╪═══════════════════════════╡
│ /podcasts/my_podcast_e... ┆ What kind of nutrition    ┆ The speaker describes     ┆ ¿Qué tipo de nutrición    ┆ <Embedding>              ┆ <Embedding>               │
│                           ┆ did y…                    ┆ using n…                  ┆ seguis…                   ┆                          ┆                           │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ /podcasts/my_podcast_e... ┆ Such a nice color on you. ┆ A brief, playful exchange ┆ Qué color tan bonito      ┆ <Embedding>              ┆ <Embedding>               │
│                           ┆ Ye…                       ┆ whe…                      ┆ llevas. …                 ┆                          ┆                           │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ /podcasts/my_podcast_e... ┆ I would never be late. My ┆ The speaker asserts       ┆ Yo nunca llegaría tarde.  ┆ <Embedding>              ┆ <Embedding>               │
│                           ┆ co…                       ┆ flawless …                ┆ Mis …                     ┆                          ┆                           │
╰───────────────────────────┴───────────────────────────┴───────────────────────────┴───────────────────────────┴──────────────────────────┴───────────────────────────╯

```

Taking a look at our results, we can see how our pipeline evolved from our podcast file paths to our transcriptions, then llm summary/translations, and embeddings. Daft's native embedding DataType intelligently stores embedding vectors for you, regardless of their size.

### Extensions and Next Steps

Anchoring back to our use-case of Podcast Analytics, transcriptions are great and all, but the reality is we want segment level timestamps for each soundbite so that we can go and generate our short-form marketing.

Both OpenAI's Whisper and other open-source transcription models like NVIDIA's Parakeet support returning timestamped transcripts at either word or segment level. A common next step is to send the timestamped content back to an LLM to extract key moments that you can then recycle as short-form content. These timestamps are also valuable for localization which has been shown to boost audience engagement now that most audiences are global and speak a variety of languages.

The great thing about daft is that extending any of these features is just another line of code. We can also update our transcription UDF to return timestamps as shown below:

```python
import daft
from daft import col
from daft.functions import embed_text, llm_generate, format, unnest
from openai import OpenAI

@daft.func()
def transcribe(file: str) -> {"transcript": str, "segments": list[{"seg_text": str, "seg_start": float, "seg_end": float}]}:
    client = OpenAI()
    with open(file, "rb") as f:
        t = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segments = [{"seg_text": s.text, "seg_start": s.start, "seg_end": s.end} for s in t.segments]
    transcript = " ".join([s.text for s in t.segments])
    return {"transcript": transcript, "segments": segments}

@daft.func()
def dump_segments(segments: list[{"seg_text": str, "seg_start": float, "seg_end": float}]) -> str:
    return json.dumps()

if __name__ == "__main__":
    paths = [
        "/path/to/a.mp3",
        "/path/to/b.mp3",
        "/path/to/c.mp3",
    ]
    save_uri = "podcasts/podcasts_summaries_openai"

    df = (
        daft.from_pydict({"path": paths})
        .with_column(
            "transcripts_with_segments", 
            transcribe(col("path"))
        )
        .select(col("path"), unnest(col("transcripts_with_segments"))) # Unnest Transcript/Segments Struct
        .with_column(
            "transcript_summary", 
            llm_generate(
                format("Summarize: {}", col("transcript")), 
                model="gpt-5-nano", 
                provider="openai"
            )
        )
        .with_column(
            "transcript_spanish", 
            llm_generate(
                format("Translate to Spanish: {}", col("transcript")), 
                model="gpt-5-nano", 
                provider="openai"
            )
        )
        .with_column(
            "transcript_chinese", # Copy/Paste add a new translation
            llm_generate(
                format("Translate to Chinese: {}", col("transcript")), 
                model="gpt-5-nano", 
                provider="openai"
            )
        )
        .with_column(
            "key_moments", # Add Key Moments
            llm_generate(
                format("Identify 3 key moments from the podcast transcript: {}", dump_segments(col("segments"))),
                model="gpt-5-nano", 
                provider="openai",
            )
        )
        .with_column(
            "transcript_embeddings", 
            embed_text(
                col("transcript"), 
                model="text-embedding-ada-002",
                provider="openai"
            )
        )
        .with_column(
            "transcript_summary_embeddings", 
            embed_text(
                col("transcript_summary"), 
                model="text-embedding-ada-002",, 
                provider="openai"
            )
        )
    )
    
    # Write and Show Results
    df.write_lance(save_uri)
    daft.read_lance(save_uri).show()
```

From here there are several directions we could take this. We could leverage the embeddings to host a Q/A chatbot that enables listeners to engage with content across episodes. Questions like "What did Sam Harris say about free will in episode 247?" or "Find all discussions about AI safety across my subscribed podcasts" become trivial vector searches against our stored embeddings. We could build recommendation engines that surface hidden gems based on semantic similarity rather than just metadata tags, or create dynamic highlight reels that auto-generate shareable clips based on sentiment spikes and topic density.

The same tooling that powers each workflow also powers your analytics dashboards showcasing trending topics across the podcast universe, or supply content for automated newsletters that curate personalized episode summaries for each listener's interests. Since everything you store is queryable  and performant, the only limit is your imagination.

A great next step would be to leverage Daft's `cosine_similarity` function makes to put together a full RAG workflow for an interactive ai experience, but I'll let you explore that one on your own. If you run into any issues the eventual team and growing open-source community are always active on [Github](https://github.com/Eventual-Inc/Daft) and [Slack](https://join.slack.com/t/dist-data/shared_invite/zt-2e77olvxw-uyZcPPV1SRchhi8ah6ZCtg). Feel free to introduce yourself or open an issue!

## Conclusion

At Eventual, we're simplifying multimodal AI so that you dont have to. If you are managing voice ai pipelines or managing thousands of hours of podcast audio what you really want is simple:

- Transcripts so your content is accessible and searchable
- Summaries so your listeners can skim and find what matters
- Translations so your audience isn’t limited by language
- Embeddings so people can ask questions like “Which episode talked about reinforcement learning?”

Pulling all that off usually means cobbling together 4–5 different tools, juggling data formats, and dealing with scaling headaches when you go from one episode to hundreds.

With daft you can rest assured that you've committed to a resilient and performant solution for your voice ai workloads. The same data engine that ingests your episodes also persists clean, queryable results to the table format of your choice, and you can immediately read them back to power search, analytics, or retrieval workflows. One tool to process, store, and query means fewer moving parts, fewer places to debug, and a dramatically shorter path from raw audio to usable insights.

For podcasters, that means more time crafting great conversations and less time wrestling with tooling. For developers, it means building the multimodal AI pipeline you actually want—fast to iterate on, easy to scale, and reliable to operate—without drowning in complexity. 
