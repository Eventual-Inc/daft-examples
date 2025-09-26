# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[lance]", "openai", "python-dotenv"]
# ///

import daft
from daft import col
from daft.functions import embed_text, llm_generate, format, unnest

from openai import OpenAI

@daft.func()
def transcribe(file: str) -> {"transcript": str, "segments": list[{"seg_text": str, "seg_start": float, "seg_end": float}]}:
    """
    Transcribes an audio file using openai whisper.
    """
    client = OpenAI()
    with open(file, "rb") as f:
        transcriptions = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        
        segments = [{"seg_text": t.text, "seg_start": t.start, "seg_end": t.end} for t in transcriptions.segments]
        transcript = " ".join([t.text for t in transcriptions.segments])

    return {"transcript": transcript, "segments": segments}

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_uri = "hf://datasets/nvidia/Granary"
    save_uri = "podcasts/podcasts_summaries_openai"
    
    df = (
        # List our files
        daft.from_pydict({"path": [
            "/Users/everett-founder/Downloads/podcasts/joe-rogan-learns-about-jamaican-food-128-ytshorts.savetube.me.mp3",
            "/Users/everett-founder/Downloads/podcasts/taylor-s-about-to-do-a-f-cking-podcast-new-episode-weds-7pm-et-128-ytshorts.savetube.me.mp3",
            "/Users/everett-founder/Downloads/podcasts/why-chefs-are-drunks-128-ytshorts.savetube.me.mp3",
        ]})
        .with_column("transcripts_with_segements", transcribe(col("path")))
    )

    df_transcripts = (
        df
        # Transcribe with Timestamps
        .with_column("transcripts_with_segements", transcribe(col("path")))
        .select(col("path"), unnest(col("transcripts_with_segements")))

        # Summarize the transcript
        .with_column("transcript_summary", llm_generate(format("Summarize the following podcast transcript: {}", col("transcript")), model="gpt-5-nano", provider="openai"))

        # Translate Segment Subtitles to Spanish for Localization
        .with_column("transcript_spanish", llm_generate(format("Translate the following text to Spanish: {}", col("transcript")), model="gpt-5-nano", provider="openai"))
        
        # Embed the transcript and summary
        .with_column("transcript_embeddings", embed_text(col("transcript"), model="text-embedding-ada-002", provider="openai"))
        .with_column("transcript_summary_embeddings", embed_text(col("transcript_summary"), model="text-embedding-ada-002", provider="openai"))
    )

    # Save and Show
    df.write_lance(save_uri)

    # Query the data and display the results
    daft.read_lance(save_uri).show()
