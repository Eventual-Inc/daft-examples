# /// script
# description = "Transcribe + VAD with Faster Whisper"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "faster-whisper"]
# ///
from dataclasses import asdict
import daft
from daft import col
from daft.functions import file, unnest
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper_schema import TranscriptionResult


@daft.cls()
class FasterWhisperTranscriber:
    def __init__(self, model="distil-large-v3", compute_type="float32", device="auto"):
        self.model = WhisperModel(model, compute_type=compute_type, device=device)
        self.pipe = BatchedInferencePipeline(self.model)

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File):
        """Transcribe Audio Files with Voice Activity Detection (VAD) using Faster Whisper"""
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(
                str(tmp.name),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True,
                batch_size=BATCH_SIZE,
            )
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])

            return {"transcript": text, "segments": segments, "info": asdict(info)}


if __name__ == "__main__":
    # Define Parameters & Constants
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    SOURCE_URI = "/Users/everettkleven/Desktop/*.mp3"
    SAMPLE_RATE = 16000
    DTYPE = "float32"
    BATCH_SIZE = 16

    # Instantiate Transcription UDF
    fwt = FasterWhisperTranscriber()

    # Transcribe the audio files
    df_transcript = (
        # Discover the audio files
        daft.from_glob_path(SOURCE_URI)
        # Wrap the path as a daft.File
        .with_column("audio_file", file(col("path")))
        # Transcribe the audio file with Voice Activity Detection (VAD) using Faster Whisper
        .with_column("result", fwt.transcribe(col("audio_file")))
        # Unpack Results
        .select("path", unnest(col("result")))
        .explode("segments")
        .select("path", "info", "transcript", unnest(col("segments")))
    ).collect()

    df_transcript.show(format="fancy", max_width=40)
