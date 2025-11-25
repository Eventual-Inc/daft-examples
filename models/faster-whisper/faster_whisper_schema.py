from daft import DataType

WordStruct = DataType.struct(
    {
        "start": DataType.float64(),
        "end": DataType.float64(),
        "word": DataType.string(),
        "probability": DataType.float64(),
    }
)

SegmentStruct = DataType.struct(
    {
        "id": DataType.int64(),
        "seek": DataType.int64(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "text": DataType.string(),
        "tokens": DataType.list(DataType.int64()),
        "avg_logprob": DataType.float64(),
        "compression_ratio": DataType.float64(),
        "no_speech_prob": DataType.float64(),
        "words": DataType.list(WordStruct),
        "temperature": DataType.float64(),
    }
)

TranscriptionOptionsStruct = DataType.struct(
    {
        "beam_size": DataType.int64(),
        "best_of": DataType.int64(),
        "patience": DataType.float64(),
        "length_penalty": DataType.float64(),
        "repetition_penalty": DataType.float64(),
        "no_repeat_ngram_size": DataType.int64(),
        "log_prob_threshold": DataType.float64(),
        "no_speech_threshold": DataType.float64(),
        "compression_ratio_threshold": DataType.float64(),
        "condition_on_previous_text": DataType.bool(),
        "prompt_reset_on_temperature": DataType.float64(),
        "temperatures": DataType.list(DataType.float64()),
        "initial_prompt": DataType.python(),
        "prefix": DataType.string(),
        "suppress_blank": DataType.bool(),
        "suppress_tokens": DataType.list(DataType.int64()),
        "without_timestamps": DataType.bool(),
        "max_initial_timestamp": DataType.float64(),
        "word_timestamps": DataType.bool(),
        "prepend_punctuations": DataType.string(),
        "append_punctuations": DataType.string(),
        "multilingual": DataType.bool(),
        "max_new_tokens": DataType.float64(),
        "clip_timestamps": DataType.python(),
        "hallucination_silence_threshold": DataType.float64(),
        "hotwords": DataType.string(),
    }
)

VadOptionsStruct = DataType.struct(
    {
        "threshold": DataType.float64(),
        "neg_threshold": DataType.float64(),
        "min_speech_duration_ms": DataType.int64(),
        "max_speech_duration_s": DataType.float64(),
        "min_silence_duration_ms": DataType.int64(),
        "speech_pad_ms": DataType.int64(),
    }
)

LanguageProbStruct = DataType.struct(
    {
        "language": DataType.string(),
        "probability": DataType.float64(),
    }
)

InfoStruct = DataType.struct(
    {
        "language": DataType.string(),
        "language_probability": DataType.float64(),
        "duration": DataType.float64(),
        "duration_after_vad": DataType.float64(),
        "all_language_probs": DataType.list(LanguageProbStruct),
        "transcription_options": TranscriptionOptionsStruct,
        "vad_options": VadOptionsStruct,
    }
)

TranscriptionResult = DataType.struct(
    {
        "transcript": DataType.string(),
        "segments": DataType.list(SegmentStruct),
        "info": InfoStruct,
    }
)
