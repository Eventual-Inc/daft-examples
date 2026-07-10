from __future__ import annotations

from dataclasses import dataclass

from models.common.speech import dominant_speaker, merge_speakers
from models.parakeet.model import _normalize_nemo_segments
from models.sortformer.model import _parse_turns


@dataclass
class FakeNemoOutput:
    text: str
    timestamp: dict


def test_dominant_speaker_uses_largest_overlap():
    speaker_segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]

    assert dominant_speaker(0.75, 2.5, speaker_segments) == "SPEAKER_01"


def test_merge_speakers_preserves_unmatched_segments_as_empty_speaker():
    segments = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hello", "speaker": "", "words": []},
        {"id": 1, "start": 5.0, "end": 6.0, "text": "later", "speaker": "", "words": []},
    ]
    speaker_segments = [{"start": 0.25, "end": 0.75, "speaker": "SPEAKER_00"}]

    assert merge_speakers(segments, speaker_segments) == [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00", "words": []},
        {"id": 1, "start": 5.0, "end": 6.0, "text": "later", "speaker": "", "words": []},
    ]


def test_parse_turns_accepts_nemo_string_and_nested_shapes():
    assert _parse_turns(["0.0 1.25 speaker_0"]) == [{"start": 0.0, "end": 1.25, "speaker": "speaker_0"}]
    assert _parse_turns([["1.0 2.5 speaker_1"]]) == [{"start": 1.0, "end": 2.5, "speaker": "speaker_1"}]


def test_normalize_nemo_segments_restores_compacted_timestamps_and_words():
    output = FakeNemoOutput(
        text="hello world",
        timestamp={
            "segment": [{"segment": "hello world", "start": 0.0, "end": 1.5}],
            "word": [
                {"word": "hello", "start": 0.2, "end": 0.5},
                {"word": "world", "start": 0.7, "end": 1.1},
            ],
        },
    )
    windows = [
        {
            "compact_start": 0.0,
            "compact_end": 2.0,
            "original_start": 10.0,
            "original_end": 12.0,
        }
    ]

    result = _normalize_nemo_segments(output, duration=2.0, windows=windows, language="en")

    assert result["transcript"] == "hello world"
    assert result["info"] == {"language": "en", "duration": 2.0}
    assert result["segments"] == [
        {
            "id": 0,
            "start": 10.0,
            "end": 11.5,
            "text": "hello world",
            "speaker": "",
            "words": [
                {"text": "hello", "start": 10.2, "end": 10.5},
                {"text": "world", "start": 10.7, "end": 11.1},
            ],
        }
    ]
