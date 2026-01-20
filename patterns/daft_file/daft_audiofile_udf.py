# /// script
# description = "Audio-oriented patterns using daft.File + soundfile (duration/sample_rate)"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[audio]>=0.7.1"]
# ///

import daft
from daft.functions import audio_file


@daft.func
def seek_and_read_audio_file(audio_file: daft.AudioFile, offset: int = 0, whence: int = 0)-> bytes:
	"""Seek and read an audio file"""
	with audio_file.open() as f: 
		f.seek(offset, whence)
		audio_bytes = f.read()
		return audio_bytes

df = (
	daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/audio/*.mp3")
	.with_column("file", audio_file(daft.col("path")))
	.with_column("audio", seek_and_read_audio_file(daft.col("file"), offset=4, whence=0))
)

df.show(3)
