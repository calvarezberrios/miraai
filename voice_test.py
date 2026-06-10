"""
voice_test.py - run from D:\aiproject to diagnose the no-audio issue.
Reuses the VOICE config from brocas_area.py.
"""

import io
import sounddevice as sd
import soundfile as sf
from brain.forebrain.cerebrum.frontal_lobe import brocas_area as b

audio = b._synthesize("Testing, one two three.")
print("got audio:", bool(audio), "| bytes:", len(audio) if audio else 0)

if audio:
    with open("test_out.wav", "wb") as f:
        f.write(audio)
    print("saved test_out.wav  <- open it in a media player to confirm sound exists")
    print("output device:", sd.query_devices(kind="output")["name"])
    data, sr = sf.read(io.BytesIO(audio), dtype="float32")
    print("samples:", len(data), "| samplerate:", sr)
    sd.play(data, sr)
    sd.wait()
    print("playback done")
else:
    print("synth returned nothing -- check the [brocas_area] error printed above")