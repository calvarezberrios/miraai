"""
devices.py - confirm the audio isn't silent, then list output devices.
Run from D:\aiproject after voice_test.py has created test_out.wav.
"""

import soundfile as sf
import sounddevice as sd

data, sr = sf.read("test_out.wav", dtype="float32")
peak = float(abs(data).max())
print("peak amplitude:", peak, "  (near 0 = silent synth; >0.05 = real audio)")
print()
print("OUTPUT DEVICES (use the index number):")
for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        print(i, "|", d["name"])