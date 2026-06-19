"""
kokoro_infer.py — Kokoro-82M TTS inference for Mira, run as a subprocess server.

Kokoro (hexgrad/Kokoro-82M) pins numpy==1.26.4 and pulls misaki[en] -> spacy,
none of which have Python 3.14 wheels — so Kokoro CANNOT run inside Mira's main
(Python 3.14) process. Instead it runs here under a dedicated Python 3.10 venv
(.venv-kokoro), the exact same subprocess pattern brocas_area already uses for
RVC. brocas_area launches this once and keeps it hot.

Modes:
  single : one synth then exit
    python kokoro_infer.py --voice af_jessica --lang a --text "hi" --output out.wav
  serve  : persistent — load the model once, then read 'output_wav|text' lines
    python kokoro_infer.py --serve --voice af_jessica --lang a
    (prints READY when hot, then OK / ERR <msg> per request)

Protocol note: requests are 'output_wav|text'. The output path never contains a
'|', so we split on the FIRST '|' and treat the remainder as the text to speak.
Output is always 24 kHz mono PCM-16 WAV (Kokoro's native rate).
"""

import argparse
import sys
import traceback

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000   # Kokoro always emits 24 kHz mono


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serve",  action="store_true", help="persistent server mode (stdin requests)")
    p.add_argument("--text",   default="",  help="text to speak (single mode)")
    p.add_argument("--output", default="",  help="output WAV path (single mode)")
    p.add_argument("--voice",  default="af_jessica", help="Kokoro voice id")
    p.add_argument("--lang",   default="a", help="lang_code: a=American English, b=British, ...")
    p.add_argument("--repo",   default="hexgrad/Kokoro-82M", help="HF repo id for the weights")
    p.add_argument("--speed",  type=float, default=1.0, help="speaking rate")
    return p.parse_args()


def build_pipeline(args):
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code=args.lang, repo_id=args.repo)
    print(f"[kokoro] pipeline ready (voice={args.voice}, lang={args.lang})", file=sys.stderr)
    return pipeline


def synth(pipeline, text, voice, speed, out_path):
    """Synthesize `text` -> 24 kHz WAV at out_path."""
    chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        # Newer Kokoro yields Result objects (.audio); older yields (gs, ps, audio).
        audio = getattr(result, "audio", None)
        if audio is None:
            audio = result[2]
        if hasattr(audio, "detach"):          # torch tensor -> numpy
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
    data = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(out_path, data, SAMPLE_RATE, format="WAV", subtype="PCM_16")


def run_single(args):
    pipeline = build_pipeline(args)
    synth(pipeline, args.text, args.voice, args.speed, args.output)
    print(f"[kokoro] done -> {args.output}", file=sys.stderr)


def run_server(args):
    """Persistent mode: load once, then read 'output_wav|text' lines from stdin.
    Print 'OK' (or 'ERR <msg>') per request so the caller can block."""
    pipeline = build_pipeline(args)
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            out_path, text = line.split("|", 1)
            synth(pipeline, text, args.voice, args.speed, out_path)
            print("OK", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"ERR {e}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.serve:
            run_server(args)
        else:
            run_single(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
