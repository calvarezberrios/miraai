"""
rvc_infer.py - RVC voice-conversion inference for Mira.

Uses the proven `rvc_python` library (not a hand-rolled model) for crisp output,
with RMVPE F0 estimation. Runs under the GPT-SoVITS bundled runtime Python
(Python 3.9 + torch+cuda + faiss), which is the only interpreter on this box
with the right deps. The main venv (Python 3.14) cannot install them.

Two modes:
  single : one conversion then exit
    python rvc_infer.py --model mira.pth --index mira.index \
        --input in.wav --output out.wav --pitch 0
  serve  : persistent — load models once, then read 'input|output|pitch' lines
    python rvc_infer.py --serve --model mira.pth --index mira.index
    (prints READY when hot, then OK / ERR <msg> per request)

Called as a subprocess by brocas_area.py.
"""

import argparse
import sys
import traceback

from rvc_python.infer import RVCInference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serve",  action="store_true", help="persistent server mode (stdin requests)")
    p.add_argument("--input",  default="", help="input WAV path")
    p.add_argument("--output", default="", help="output WAV path")
    p.add_argument("--model",  required=True, help="RVC model .pth path")
    p.add_argument("--index",  default="",    help="RVC .index path (optional)")
    p.add_argument("--pitch",  type=int, default=0, help="pitch shift in semitones")
    p.add_argument("--index_rate",    type=float, default=0.5)
    p.add_argument("--f0method",      default="rmvpe", help="rmvpe (crisp) | harvest | pm | crepe")
    p.add_argument("--filter_radius", type=int,   default=3)
    p.add_argument("--rms_mix_rate",  type=float, default=0.25)
    p.add_argument("--protect",       type=float, default=0.33)
    return p.parse_args()


def build_engine(args):
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(f"[rvc] device={device}", file=sys.stderr)
    rvc = RVCInference(device=device)
    rvc.load_model(args.model, version="v2", index_path=args.index or "")
    rvc.set_params(
        f0method=args.f0method,
        f0up_key=args.pitch,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )
    print(f"[rvc] model loaded (f0={args.f0method}, index_rate={args.index_rate})", file=sys.stderr)
    return rvc


def run_single(args):
    rvc = build_engine(args)
    rvc.infer_file(args.input, args.output)
    print(f"[rvc] done -> {args.output}", file=sys.stderr)


def run_server(args):
    """Persistent mode: load once, then read 'input|output|pitch' lines from stdin.
    Print 'OK' (or 'ERR <msg>') per request so the caller can block."""
    rvc = build_engine(args)
    print("READY", flush=True)
    cur_pitch = args.pitch
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            parts = line.split("|")
            inp, out = parts[0], parts[1]
            pitch = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            if pitch != cur_pitch:
                rvc.set_params(f0up_key=pitch)
                cur_pitch = pitch
            rvc.infer_file(inp, out)
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
