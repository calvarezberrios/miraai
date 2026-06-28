"""Minimal .env loader — no external dependency (no python-dotenv).

Reads `KEY=VALUE` lines from a `.env` file in the project root and puts them into
`os.environ`, so config like `MIRA_MODEL` can live in `.env` instead of the system
environment. Call `load_env()` at the very top of the entry point, BEFORE importing
modules that read `os.environ` at import time (e.g. prefrontal_cortex reads
`MIRA_MODEL` when it's imported).

A real, already-set environment variable still wins over `.env` (same behaviour as
python-dotenv's default `override=False`), so you can still override per-run with a
system/shell env var when you want to.
"""
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path: str = _DEFAULT_PATH) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):          # tolerate `export KEY=VALUE`
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]                      # quoted -> literal contents (keep any '#')
            else:
                hash_at = val.find(" #")             # unquoted -> drop an inline " # comment"
                if hash_at != -1:
                    val = val[:hash_at]
                val = val.rstrip()
            if key:
                os.environ.setdefault(key, val)      # don't clobber real env vars
