"""
Central config for Mira Live (the new-model, web-UI rebuild).

Everything is env-overridable so start_mira_live.bat is the single place to tune it. Defaults
target: LM Studio serving Hermes-3-Llama-3.1-8B on this laptop, web UI on the LAN.
"""

from __future__ import annotations

import os

# Load .env (DISCORD/TWITCH tokens etc.) if present, same as the rest of the repo.
try:
    import env_loader  # noqa: F401  auto-loads .env into os.environ on import
except Exception:
    pass

# --- LLM (LM Studio, OpenAI-compatible) -------------------------------------
# LM Studio's local server defaults to http://localhost:1234/v1. Start the server in LM Studio
# (Developer tab -> Start Server) with Hermes-3 loaded, GPU offload max, flash attention on.
LLM_BASE_URL = os.environ.get("MIRA_LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LLM_API_KEY = os.environ.get("MIRA_LLM_API_KEY", "lm-studio")  # LM Studio ignores it
# Empty = auto-detect the loaded model from /v1/models (recommended; LM Studio serves whatever's
# loaded). Set MIRA_LLM_MODEL to pin a specific id.
LLM_MODEL = os.environ.get("MIRA_LLM_MODEL", "").strip()

# Context window you configured in LM Studio (for the UI's context meter / new-chat hint).
CONTEXT_LIMIT = int(os.environ.get("MIRA_CONTEXT_LIMIT", "8192"))

# Sampling — a touch warm + repetition controls so she stays fresh and natural, not looping.
# She sees the whole session each turn, so without these a small model parrots its own earlier
# lines. Two layers: OpenAI presence/frequency penalties, AND llama.cpp's stronger samplers
# (classic repeat penalty + the DRY "don't repeat yourself" sampler, which is the real fix for
# verbatim phrase loops). DRY off = dry_multiplier 0.
TEMPERATURE = float(os.environ.get("MIRA_TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("MIRA_TOP_P", "0.9"))
# Keep OpenAI presence/frequency penalties LOW — high values make a small model ramble and
# avoid stopping. Loop prevention is handled by repeat_penalty + DRY below instead.
PRESENCE_PENALTY = float(os.environ.get("MIRA_PRESENCE_PENALTY", "0.3"))
FREQUENCY_PENALTY = float(os.environ.get("MIRA_FREQUENCY_PENALTY", "0.3"))
MAX_TOKENS = int(os.environ.get("MIRA_MAX_TOKENS", "200"))

# llama.cpp samplers (passed through as extra_body; ignored by servers that don't support them).
# DRY is the targeted fix for verbatim phrase loops without hurting coherence/brevity.
REPEAT_PENALTY = float(os.environ.get("MIRA_REPEAT_PENALTY", "1.12"))
REPEAT_LAST_N = int(os.environ.get("MIRA_REPEAT_LAST_N", "320"))
DRY_MULTIPLIER = float(os.environ.get("MIRA_DRY_MULTIPLIER", "0.7"))   # 0 disables DRY
DRY_BASE = float(os.environ.get("MIRA_DRY_BASE", "1.75"))
DRY_ALLOWED_LENGTH = int(os.environ.get("MIRA_DRY_ALLOWED_LENGTH", "2"))

# --- persona ----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PERSONA_PATH = os.environ.get("MIRA_PERSONA_PATH", os.path.join(_HERE, "persona.txt"))


def load_persona() -> str:
    try:
        with open(PERSONA_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[mira_live] could not read persona at {PERSONA_PATH}: {e}")
        return "You are Mira, a sarcastic, playful anime kitsune VTuber. You are not an assistant."


# --- web server -------------------------------------------------------------
HOST = os.environ.get("MIRA_WEB_HOST", "0.0.0.0")   # 0.0.0.0 = reachable on the LAN (for OBS)
PORT = int(os.environ.get("MIRA_WEB_PORT", "8900"))

# Where session transcripts are stored (JSON per session) for the sidebar history.
SESSIONS_DIR = os.environ.get("MIRA_SESSIONS_DIR", os.path.join(_HERE, "sessions"))
