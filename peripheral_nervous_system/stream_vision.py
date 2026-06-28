"""
Stream vision — lets Mira SEE what's on screen, so she can talk about the game.

A local vision model won't fit the 6 GB streaming GPU next to Whisper + TTS + the game, so
this OFFLOADS sight the same way the chat brain is offloaded: the desktop just grabs a frame
every few seconds (cheap), shrinks it, and sends it to a multimodal model on the laptop
(Qwen2.5-VL on llama.cpp) for a one-sentence caption. The caption is surfaced as AMBIENT
context (summary()), folded into her situation each turn, so she references what's happening
on screen without it being a "turn" she must answer.

Cost control: periodic frames (every CAPTURE_SEC), NOT a live video feed; the image is
downscaled to MAX_DIM so the VL prompt stays small.

Config (.env / env):
    MIRA_VISION_BASE_URL=http://192.168.12.116:8080/v1   (the laptop VL endpoint; defaults to
                                                          OLLAMA_BASE_URL so one VL model can be
                                                          both her brain and her eyes)
    MIRA_VISION_MODEL=qwen2.5-vl                          (model name the endpoint serves)
    MIRA_VISION_EVERY=8                                   (seconds between captures)
    MIRA_VISION_MONITOR=1                                 (mss monitor index; 1 = primary)
    MIRA_VISION_MAX_DIM=768                               (longest edge sent to the model)
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Optional

CAPTURE_SEC = float(os.environ.get("MIRA_VISION_EVERY", "8"))
MAX_DIM = int(os.environ.get("MIRA_VISION_MAX_DIM", "768"))
# MIRA_VISION_MONITOR: a specific mss monitor index, or unset/"" = auto-pick the PRIMARY
# (main) monitor. Index 0 is the all-screens virtual desktop; 1+ are individual monitors.

_PROMPT = ("In ONE short sentence, describe what is happening on this game/stream screen right "
           "now — name the game or app if recognizable and the current scene or action. Be "
           "concrete and brief. No preamble, just the sentence.")


class StreamVision:
    def __init__(self) -> None:
        self._client = None
        self._model = ""
        self._grab = None                  # callable -> PIL.Image of the screen
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._caption = ""
        self._caption_ts = 0.0

    # ---------------- lifecycle ----------------
    def start(self) -> bool:
        if not self._init_capture():
            return False
        if not self._init_client():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="stream-vision", daemon=True)
        self._thread.start()
        print(f"[vision] watching the screen every {CAPTURE_SEC:.0f}s -> {self._model} "
              f"(captions injected as ambient context).")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _init_capture(self) -> bool:
        """Prefer mss (fast); fall back to PIL.ImageGrab. Both need Pillow for resize/encode."""
        try:
            from PIL import Image  # noqa: F401
        except Exception:
            print("[vision] disabled — needs Pillow:  python -m pip install pillow mss")
            return False
        try:
            import mss
            from PIL import Image
            sct = mss.mss()
            mon, idx = self._pick_monitor(sct)

            def grab():
                shot = sct.grab(mon)
                return Image.frombytes("RGB", shot.size, shot.rgb)
            self._grab = grab
            tag = " (primary/main)" if mon.get("left", 0) == 0 and mon.get("top", 0) == 0 else ""
            print(f"[vision] capturing monitor #{idx}: {mon['width']}x{mon['height']} "
                  f"at ({mon['left']},{mon['top']}){tag}")
            return True
        except Exception:
            try:
                from PIL import ImageGrab
                self._grab = lambda: ImageGrab.grab().convert("RGB")
                return True
            except Exception as e:
                print(f"[vision] disabled — no screen capture backend ({e}); "
                      f"python -m pip install mss pillow")
                return False

    def _pick_monitor(self, sct):
        """Choose which monitor to watch. MIRA_VISION_MONITOR=<index> forces one; otherwise
        auto-pick the PRIMARY (main) monitor — the physical screen whose top-left is (0,0).
        Never index 0 (that's all screens stitched into one giant frame)."""
        mons = sct.monitors                       # [0]=all-screens virtual desktop, [1..]=physical
        want = os.environ.get("MIRA_VISION_MONITOR", "").strip()
        if want.isdigit():
            i = int(want)
            if 0 <= i < len(mons):
                return mons[i], i
        for i in range(1, len(mons)):             # primary = the one anchored at (0,0)
            m = mons[i]
            if m.get("left", 0) == 0 and m.get("top", 0) == 0:
                return m, i
        return (mons[1], 1) if len(mons) > 1 else (mons[0], 0)   # fallback: first physical

    def _init_client(self) -> bool:
        try:
            from openai import OpenAI
        except Exception as e:
            print(f"[vision] disabled — openai client missing: {e}")
            return False
        base = (os.environ.get("MIRA_VISION_BASE_URL", "").strip()
                or os.environ.get("OLLAMA_BASE_URL", "").strip()
                or "http://localhost:11434/v1")
        self._model = os.environ.get("MIRA_VISION_MODEL", "qwen2.5-vl").strip()
        self._client = OpenAI(base_url=base, api_key=os.environ.get("MIRA_VISION_API_KEY", "sk-noauth"))
        print(f"[vision] endpoint {base} (model {self._model})")
        return True

    # ---------------- capture + caption ----------------
    def _frame_data_uri(self) -> Optional[str]:
        from PIL import Image
        img = self._grab()
        if max(img.size) > MAX_DIM:                       # downscale longest edge to MAX_DIM
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _caption_frame(self, data_uri: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            max_tokens=60,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    def _loop(self):
        fails = 0
        while self._running:
            time.sleep(CAPTURE_SEC)
            if not self._running:
                break
            try:
                uri = self._frame_data_uri()
                if not uri:
                    continue
                cap = self._caption_frame(uri)
                if cap:
                    with self._lock:
                        self._caption = cap
                        self._caption_ts = time.time()
                    fails = 0
            except Exception as e:
                fails += 1
                if fails in (1, 5):     # don't spam: report the first, then every few
                    print(f"[vision] caption failed ({e}); is the laptop VL endpoint up?")

    # ---------------- ambient read ----------------
    def summary(self) -> str:
        with self._lock:
            cap, ts = self._caption, self._caption_ts
        if not cap or time.time() - ts > max(CAPTURE_SEC * 4, 30):
            return ""                      # stale (endpoint stalled) -> say nothing rather than lie
        return f"On screen right now: {cap}"
