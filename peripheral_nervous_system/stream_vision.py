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
# How long a caption stays usable before summary() treats it as stale and says nothing. The
# captioner shares the VL server with her chat brain, so under load a caption can be a bit old;
# a generous window means she still describes the screen (a few moments old) instead of falsely
# saying "I can't see." Tune with MIRA_VISION_STALE_SEC.
STALE_SEC = float(os.environ.get("MIRA_VISION_STALE_SEC", "120"))
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
        self._frame_url = ""               # remote frame URL (set in remote mode; mutable for re-point)
        self._opener = None                # proxy-bypass urllib opener for the remote fetch
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
        """Pick the frame SOURCE. If MIRA_VISION_FRAME_URL is set, pull frames from the desktop
        senses companion over the LAN (Mira's on a different machine than the screen). Otherwise
        grab a LOCAL monitor: prefer mss (fast), fall back to PIL.ImageGrab. Either way the frame
        is a PIL.Image that the rest of the pipeline downscales + captions on the local VL."""
        try:
            from PIL import Image  # noqa: F401
        except Exception:
            print("[vision] disabled — needs Pillow:  python -m pip install pillow mss")
            return False
        remote = os.environ.get("MIRA_VISION_FRAME_URL", "").strip()
        if remote:
            return self._init_remote_capture(remote)
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

    def _init_remote_capture(self, url: str) -> bool:
        """Frame source = the desktop senses companion (tools/desktop_senses.py serving /frame).
        Mira runs on the laptop; the screen lives on the desktop, so we fetch the JPEG over the
        LAN instead of grabbing a local monitor. Captioning still happens on the local VL.

        The URL is mutable (`self._frame_url`) so that if the desktop's DHCP IP moves and fetches
        start timing out, _loop can rescan the LAN and re-point without a restart."""
        try:
            import urllib.request
            from PIL import Image
        except Exception as e:
            print(f"[vision] disabled — remote frame needs Pillow ({e})")
            return False
        self._frame_url = url
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LAN: no proxy

        def grab():
            with self._opener.open(self._frame_url, timeout=5) as r:
                data = r.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        self._grab = grab
        print(f"[vision] pulling screen frames from the desktop senses companion: {url}")
        return True

    def _try_rediscover(self) -> bool:
        """The desktop senses companion stopped answering — its DHCP IP may have moved. Rescan
        the LAN for it and re-point self._frame_url. Returns True if we found a new host."""
        from peripheral_nervous_system import lan_discovery as lan
        if not (self._frame_url and lan.enabled()):
            return False
        cur = lan.host_of(self._frame_url)
        print("[vision] frame source unreachable — scanning the LAN for the senses companion...")
        host = lan.find_companion(port=lan.port_of(self._frame_url), exclude=cur)
        if host and host != cur:
            self._frame_url = lan.repoint_url(self._frame_url, host)
            print(f"[vision] found it — re-pointed to {self._frame_url}")
            return True
        print("[vision] companion not found on the LAN (is desktop_senses.py running?).")
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
        first = True
        while self._running:
            # Caption FIRST, sleep after — so the very first caption lands within a few seconds
            # of startup instead of after a full CAPTURE_SEC blind window (the gap where "what do
            # you see?" used to get "I can't see").
            try:
                uri = self._frame_data_uri()
                if uri:
                    cap = self._caption_frame(uri)
                    if cap:
                        with self._lock:
                            self._caption = cap
                            self._caption_ts = time.time()
                        if first:
                            print(f"[vision] eyes online — first caption: {cap}")
                            first = False
                        fails = 0
            except Exception as e:
                fails += 1
                if fails in (1, 5, 20):     # don't spam: report the first, then occasionally
                    print(f"[vision] caption failed ({e}); is the VL endpoint up + the frame "
                          f"source reachable?")
                # Self-heal a moved desktop: after a few straight failures on a REMOTE frame
                # source, rescan the LAN and re-point. Retried periodically while it stays down.
                if getattr(self, "_frame_url", "") and fails in (3, 12, 40):
                    try:
                        if self._try_rediscover():
                            fails = 0
                    except Exception as de:
                        print(f"[vision] rediscover error: {de}")
            time.sleep(CAPTURE_SEC)

    # ---------------- ambient read ----------------
    def summary(self) -> str:
        with self._lock:
            cap, ts = self._caption, self._caption_ts
        if not cap:
            return ""
        age = time.time() - ts
        if age > STALE_SEC:
            return ""                      # endpoint stalled for a long time -> say nothing vs lie
        # Assert she's actually LOOKING at it (the weak VL needs telling it has eyes), and flag
        # when the view is a little old so she doesn't state stale detail as this-instant fact.
        when = "right now" if age <= max(CAPTURE_SEC * 2, 16) else "a few moments ago"
        return (f"You can SEE the stream screen — this is your own live view. On screen {when}: "
                f"{cap} Talk about it naturally as something you're looking at.")
