"""
motor_cortex.py — voluntary movement output for Mira's body (the avatar).

Brain analogy: the precentral gyrus initiates voluntary movement, mapped to body
parts (the "motor homunculus"). Here it drives the VRM avatar's face: it hosts a
tiny local web server that (a) serves the browser renderer in avatar/ and (b)
streams blendshape targets to it over a WebSocket.

The rest of the brain only calls the small public API:
    start() / stop()          lifecycle
    set_mood(mood)            map an amygdala mood -> facial expression
    lipsync(level)            drive the mouth-open viseme (0..1) during speech
    set_expressions({...})    raw blendshape targets (escape hatch)

Smoothing/timing of these values is the cerebellum's job (applied on top, later);
this module just maps intent -> blendshape targets and broadcasts the latest pose.
The browser also does light per-frame lerping so motion never snaps.

Server runs in its own thread with its own asyncio loop (mirrors the discord
adapter pattern) so the synchronous brain code can call in from any thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import webbrowser
from typing import Dict, Optional

from aiohttp import web, WSMsgType

# --- config -----------------------------------------------------------------
AVATAR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "avatar")
)
# Bind address/port for the avatar web server. Default 0.0.0.0 = listen on ALL interfaces, so
# the renderer can be opened from another machine on the LAN (e.g. view/capture it on the
# desktop while the brain runs on the laptop). Set MIRA_AVATAR_HOST=127.0.0.1 to restrict it
# to this machine only.
HOST = os.environ.get("MIRA_AVATAR_HOST", "0.0.0.0")
PORT = int(os.environ.get("MIRA_AVATAR_PORT", "8234"))


def _lan_ip() -> Optional[str]:
    """Best-effort primary LAN IPv4 of this machine, for the 'open it on another device' URL.
    Opens a UDP socket toward a public IP to pick the outbound interface (no packets are
    actually sent); returns None if it can't determine one."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None

# VRM1 expression presets we drive. three-vrm maps VRM0 blendshape groups onto
# these same names (joy->happy, sorrow->sad, fun->relaxed, a->aa), so one set of
# names works for both VRM0 and VRM1 models.
_EXPRS = ("neutral", "happy", "angry", "sad", "relaxed", "surprised")
_MOUTH = "aa"

# amygdala.mood -> a target expression pose (values 0..1).
MOOD_MAP: Dict[str, Dict[str, float]] = {
    "neutral": {"neutral": 1.0},
    "happy":   {"happy": 0.85},
    "excited": {"happy": 1.0, "surprised": 0.35},
    "annoyed": {"angry": 0.7},
    # graceful extras in case the amygdala vocabulary grows
    "sad":     {"sad": 0.8},
    "flirty":  {"happy": 0.6, "relaxed": 0.4},
}

# Persistent activity STATES the browser knows (set via set_state; see STATE_CLIPS in
# avatar/index.html). idle/talking are procedural living motion; thinking is a looping clip.
STATES = ("idle", "talking", "thinking")
# One-shot GESTURES the browser knows (played via play_gesture; see GESTURES in index.html).
# Fired by what she's saying, they blend in over the current state and auto-return to it.
GESTURES = (
    "wave", "clapping", "flirty", "surprised", "angry", "sad", "jump", "sleepy", "look",
)

# --- server state -----------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_runner: Optional[web.AppRunner] = None
_clients: "set[web.WebSocketResponse]" = set()
_started = threading.Event()

# Latest full pose, so a client that connects (or reconnects) gets the current
# face immediately instead of a blank neutral.
_pose: Dict[str, float] = {e: 0.0 for e in _EXPRS}
_pose["neutral"] = 1.0
_pose[_MOUTH] = 0.0
_pose_lock = threading.Lock()


# --- HTTP / WS handlers -----------------------------------------------------
async def _index(_request):
    return web.FileResponse(os.path.join(AVATAR_DIR, "index.html"))


async def _ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    _clients.add(ws)
    # send the current pose right away
    with _pose_lock:
        snapshot = dict(_pose)
    await ws.send_str(json.dumps({"type": "blendshapes", "values": snapshot}))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
            # renderer is output-only for now; ignore inbound frames
    finally:
        _clients.discard(ws)
    return ws


@web.middleware
async def _no_cache(request, handler):
    """Serve everything no-store. Without this, Chrome aggressively caches the
    localhost HTML + ES modules and keeps running a STALE index.html across
    reloads/relaunches — so renderer edits silently appear to have no effect."""
    resp = await handler(request)
    # The /ws response is already prepared/sent by the time it returns; only
    # touch headers on normal (not-yet-sent) responses.
    if not resp.prepared:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def _build_app() -> web.Application:
    app = web.Application(middlewares=[_no_cache])
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws_handler)
    # everything else (mira.vrm, node_modules/...) served straight from disk
    app.router.add_static("/", AVATAR_DIR, show_index=False)
    return app


def _run_server(open_browser: bool):
    global _loop, _runner
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def _boot():
        global _runner
        _runner = web.AppRunner(_build_app())
        await _runner.setup()
        site = web.TCPSite(_runner, HOST, PORT)
        await site.start()

    _loop.run_until_complete(_boot())
    local_url = f"http://127.0.0.1:{PORT}/"
    print(f"[motor_cortex] avatar server on {local_url}")
    # Listening on all interfaces -> also surface the LAN URL to open from another device.
    if HOST in ("0.0.0.0", "::", ""):
        ip = _lan_ip()
        if ip:
            print(f"[motor_cortex] also reachable on your network at http://{ip}:{PORT}/ "
                  f"- open that URL in a browser on the desktop")
            print(f"[motor_cortex] (if the desktop can't reach it, allow Python through "
                  f"Windows Firewall on Private networks)")
    if open_browser:
        try:
            webbrowser.open(local_url)        # always open the loopback URL locally
        except Exception:
            pass
    _started.set()
    _loop.run_forever()


# --- broadcast --------------------------------------------------------------
async def _broadcast(payload: str):
    if not _clients:
        return
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_str(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def _emit(message: dict):
    """Push a JSON message to every connected renderer (thread-safe)."""
    if _loop is not None and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(json.dumps(message)), _loop)


def _send(values: Dict[str, float]):
    """Merge blendshape targets into the latest pose and broadcast them."""
    with _pose_lock:
        _pose.update(values)
    _emit({"type": "blendshapes", "values": values})


# --- public API -------------------------------------------------------------
def start(open_browser: bool = True, wait: bool = True) -> None:
    """Launch the avatar server thread. Safe to call once."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(
        target=_run_server, args=(open_browser,), daemon=True, name="motor_cortex"
    )
    _thread.start()
    if wait:
        _started.wait(timeout=10)


def stop() -> None:
    if _loop is None or not _loop.is_running():
        return

    async def _shutdown():
        for ws in list(_clients):
            try:
                await ws.close()
            except Exception:
                pass
        _clients.clear()
        if _runner is not None:
            await _runner.cleanup()      # close the site/connections cleanly
        _loop.stop()

    fut = asyncio.run_coroutine_threadsafe(_shutdown(), _loop)
    try:
        fut.result(timeout=5)
    except Exception:
        _loop.call_soon_threadsafe(_loop.stop)


def set_expressions(values: Dict[str, float]) -> None:
    """Raw blendshape targets (0..1). Unknown keys are ignored by the renderer."""
    _send(values)


def set_mood(mood: str) -> None:
    """Map an amygdala mood onto a facial expression pose."""
    pose = MOOD_MAP.get(mood, MOOD_MAP["neutral"])
    # zero every expression we manage, then apply the mood's nonzero ones,
    # so switching moods fully clears the previous face.
    values = {e: 0.0 for e in _EXPRS}
    values.update(pose)
    _send(values)


def lipsync(level: float) -> None:
    """Set the mouth-open viseme (0..1). Used by brocas_area during playback."""
    _send({_MOUTH: max(0.0, min(1.0, float(level)))})


def play_gesture(name: str) -> None:
    """Play a one-shot body gesture, then auto-return to the current state.

    Name is one of GESTURES (wave, clapping, flirty, surprised, ...). Unknown names are
    ignored by the renderer.
    """
    if not name:
        return
    _emit({"type": "gesture", "name": str(name)})


def set_state(name: str) -> None:
    """Set Mira's persistent body activity in the renderer: 'idle', 'talking', or 'thinking'.
    idle/talking are procedural living motion (talking tracks her voice); thinking plays a
    looping clip. The renderer blends from her current pose, so a change never snaps. Unknown
    names are ignored; no-op-safe if the avatar isn't up.
    """
    if name:
        _emit({"type": "state", "name": str(name)})


def subtitle(text: str, duration: float = 0.0) -> None:
    """Show a caption under the avatar, revealed word-by-word across `duration` seconds (the
    spoken line's audio length) so it tracks the voice. Called by brocas_area as each line
    starts playing. Empty text clears it. No-op-safe if the avatar isn't up.
    """
    _emit({"type": "subtitle", "text": str(text or ""), "dur": float(duration or 0.0)})


if __name__ == "__main__":
    # Manual smoke test: open the avatar and cycle a couple expressions.
    import time
    start(open_browser=True)
    print("[motor_cortex] smoke test — cycling expressions. Ctrl+C to quit.")
    try:
        for mood in ["happy", "excited", "annoyed", "neutral"] * 100:
            print("  mood ->", mood)
            set_mood(mood)
            time.sleep(2)
    except KeyboardInterrupt:
        stop()
