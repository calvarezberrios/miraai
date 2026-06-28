"""
Composite adapter — run more than one "ears + mouth" at once.

The brain is built around ONE active adapter, but a live stream needs two input
sources at the same time:

  * Discord  — you talk to Mira in a voice channel; her replies play back INTO the
               VC, which OBS captures and sends to the stream. This is also her one
               and only MOUTH: everything she says (answers to you, reactions to
               Twitch chat, autonomous host banter) is spoken into the VC so it all
               reaches the stream.
  * Twitch   — she reads live chat (read-only, free) and reacts to it, talking back
               by voice through the Discord VC.

This adapter fans both sources into the single brain callback and routes ALL of her
speech to the voice adapter (Discord). Design points:

  - INPUT SERIALIZATION. Each sub-adapter delivers events from its own thread. A
    FINAL/INTERRUPT turn runs the model (recall + think + speak), so two of them at
    once would mean two model calls fighting over 6GB of VRAM. A single gate lets
    one such turn run at a time; cheap PARTIAL/PREFILL events (live captions,
    speculative warmups) pass straight through for latency.

  - OUTPUT ROUTING. speak()/wait/pause all go to the voice adapter. A turn that came
    from Discord keeps Discord's own target (a voice turn -> VC, a text turn -> that
    text channel, so a DM isn't read aloud on stream). Anything else — Twitch chat,
    or an autonomous daydream that doesn't arrive through a turn — is forced into the
    VC via the voice adapter's set_voice_target().

  - DON'T HEAR HERSELF. pause_input()/resume_input() forward to every sub-adapter, so
    Discord's voice ears are muted while she's speaking a Twitch- or daydream-driven
    line too (not only its own turns).

She must be IN a voice channel for chat-driven / autonomous speech to be heard: say
"mira join" in Discord first. Until then those lines are generated but have nowhere
to play (the VC mouth no-ops) — addressed Discord-voice replies still work as usual.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from .io_adapter import IOAdapter, InputEvent, OnEvent, FINAL, INTERRUPT


class CompositeAdapter(IOAdapter):
    name = "composite"

    def __init__(self, adapters: List[IOAdapter], voice: IOAdapter) -> None:
        """adapters: every source to run. voice: the one that owns the mouth (Discord) —
        it must implement set_voice_target() so non-Discord turns can be aimed at the VC."""
        if voice not in adapters:
            adapters = [voice, *adapters]
        self._adapters = adapters
        self._voice = voice
        self._on_event: Optional[OnEvent] = None
        self._gate = threading.Lock()        # one model-driving turn at a time
        self._active_channel = ""             # channel of the turn currently being handled

    # ---------------- lifecycle ----------------
    def start(self, on_event: OnEvent) -> None:
        self._on_event = on_event
        for a in self._adapters:
            a.start(self._wrap(a))

    def stop(self) -> None:
        for a in self._adapters:
            try:
                a.stop()
            except Exception as e:
                print(f"[composite] {getattr(a, 'name', a)} stop failed: {e}")

    def warmup(self) -> None:
        for a in self._adapters:
            try:
                a.warmup()
            except Exception as e:
                print(f"[composite] {getattr(a, 'name', a)} warmup failed: {e}")

    # ---------------- input fan-in ----------------
    def _wrap(self, source: IOAdapter) -> OnEvent:
        """Wrap the brain callback per source. FINAL/INTERRUPT turns run under the gate
        (serialized across sources); everything else passes straight through."""
        def cb(ev: InputEvent) -> None:
            if self._on_event is None:
                return
            if ev.kind in (FINAL, INTERRUPT):
                with self._gate:
                    self._active_channel = ev.channel
                    try:
                        self._on_event(ev)      # synchronous: recall + think + speak happen here
                    finally:
                        # Reset so a later autonomous daydream (which doesn't arrive as a turn)
                        # routes to the VC rather than inheriting a discord_text target.
                        self._active_channel = ""
            else:
                self._on_event(ev)
        return cb

    # ---------------- mouth (all speech -> the voice adapter) ----------------
    def speak(self, text: str) -> None:
        if not text:
            return
        ch = self._active_channel or ""
        if ch.startswith("discord"):
            # A Discord turn already aimed its target correctly (voice turn -> VC,
            # text turn -> that text channel). Don't override it.
            self._voice.speak(text)
        else:
            # Twitch chat, local, or an autonomous daydream -> speak it in the VC so it's
            # heard on the stream.
            try:
                self._voice.set_voice_target()       # type: ignore[attr-defined]
            except AttributeError:
                pass
            self._voice.speak(text)

    def wait_until_done(self) -> None:
        self._voice.wait_until_done()

    def notify(self, text: str) -> None:
        self._voice.notify(text)

    # ---------------- input gating (forwarded to all sources) ----------------
    def pause_input(self) -> None:
        for a in self._adapters:
            a.pause_input()

    def resume_input(self) -> None:
        for a in self._adapters:
            a.resume_input()

    def flush_input(self) -> None:
        for a in self._adapters:
            a.flush_input()
