"""
Local adapter — today's setup behind the common interface.

Ears  = wernickes_area (faster-whisper continuous listening, local mic)
Mouth = brocas_area     (GPT-SoVITS TTS -> local speaker)

This is a thin wrapper: it delegates to the two existing modules and converts
their three callbacks (on_final / on_partial / on_interrupt) into the single
InputEvent stream the brain consumes. Nothing about the local behaviour changes;
it just gains a uniform shape so a Discord adapter can replace it wholesale.

NOTE: adjust the two imports below if your package import paths differ from the
handoff tree (brain/forebrain/cerebrum/...).
"""

from __future__ import annotations

from typing import Optional

from .io_adapter import IOAdapter, InputEvent, OnEvent, FINAL, PARTIAL, INTERRUPT, PREFILL

# existing modules (paths per the current folder structure)
from brain.forebrain.cerebrum.temporal_lobe import wernickes_area as ears
from brain.forebrain.cerebrum.frontal_lobe import brocas_area as mouth

OWNER = "You"  # the single local speaker (always addressing her directly)


class LocalAdapter(IOAdapter):
    name = "local"

    def __init__(self) -> None:
        self._on_event: Optional[OnEvent] = None

    # --- ears ---
    def start(self, on_event: OnEvent) -> None:
        self._on_event = on_event
        ears.start(
            on_final=self._final,
            on_partial=self._partial,
            on_interrupt=self._interrupt,
            on_prefill=self._prefill_hint,
        )

    def stop(self) -> None:
        ears.stop()

    def pause_input(self) -> None:
        ears.pause()

    def resume_input(self) -> None:
        ears.resume()

    def flush_input(self) -> None:
        ears.flush()

    # wernickes_area callbacks -> InputEvent
    def _emit(self, text: str, kind: str) -> None:
        if self._on_event is not None:
            self._on_event(
                InputEvent(text=text, speaker=OWNER, kind=kind,
                           channel="local", mentioned=True)
            )

    def _final(self, text: str) -> None:
        self._emit(text, FINAL)

    def _partial(self, text: str) -> None:
        self._emit(text, PARTIAL)

    def _interrupt(self, text: str) -> None:
        self._emit(text, INTERRUPT)

    def _prefill_hint(self, text: str) -> None:
        # text="" is the cancel signal (speaker resumed); pass it through unchanged.
        self._emit(text, PREFILL)

    # --- mouth ---
    def speak(self, text: str) -> None:
        mouth.say(text)

    def wait_until_done(self) -> None:
        mouth.wait_until_done()

    def warmup(self) -> None:
        # Heat the EARS (STT): load the Whisper model and run one throwaway forward pass so
        # the first words spoken are transcribed instantly instead of the model cold-loading
        # while the user talks. (The mouth/TTS is warmed separately at startup.) start() then
        # reuses the loaded model and just opens the mic.
        ears.warmup()