"""
Peripheral nervous system — I/O adapter contract.

The brain core (prefrontal, thalamus, amygdala, hippocampus, basal ganglia...)
should never know or care WHERE input came from or WHERE its reply goes. An
adapter is a swappable pair of "ears + mouth": local mic/speaker today, a Discord
voice channel tomorrow. Exactly one adapter is active per session (local OR
Discord), selected at launch.

Every input — typed, local mic, or a Discord speaker — arrives as one InputEvent
carrying WHO said it. Locally that's always the owner ("you"); on Discord it's the
speaking member. Same event shape either way, so handle_turn never branches on
source, and Phase 4's attention/turn-taking gets speaker identity for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


# kind values
FINAL = "final"          # a complete utterance -> drive a turn
PARTIAL = "partial"      # live in-progress transcript -> just display
INTERRUPT = "interrupt"  # speaker ran long; partial handed over so Mira can cut in


@dataclass
class InputEvent:
    text: str
    speaker: str                 # "You" locally; member name on Discord
    kind: str = FINAL            # FINAL | PARTIAL | INTERRUPT
    channel: str = "local"       # "local" | "discord_voice" | "discord_text"
    raw: Any = None              # underlying source object, if an adapter needs it
    mentioned: bool = False      # adapter signal: bot was @-mentioned / DM'd / local
    reply_to_her: bool = False   # adapter signal: this message replies to Mira
    is_dm: bool = False          # adapter signal: private DM (no audience)


# An adapter pushes events to this callback; the brain registers one handler.
OnEvent = Callable[[InputEvent], None]


class IOAdapter(ABC):
    """A swappable ears+mouth. Mode = which concrete adapter is active."""

    name: str = "adapter"

    # --- lifecycle / sensory (ears) ---
    @abstractmethod
    def start(self, on_event: OnEvent) -> None:
        """Open the input source; deliver InputEvents to on_event."""

    @abstractmethod
    def stop(self) -> None:
        """Close the input source and release resources."""

    def pause_input(self) -> None:
        """Mute the ears (main loop mutes while Mira speaks). Default: no-op."""

    def resume_input(self) -> None:
        """Un-mute the ears. Default: no-op."""

    def flush_input(self) -> None:
        """Discard captured-but-unfinalized audio (used after an interrupt)."""

    # --- motor (mouth) ---
    @abstractmethod
    def speak(self, text: str) -> None:
        """Enqueue Mira's reply through this channel's mouth. Returns fast."""

    def wait_until_done(self) -> None:
        """Block until everything queued by speak() has finished. Default: no-op."""

    # --- optional ---
    def warmup(self) -> None:
        """Heat up any cold pipeline (e.g. CUDA kernels) at startup. Default: no-op."""