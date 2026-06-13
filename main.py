import argparse
import datetime
import shutil
import sys
import threading
import time

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think, consider_speaking
from brain.forebrain.subcortical_structures.thalamus import receive, remember_reply
from brain.forebrain.subcortical_structures import hypothalamus
from brain.forebrain.subcortical_structures.limbic_system.amygdala import feel, color
import brain.forebrain.subcortical_structures.limbic_system.amygdala as amygdala
from brain.forebrain.subcortical_structures.limbic_system.hippocampus import (
    recall, observe, consolidate, summarize_session, last_session,
)
from brain.forebrain.subcortical_structures.basal_ganglia.action_selector import (
    should_respond, mark_engaged,
)
from peripheral_nervous_system.io_adapter import InputEvent, PARTIAL, INTERRUPT

CONSOLIDATE_EVERY_MESSAGES = 95       # consolidate after this many messages...
CONSOLIDATE_EVERY_SECONDS = 30 * 60   # ...or this long, whichever comes first
LOG_HEARD = True                      # print messages she hears but doesn't answer

INTERRUPT_NOTE = (
    " The viewer has been talking for a long time without pausing. "
    "Playfully interrupt them mid-thought with a short reaction to what "
    "they've said so far — they are still mid-ramble."
)

# --- mode select: local mic/speaker (default) OR Discord -----------------------
parser = argparse.ArgumentParser(description="Run Mira.")
parser.add_argument(
    "--discord",
    action="store_true",
    help="Run connected to Discord instead of the local mic/speaker.",
)
args = parser.parse_args()

if args.discord:
    from peripheral_nervous_system.discord_adapter import DiscordAdapter
    adapter = DiscordAdapter()
    print("Mira is coming up in Discord mode.\n")
else:
    from peripheral_nervous_system.local_adapter import LocalAdapter
    adapter = LocalAdapter()
    print("Talk to Mira — just speak. (Type 'quit' + Enter to exit; typing a message also works.)\n")

message_count = 0
last_consolidation = time.time()
turn_lock = threading.Lock()          # voice + typed turns never overlap

previous_session = last_session()     # recap of the prior run, if any
if previous_session:
    print(f"[Mira remembers last time: {previous_session}]\n")

# how long she's been away since anyone last talked to her (across the shutdown)
_startup_last_active = hypothalamus.last_active()


def run_consolidation():
    try:
        stored = consolidate()
        if stored:
            print(f"[Mira tucked away {len(stored)} memory(ies)]\n")
    except Exception as e:
        print(f"[consolidation skipped: {e}]\n")


_last_seen = {}   # chan_key -> timestamp of the previous inbound message


def _humanize_gap(seconds):
    if seconds < 90:
        return "a moment ago"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"about {int(round(minutes))} minutes ago"
    hours = minutes / 60.0
    if hours < 24:
        n = int(round(hours))
        return f"about {n} hour{'s' if n != 1 else ''} ago"
    days = int(round(hours / 24.0))
    return f"about {days} day{'s' if days != 1 else ''} ago"


def describe_situation(event, prev_seen, now):
    """A short note about WHEN and WHERE this is happening, for the prompt."""
    parts = []
    dt = datetime.datetime.fromtimestamp(now)
    parts.append("Right now it is " + dt.strftime("%A, %B %d, %Y, at %I:%M %p").lstrip("0") + ".")

    if prev_seen is not None and now - prev_seen > 1:
        parts.append("The previous message before this one came in " + _humanize_gap(now - prev_seen) + ".")

    if getattr(event, "is_dm", False) or event.channel == "local":
        parts.append(
            "This is a private one-on-one conversation (a direct message). No one "
            "else can see it, so do not act as if there is an audience or public chat watching."
        )
    elif str(event.channel).startswith("discord"):
        parts.append(
            "This is a public server channel where other people can see the "
            "conversation and may join in."
        )
    return " ".join(parts)


def handle_message(event, interrupting=False):
    global message_count, last_consolidation
    with turn_lock:
        now = time.time()

        # She follows the whole room: every message updates short-term context,
        # whether or not she ends up replying to it.
        context = receive(event.text)

        chan_key = getattr(event.raw, "id", None)
        chan_key = str(chan_key) if chan_key is not None else event.channel

        # silence tracking: gap since the previous message, then mark this one.
        # On the first message after a restart, bridge from the persisted
        # last-active time so she knows how long she was away.
        prev_seen = _last_seen.get(chan_key)
        if prev_seen is None and _startup_last_active is not None:
            prev_seen = _startup_last_active
        _last_seen[chan_key] = now
        hypothalamus.touch(now)               # persist "last talked to" across restarts

        # Is she being directly addressed? (name / @ / reply / DM)
        addressed = False
        if not interrupting:
            decision = should_respond(
                event.text,
                mentioned=getattr(event, "mentioned", False),
                reply_to_her=getattr(event, "reply_to_her", False),
                channel=chan_key,
            )
            addressed = decision.respond and decision.reason == "addressed"

        # She reacts to the room and recalls regardless of whether she ends up
        # speaking, so the autonomous decision has mood + memory to work with.
        feel(event.text)                              # amygdala updates mood
        memories = recall(event.text)                 # hippocampus -> long-term recall
        if previous_session:
            memories = [f"From our last session: {previous_session}"] + memories

        flavor = color() + (INTERRUPT_NOTE if interrupting else "")
        situation = describe_situation(event, prev_seen, now)

        if interrupting or addressed:
            # Directly addressed (or a long-ramble interrupt) -> she always answers.
            reply = think(context, flavor, memories, situation=situation)
        else:
            # Not addressed -> she decides for herself whether to jump in.
            reply = consider_speaking(context, flavor, memories, situation=situation)
            if not reply:
                if LOG_HEARD:
                    print(f"[heard:quiet] {event.speaker}: {event.text}")
                return

        print(f"\n{event.speaker}: {event.text}")
        remember_reply(reply)
        observe(event.text, reply)
        print(f"Mira ({amygdala.mood}): {reply}\n")

        adapter.pause_input()                         # don't let her hear herself
        adapter.speak(reply)
        adapter.wait_until_done()
        if interrupting:
            adapter.flush_input()                     # drop the ramble she just answered
        adapter.resume_input()

        # She spoke -> keep the active-conversation window alive for this channel,
        # so follow-ups get relevance-checked instead of needing her name again.
        mark_engaged(chan_key)

        message_count += 2
        if (message_count >= CONSOLIDATE_EVERY_MESSAGES
                or time.time() - last_consolidation >= CONSOLIDATE_EVERY_SECONDS):
            run_consolidation()
            message_count = 0
            last_consolidation = time.time()


def on_event(ev):
    """Single entry point for everything the active adapter hears."""
    if ev.kind == PARTIAL:
        # Live "dynamic subtitle": overwrite ONE line in place. Truncate to the
        # terminal width (showing the most recent words) so a long sentence can't
        # wrap onto extra lines and leave stacked-up copies behind. Pad-clear the
        # rest of the line with spaces + '\r' so it works in a plain Windows console.
        cols = shutil.get_terminal_size((100, 20)).columns
        prefix = "[mic] "
        avail = max(10, cols - len(prefix) - 1)
        text = ev.text or ""
        if len(text) > avail:
            text = "..." + text[-(avail - 3):]      # keep the tail (newest words)
        line = prefix + text
        print("\r" + line + " " * max(0, cols - len(line) - 1), end="", flush=True)
    elif ev.kind == INTERRUPT:
        handle_message(ev, interrupting=True)
    else:  # FINAL
        handle_message(ev)


try:
    if _startup_last_active is not None:
        away = max(0.0, time.time() - _startup_last_active)
        if away > 60:
            print(f"[Mira's been away — last talked to {_humanize_gap(away)}]\n")

    adapter.start(on_event)
    adapter.warmup()    # heat up TTS so her first real reply is fast

    while True:
        typed = input().strip()
        if typed.lower() in ("quit", "exit"):
            adapter.stop()
            run_consolidation()                       # flush atomic facts
            try:
                if summarize_session():               # write the session recap
                    print("[Mira saved a recap of this session]\n")
            except Exception as e:
                print(f"[session summary skipped: {e}]\n")
            adapter.wait_until_done()                 # let her finish her last line
            break
        if typed:
            if args.discord:
                print("[typing is off in Discord mode — talk to her in Discord. 'quit' exits.]")
            else:
                # typed local input is always addressed to her
                handle_message(InputEvent(text=typed, speaker="You",
                                          channel="local", mentioned=True))

except NotImplementedError as e:
    # An adapter (or a half-built feature) reported it's not ready yet — exit with
    # a short note instead of a scary traceback. Used by the Discord stub for now,
    # and useful as we build the real adapter piece by piece.
    print(f"\n[Mira isn't ready for that yet] {e}\n")
    sys.exit(0)