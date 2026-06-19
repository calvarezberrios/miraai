import argparse
import datetime
import shutil
import sys
import threading
import time

# The local model occasionally slips an emoji into a reply despite the persona,
# and Mira now also prints lines from a background thread (the subconscious). On a
# legacy-codepage Windows console that would raise UnicodeEncodeError mid-turn, so
# make console output lossy-safe rather than fatal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load .env into os.environ BEFORE importing brain modules — several of them
# (e.g. prefrontal_cortex's MIRA_MODEL) read their config at import time.
from env_loader import load_env
load_env()

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think
from brain.forebrain.cerebrum.frontal_lobe import motor_cortex
from brain.forebrain.cerebrum.frontal_lobe import brocas_area
from brain.forebrain.cerebrum.cingulate_cortex import posterior_cingulate_cortex as subconscious
from brain.hindbrain.cerebellum import coordinator as cerebellum
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


def speak_reply(reply, *, user_text=None, channel="local", interrupting=False,
                speaker=None, source="reply"):
    """The one and only "Mira speaks" sequence, shared by the conscious foreground
    (directly-addressed turns) and the subconscious (overheard chime-ins, spoken
    daydreams). The turn_lock serializes it, so two lines can never overlap on the
    voice/avatar no matter which part of her mind started them.

    user_text is the message she's replying to (None for a spontaneous daydream)."""
    global message_count, last_consolidation
    if not (reply and reply.strip()):
        return
    # A daydream is something she muses ALOUD, not part of the conversation: it must
    # never enter the chat log (working memory / session log) — it already lives in
    # her private subconscious_log. Real replies and chime-ins do get logged.
    is_daydream = source == "daydream"
    with turn_lock:
        if not is_daydream:
            remember_reply(reply)
            observe(user_text if user_text is not None else "", reply)

        if is_daydream:
            print(f"\nMira (daydream, {amygdala.mood}): {reply}\n")
        else:
            if speaker and user_text is not None:
                print(f"\n{speaker}: {user_text}")
            tag = f"Mira ({amygdala.mood})" if source == "reply" else f"Mira (chimes in, {amygdala.mood})"
            print(f"{tag}: {reply}\n")

        cerebellum.gesture_for_speech(reply)          # gesture only if her words call for one
        adapter.pause_input()                         # don't let her hear herself
        adapter.speak(reply)
        adapter.wait_until_done()
        cerebellum.speaking_stopped()                 # close the mouth
        if interrupting:
            adapter.flush_input()                     # drop the ramble she just answered
        adapter.resume_input()

        if not is_daydream:
            # She replied -> keep the active-conversation window alive for this
            # channel, so follow-ups get relevance-checked without re-addressing.
            mark_engaged(channel)
        subconscious.touch()                          # speaking counts as activity (delays wandering)

    if is_daydream:
        return
    # consolidation bookkeeping (a reply-to-someone is an exchange; a chime-in one line)
    message_count += 2 if user_text is not None else 1
    if (message_count >= CONSOLIDATE_EVERY_MESSAGES
            or time.time() - last_consolidation >= CONSOLIDATE_EVERY_SECONDS):
        run_consolidation()
        message_count = 0
        last_consolidation = time.time()


def handle_message(event, interrupting=False):
    now = time.time()

    # She follows the whole room: every message updates short-term context,
    # whether or not she ends up replying to it. (The subconscious reads this.)
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

    feel(event.text)                      # amygdala updates mood
    cerebellum.set_mood(amygdala.mood)    # her face follows the mood
    subconscious.touch(now)               # any input resets the mind-wandering timer

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

    if interrupting or addressed:
        # Directly addressed -> she answers now. While the person was talking, her
        # subconscious was already drafting a reply against the live transcript; if
        # that draft still fits what they actually said, speak it straight away
        # (no transcribe-then-think gap). Otherwise think fresh as a fallback.
        reply = None
        if addressed and not interrupting:
            reply = subconscious.take_draft(event.text)
        else:
            subconscious.end_listening()   # interrupt: don't reuse the draft

        if not reply:
            memories = recall(event.text)
            if previous_session:
                memories = [f"From our last session: {previous_session}"] + memories
            flavor = color() + (INTERRUPT_NOTE if interrupting else "")
            situation = describe_situation(event, prev_seen, now)
            reply = think(context, flavor, memories, situation=situation,
                          inner_thoughts=subconscious.recent_thoughts(event.text))

        speak_reply(reply, user_text=event.text, channel=chan_key,
                    interrupting=interrupting, speaker=event.speaker)
    else:
        # Not addressed -> hand it to the subconscious to mull over in the
        # background. It decides on its own whether/what/when to chime in.
        subconscious.end_listening()       # stop drafting; this wasn't for her to answer now
        if LOG_HEARD:
            print(f"[heard] {event.speaker}: {event.text}")
        subconscious.heard(now, channel=chan_key)


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
        # She listens AND drafts at the same time: each partial refines a reply that
        # will be ready the instant the speaker stops.
        subconscious.observe_partial(ev.text or "", channel=ev.channel)
    elif ev.kind == INTERRUPT:
        handle_message(ev, interrupting=True)
    else:  # FINAL
        handle_message(ev)


try:
    if _startup_last_active is not None:
        away = max(0.0, time.time() - _startup_last_active)
        if away > 60:
            print(f"[Mira's been away — last talked to {_humanize_gap(away)}]\n")

    # Bring up the avatar (body). Non-fatal: if it can't start, the brain/voice
    # still run headless.
    try:
        # Open a browser only for local desktop use; on a server (Discord mode)
        # there's no display — capture the avatar by opening the exposed port.
        motor_cortex.start(open_browser=not args.discord)
        brocas_area.set_lip_callback(cerebellum.lip)   # TTS speech energy -> mouth
    except Exception as e:
        print(f"[avatar not available: {e}]\n")

    adapter.start(on_event)
    adapter.warmup()    # heat up TTS so her first real reply is fast

    # Bring her subconscious online: it listens to everything she overhears and
    # decides on its own when to chime in, and lets her mind wander when it's quiet.
    subconscious.start(speak=speak_reply, session_recap=previous_session)

    while True:
        typed = input().strip()
        if typed.lower() in ("quit", "exit"):
            subconscious.stop()                       # quiet her mind first
            adapter.stop()
            run_consolidation()                       # flush atomic facts
            try:
                if summarize_session():               # write the session recap
                    print("[Mira saved a recap of this session]\n")
            except Exception as e:
                print(f"[session summary skipped: {e}]\n")
            adapter.wait_until_done()                 # let her finish her last line
            motor_cortex.stop()                       # close the avatar server
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