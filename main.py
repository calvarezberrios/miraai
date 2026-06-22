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

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think, think_stream
from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex
from brain.forebrain.cerebrum.frontal_lobe import motor_cortex
from brain.forebrain.cerebrum.frontal_lobe import brocas_area
from brain.forebrain.cerebrum.frontal_lobe import dorsolateral_prefrontal_cortex as scribe
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
parser.add_argument(
    "--subconscious",
    action="store_true",
    help="Enable the background subconscious: pre-drafting, autonomous chime-ins, and "
         "mind-wandering. Off by default — the plain path is STT -> Mira -> TTS, where she "
         "replies only when addressed.",
)
args = parser.parse_args()

# Simplified pipeline by default: speech in -> Mira thinks (grounded in memories + this
# session) -> speech out, with no background mind. Opt into the full subconscious with
# --subconscious.
USE_SUBCONSCIOUS = args.subconscious

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


def _speak_streaming(stream, *, speaker, user_text, source):
    """Speak Mira's reply sentence-by-sentence as the model writes it (called inside
    turn_lock, by speak_reply). Enqueues each sentence the moment it lands so synthesis +
    playback overlap generation, prints the line as it streams, and returns the full
    assembled text for memory/logging. Pauses the ears (caller resumes them)."""
    if speaker and user_text is not None:
        print(f"\n{speaker}: {user_text}")
    tag = f"Mira ({amygdala.mood})" if source == "reply" else f"Mira (chimes in, {amygdala.mood})"
    adapter.pause_input()                             # don't let her hear herself
    parts = []
    try:
        for sentence in stream:
            if not (sentence and sentence.strip()):
                continue
            if not parts:
                print(f"{tag}: {sentence}", end="", flush=True)
                cerebellum.gesture_for_speech(sentence)   # first sentence sets the gesture
            else:
                print(f" {sentence}", end="", flush=True)
            parts.append(sentence)
            adapter.speak(sentence)                   # returns fast; the queue does the rest
    except Exception as e:
        print(f"\n[stream error: {e}]")
    if parts:
        print()                                       # close the streamed line
    return " ".join(parts).strip()


def speak_reply(reply, *, user_text=None, channel="local", interrupting=False,
                speaker=None, source="reply"):
    """The one and only "Mira speaks" sequence, shared by the conscious foreground
    (directly-addressed turns) and the subconscious (overheard chime-ins, spoken
    daydreams). The turn_lock serializes it, so two lines can never overlap on the
    voice/avatar no matter which part of her mind started them.

    `reply` is either a finished string OR a stream (iterator) of sentences from
    think_stream(): in the streaming case she starts speaking sentence 1 while the model
    is still writing the rest, then the full line is assembled for memory/logging.

    user_text is the message she's replying to (None for a spontaneous daydream)."""
    global message_count, last_consolidation
    streaming = not isinstance(reply, str)
    if not streaming and not (reply and reply.strip()):
        return
    # A daydream is something she muses ALOUD, not part of the conversation: it must
    # never enter the chat log (working memory / session log) — it already lives in
    # her private subconscious_log. Real replies and chime-ins do get logged.
    is_daydream = source == "daydream"
    with turn_lock:
        if streaming:
            # Low-latency: speak each sentence the instant it's generated, then assemble
            # the full line. (Daydreams/chime-ins/pre-drafted replies pass a plain string.)
            reply = _speak_streaming(reply, speaker=speaker, user_text=user_text, source=source)
            if not (reply and reply.strip()):
                adapter.resume_input()                # _speak_streaming paused the ears
                return
            if not is_daydream:
                remember_reply(reply)
                observe(user_text if user_text is not None else "", reply)
        else:
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

            cerebellum.gesture_for_speech(reply)      # gesture only if her words call for one
            adapter.pause_input()                     # don't let her hear herself
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
        if USE_SUBCONSCIOUS:
            subconscious.touch()                      # speaking counts as activity (delays wandering)

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

    # Note-taking mode (DLPFC scribe). When she's taking notes she just listens: this
    # consumes the event (records it, or handles a start/stop/recap/cast command) and
    # she neither replies nor lets her subconscious chime in. Returns False when there's
    # no active session AND this isn't a "take notes" command, so normal chat proceeds.
    if scribe.intercept(event, notify=adapter.notify):
        return

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
    if USE_SUBCONSCIOUS:
        subconscious.touch(now)           # any input resets the mind-wandering timer
        subconscious.note_input(now)      # a human just talked -> allows voicing daydreams again

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
        # Directly addressed -> she answers now, thinking fresh and grounded in this
        # session + her recalled memories. (With --subconscious she may instead reuse a
        # reply her background mind already drafted while the person was still talking.)
        reply = None
        inner_thoughts = None
        if USE_SUBCONSCIOUS:
            if addressed and not interrupting:
                reply = subconscious.take_draft(event.text)
            else:
                subconscious.end_listening()   # interrupt: don't reuse the draft
            inner_thoughts = subconscious.recent_thoughts(event.text)

        if not reply:
            memories = recall(event.text)
            if previous_session:
                memories = [f"From our last session: {previous_session}"] + memories
            flavor = color() + (INTERRUPT_NOTE if interrupting else "")
            situation = describe_situation(event, prev_seen, now)
            # Stream the fresh reply so her voice starts on sentence 1 while the rest is
            # still being written (the big win for longer rants).
            reply = think_stream(context, flavor, memories, situation=situation,
                                 inner_thoughts=inner_thoughts)

        speak_reply(reply, user_text=event.text, channel=chan_key,
                    interrupting=interrupting, speaker=event.speaker)
    elif event.channel == "discord_voice":
        # In a voice channel she's an active participant, not a command bot: she wasn't
        # named, so SHE decides whether to chime in on what was said — joining if it's
        # relevant or about her, staying quiet otherwise. One call both decides and writes
        # the line (returns "" to stay silent).
        memories = recall(event.text)
        if previous_session:
            memories = [f"From our last session: {previous_session}"] + memories
        situation = describe_situation(event, prev_seen, now)
        line = prefrontal_cortex.consider_speaking(
            context, color(), memories, situation=situation)
        if line:
            speak_reply(line, user_text=event.text, channel=chan_key,
                        speaker=event.speaker, source="chime-in")
        elif LOG_HEARD:
            print(f"[heard] {event.speaker}: {event.text}")
    else:
        # Other un-addressed input (e.g. Discord text not @/named). With the subconscious
        # on, hand it off to consider a chime-in; otherwise she just listens.
        if USE_SUBCONSCIOUS:
            subconscious.end_listening()   # stop drafting; this wasn't for her to answer now
        if LOG_HEARD:
            print(f"[heard] {event.speaker}: {event.text}")
        if USE_SUBCONSCIOUS:
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
        # will be ready the instant the speaker stops. (Not while taking notes — then
        # she only listens, so there's no draft and the subconscious is paused.)
        if USE_SUBCONSCIOUS and not scribe.is_active():
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
    # Heat the LLM too: on a CPU-expert model (turbo) the first persona prefill is a ~2 min
    # cold cost — do it in the background now so the user's first turn is warm.
    threading.Thread(target=prefrontal_cortex.warmup, daemon=True).start()

    # Bring her subconscious online (only with --subconscious): it listens to everything
    # she overhears, decides when to chime in, and lets her mind wander when it's quiet.
    if USE_SUBCONSCIOUS:
        subconscious.start(speak=speak_reply, session_recap=previous_session)
        print("[subconscious: ON — drafting, chime-ins, mind-wandering]\n")
    else:
        print("[subconscious: OFF — plain STT -> Mira -> TTS, replies when addressed]\n")

    while True:
        typed = input().strip()
        if typed.lower() in ("quit", "exit"):
            scribe.finalize_if_active(notify=adapter.notify)  # save any open note session
            if USE_SUBCONSCIOUS:
                subconscious.stop()                   # quiet her mind first
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