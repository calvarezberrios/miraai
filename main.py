import argparse
import datetime
import io
import re
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
from brain.forebrain.cerebrum.frontal_lobe.games import game_master
from brain.forebrain.cerebrum.cingulate_cortex import posterior_cingulate_cortex as subconscious
from brain.hindbrain.cerebellum import coordinator as cerebellum
from brain.forebrain.subcortical_structures.thalamus import receive, remember_reply, snapshot
from brain.forebrain.subcortical_structures import hypothalamus
from brain.forebrain.subcortical_structures.limbic_system.amygdala import feel, color
import brain.forebrain.subcortical_structures.limbic_system.amygdala as amygdala
from brain.forebrain.subcortical_structures.limbic_system.hippocampus import (
    recall, observe, consolidate, summarize_session, last_session,
    recall_document, remember_document, list_documents, forget_document,
)
from brain.forebrain.subcortical_structures.basal_ganglia.action_selector import (
    should_respond, mark_engaged,
)
from peripheral_nervous_system.io_adapter import InputEvent, PARTIAL, INTERRUPT, PREFILL

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
parser.add_argument(
    "--draft",
    action="store_true",
    help="Pre-draft a reply WHILE you're still talking so she answers the instant you stop "
         "(faster turns), WITHOUT the full subconscious's chime-ins or mind-wandering. "
         "Implied by --subconscious.",
)
args = parser.parse_args()

# Simplified pipeline by default: speech in -> Mira thinks (grounded in memories + this
# session) -> speech out, with no background mind. Opt into the full subconscious with
# --subconscious.
USE_SUBCONSCIOUS = args.subconscious
# Pre-drafting (job 1 of the subconscious) — speak faster by having a reply ready the moment
# you stop. The full subconscious includes this; --draft turns ON just the drafting, with no
# chime-ins or daydreams.
DRAFTING = USE_SUBCONSCIOUS or args.draft

if args.discord:
    from peripheral_nervous_system.discord_adapter import DiscordAdapter
    adapter = DiscordAdapter()
    print("Mira is coming up in Discord mode.\n")
else:
    from peripheral_nervous_system.local_adapter import LocalAdapter
    adapter = LocalAdapter()
    print("Mira is starting up — warming the brain, voice, and ears. Please wait to talk...\n")

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


def _vc_human_count(event):
    """How many non-bot humans are in the event's voice channel (0 if not a VC event).
    For discord_voice events, event.raw is the discord VoiceChannel, whose .members lists
    everyone currently in it; we exclude Mira (the bot)."""
    members = getattr(getattr(event, "raw", None), "members", None)
    if not members:
        return 0
    return sum(1 for m in members if not getattr(m, "bot", False))


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
    elif event.channel == "discord_voice":
        humans = _vc_human_count(event)
        if humans <= 1:
            parts.append(
                "You're in a voice call with just one person, and they are talking directly "
                "to you. Respond to them."
            )
        else:
            parts.append(
                f"You're in a voice call with {humans} people. Not everything said is aimed "
                "at you — some of it is people talking to each other. Only jump in if it's a "
                "general topic for the group or it clearly involves you; otherwise stay out "
                "of their side conversation."
            )
    elif str(event.channel).startswith("discord"):
        parts.append(
            "This is a public server channel where other people can see the "
            "conversation and may join in."
        )
    return " ".join(parts)


# --- TTS text cleanup ----------------------------------------------------------
# The LLM sometimes uses Markdown-style emphasis like *finally* or **amusing**.
# Some TTS engines read those literal asterisks aloud, so clean emphasis markers
# immediately before speech synthesis. Action/gesture beats such as *giggles* or
# *tail wag* are preserved so the rest of Mira's body/gesture pipeline can still
# recognize them if needed.
_TTS_ACTION_CUES = {
    "giggle", "giggles", "giggling",
    "laugh", "laughs", "laughing",
    "chuckle", "chuckles", "chuckling",
    "smile", "smiles", "smiling", "smirk", "smirks", "smirking", "grin", "grins", "grinning",
    "blush", "blushes", "blushing",
    "pout", "pouts", "pouting",
    "sigh", "sighs", "sighing",
    "nod", "nods", "nodding", "shake", "shakes", "shaking",
    "wink", "winks", "winking",
    "wave", "waves", "waving",
    "tilt", "tilts", "tilting",
    "shrug", "shrugs", "shrugging",
    "huff", "huffs", "huffing",
    "hmph",
    "tail", "wag", "wags", "wagging",
    "ear", "ears", "flick", "flicks", "flicking", "twitch", "twitches", "twitching",
}

def _is_tts_action_or_gesture(inner_text):
    """Return True for short action beats that should keep their asterisks.

    Examples kept: *giggles*, *tail wag*, *ears twitch*, *pouts playfully*
    Examples cleaned: *finally*, **amusing**, *very important*
    """
    words = re.findall(r"[a-zA-Z']+", (inner_text or "").lower())
    if not words or len(words) > 5:
        return False
    return any(word in _TTS_ACTION_CUES for word in words)


def clean_text_for_tts(text):
    """Strip Markdown emphasis asterisks unless the starred text is an action beat."""
    if not text:
        return text

    def replace_starred(match):
        stars = match.group("stars")
        inner = match.group("inner").strip()
        if _is_tts_action_or_gesture(inner):
            return f"{stars}{inner}{stars}"
        return inner

    # Handles *word/phrase* and **word/phrase** without crossing line breaks.
    return re.sub(
        r"(?P<stars>\*{1,2})(?P<inner>[^*\n]{1,120}?)(?P=stars)",
        replace_starred,
        text,
    )


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
            adapter.speak(clean_text_for_tts(sentence))  # TTS gets emphasis-safe text; print/log keeps original
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
                observe(user_text if user_text is not None else "", reply, speaker=speaker)
        else:
            if not is_daydream:
                remember_reply(reply)
                observe(user_text if user_text is not None else "", reply, speaker=speaker)

            if is_daydream:
                print(f"\nMira (daydream, {amygdala.mood}): {reply}\n")
            else:
                if speaker and user_text is not None:
                    print(f"\n{speaker}: {user_text}")
                tag = f"Mira ({amygdala.mood})" if source == "reply" else f"Mira (chimes in, {amygdala.mood})"
                print(f"{tag}: {reply}\n")

            cerebellum.gesture_for_speech(reply)      # gesture only if her words call for one
            adapter.pause_input()                     # don't let her hear herself
            adapter.speak(clean_text_for_tts(reply))

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


def _recall_for(event):
    """Recall memories relevant to the MESSAGE and to WHO is speaking, then decide whether she
    recognizes them. Recalling on the speaker's name (not just the text) is what surfaces a
    stored identity fact ("Discord user X is GameRaiderX") whenever that person talks, even on an
    unrelated line. Identity gating only applies to Discord speakers; locally the speaker is the
    owner ("You"), so we skip it. Also pulls relevant excerpts from any loaded reference document
    (e.g. a game rulebook). Returns (memories, ident_speaker, speaker_known, documents)."""
    memories = recall(event.text)
    ident_speaker = event.speaker if str(event.channel).startswith("discord") else None
    if ident_speaker:
        # a bare display name lands near the recall distance bar, so query it more leniently
        for m in recall(ident_speaker, max_distance=1.1):
            if m not in memories:
                memories.append(m)
    speaker_known = bool(ident_speaker) and any(
        ident_speaker.lower() in m.lower() for m in memories)
    # reference-document excerpts relevant to this message (rulebooks etc.), labeled by source
    documents = [f"[{r['name']}] {r['text']}" for r in recall_document(event.text)]
    return memories, ident_speaker, speaker_known, documents


# --- speculative prefill (warm the LLM during the end-of-utterance silence) ----------
# When the speaker pauses mid/after an utterance (before the turn is finalized), wernickes
# fires a PREFILL event with the partial transcript. We do the per-turn prep that normally
# only happens AFTER they fully stop — recall the relevant memories (the embedding +
# vector-search wall-clock) and run a 1-token LLM prefill so the server caches this turn's
# prompt prefix — overlapping it with the dead time of the silence window. If the finalized
# turn matches, handle_message reuses the recalled memories and think_stream's first token
# lands off the warm cache. A generation counter makes it cancel-safe: every new or aborted
# prefill bumps it, so a stale in-flight one's result is dropped.
_prefill_lock = threading.Lock()
_prefill_gen = 0
_prefill_cache = {}   # last completed prefill: {text, memories, ident_speaker, speaker_known, documents}


def _prefill_prev_seen(event):
    """The 'previous message' timestamp for this channel, mirroring handle_message."""
    chan_key = getattr(event.raw, "id", None)
    chan_key = str(chan_key) if chan_key is not None else event.channel
    prev_seen = _last_seen.get(chan_key)
    if prev_seen is None and _startup_last_active is not None:
        prev_seen = _startup_last_active
    return prev_seen


def _cancel_prefill():
    """Speaker resumed (or the turn was consumed): invalidate any in-flight/stored prefill."""
    global _prefill_gen
    with _prefill_lock:
        _prefill_gen += 1
        _prefill_cache.clear()


def _speculative_prefill(event):
    """Warm recall + the LLM for `event.text` in the background, during the silence window."""
    # With drafting on, the subconscious already pre-generates a full reply from the partials,
    # so a prefill would be redundant work fighting it for the GPU.
    if DRAFTING:
        return
    text = (event.text or "").strip()
    if not text:
        return
    global _prefill_gen
    with _prefill_lock:
        if _prefill_cache.get("text") == text:
            return                              # already warmed this exact transcript
        _prefill_gen += 1
        gen = _prefill_gen

    def run():
        try:
            memories, ident_speaker, speaker_known, documents = _recall_for(event)
            with _prefill_lock:
                if gen != _prefill_gen:         # superseded/cancelled while recalling
                    return
                _prefill_cache.update(
                    text=text, memories=memories, ident_speaker=ident_speaker,
                    speaker_known=speaker_known, documents=documents)
            # Build the SAME prompt the real turn will and 1-token-warm it: mirror the
            # session-recap prepend and the "Speaker: text" user-turn formatting that
            # handle_message + thalamus.receive produce, so the cached prefix actually matches.
            warm_mem = ([f"From our last session: {previous_session}"] + memories
                        if previous_session else memories)
            history, _ = snapshot()
            ctx = history + [{"role": "user", "content": f"{event.speaker}: {text}"}]
            situation = describe_situation(event, _prefill_prev_seen(event), time.time())
            prefrontal_cortex.prefill(ctx, color(), warm_mem, situation=situation,
                                      speaker=ident_speaker, speaker_known=speaker_known,
                                      documents=documents)
        except Exception as e:
            print(f"[prefill skipped: {e}]")

    threading.Thread(target=run, daemon=True).start()


def _take_prefill(text):
    """If a completed prefill matches this finalized turn, hand back its recalled memories so
    handle_message can skip re-recalling (the LLM prefix is already warm). Consumes the cache."""
    text = (text or "").strip()
    with _prefill_lock:
        if _prefill_cache.get("text") == text:
            cached = (_prefill_cache["memories"], _prefill_cache["ident_speaker"],
                      _prefill_cache["speaker_known"], _prefill_cache["documents"])
            _prefill_cache.clear()
            return cached
    return None


def handle_message(event, interrupting=False):
    now = time.time()

    # Note-taking mode (DLPFC scribe). When she's taking notes she just listens: this
    # consumes the event (records it, or handles a start/stop/recap/cast command) and
    # she neither replies nor lets her subconscious chime in. Returns False when there's
    # no active session AND this isn't a "take notes" command, so normal chat proceeds.
    if scribe.intercept(event, notify=adapter.notify):
        return

    # Deep IQ game mode (Magic: The Gathering vs the "Deep IQ" AI). When a game is active (or the
    # player says "let's play magic"), this owns the turn: a real dice + state engine runs and Mira
    # narrates it. Consumes the event so normal chat doesn't double-reply. Returns False when no
    # game is running and this isn't a start command.
    if game_master.intercept(event, notify=adapter.notify, speak=speak_reply):
        return

    # She follows the whole room: every message updates short-term context,
    # whether or not she ends up replying to it. (The subconscious reads this.)
    # Tag the turn with WHO said it so multi-speaker context stays attributable.
    context = receive(event.text, speaker=event.speaker)

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

    # A 1-on-1 voice call (just her and one other person) means that person is talking TO
    # her -> always respond, like the local mic, no chime-in deliberation. A group VC stays
    # in chime-in mode (the else branch), where she decides per the situation.
    if event.channel == "discord_voice" and _vc_human_count(event) <= 1:
        addressed = True

    # Conversation continuity (text / DM / local): a follow-up WITHOUT her name, while a
    # conversation she's already in is still live in THIS channel, should keep going — she
    # shouldn't need to be re-@'d or re-named on every message. should_respond returns
    # reason="consider" for exactly that (she spoke here within ACTIVE_WINDOW_SEC). A cheap
    # YES/NO relevance check confirms the message actually continues the thread (so she doesn't
    # grab unrelated chatter in a busy channel), then she answers it like an addressed turn.
    # (Voice has its own chime-in path below.)
    if (not interrupting and not addressed and event.channel != "discord_voice"
            and decision.reason == "consider"
            and prefrontal_cortex.judge_relevance(context)):
        addressed = True

    if interrupting or addressed:
        # Directly addressed -> she answers now, thinking fresh and grounded in this
        # session + her recalled memories. (With --subconscious she may instead reuse a
        # reply her background mind already drafted while the person was still talking.)
        reply = None
        inner_thoughts = None
        if DRAFTING:
            if addressed and not interrupting:
                reply = subconscious.take_draft(event.text)   # the reply she drafted while you talked
            else:
                subconscious.end_listening()   # interrupt: don't reuse the draft
            if USE_SUBCONSCIOUS:
                inner_thoughts = subconscious.recent_thoughts(event.text)

        if not reply:
            # Reuse the speculative prefill if it warmed this exact turn (recall already done,
            # LLM prefix already cached); otherwise recall now.
            cached = None if interrupting else _take_prefill(event.text)
            if cached:
                memories, ident_speaker, speaker_known, documents = cached
            else:
                memories, ident_speaker, speaker_known, documents = _recall_for(event)
            if previous_session:
                memories = [f"From our last session: {previous_session}"] + memories
            flavor = color() + (INTERRUPT_NOTE if interrupting else "")
            situation = describe_situation(event, prev_seen, now)
            # Stream the fresh reply so her voice starts on sentence 1 while the rest is
            # still being written (the big win for longer rants).
            reply = think_stream(context, flavor, memories, situation=situation,
                                 inner_thoughts=inner_thoughts,
                                 speaker=ident_speaker, speaker_known=speaker_known,
                                 documents=documents)

        speak_reply(reply, user_text=event.text, channel=chan_key,
                    interrupting=interrupting, speaker=event.speaker)
    elif event.channel == "discord_voice":
        # In a voice channel she's an active participant, not a command bot: she wasn't
        # named, so SHE decides whether to chime in on what was said — joining if it's
        # relevant or about her, staying quiet otherwise. One call both decides and writes
        # the line (returns "" to stay silent).
        memories, ident_speaker, speaker_known, documents = _recall_for(event)
        if previous_session:
            memories = [f"From our last session: {previous_session}"] + memories
        situation = describe_situation(event, prev_seen, now)
        line = prefrontal_cortex.consider_speaking(
            context, color(), memories, situation=situation,
            speaker=ident_speaker, speaker_known=speaker_known, documents=documents)
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
        if DRAFTING and not scribe.is_active():
            subconscious.observe_partial(ev.text or "", channel=ev.channel, speaker=ev.speaker)
    elif ev.kind == PREFILL:
        # Brief pause in an utterance: warm the LLM on the partial (text), or cancel if the
        # speaker resumed (empty text). Never while taking notes — she only listens then.
        if ev.text and ev.text.strip():
            if not scribe.is_active():
                _speculative_prefill(ev)
        else:
            _cancel_prefill()
    elif ev.kind == INTERRUPT:
        _clear_mic_line()
        handle_message(ev, interrupting=True)
    else:  # FINAL
        # The utterance is done: wipe the rolling [mic] caption so the committed turn
        # ("Speaker: text" + her reply, or a "[heard]" line) replaces it in place, instead
        # of the same transcript showing again underneath it.
        _clear_mic_line()
        handle_message(ev)


def _clear_mic_line():
    """Erase the in-place [mic] live-caption line (no-op-safe if there isn't one — it just
    blanks the current empty console line, e.g. in Discord-text mode)."""
    cols = shutil.get_terminal_size((100, 20)).columns
    print("\r" + " " * (cols - 1) + "\r", end="", flush=True)


def _warm_all():
    """Warm all three engines behind ONE progress bar, blocking until everything is hot so the
    very first turn (hear -> think -> speak) is instant and the user never talks into a cold
    pipeline. Returns when brain, voice, and ears are all warm:
      - brain (LLM persona prefill — turbo's first prefill is a ~2 min cold cost)
      - voice (Kokoro TTS — first synth loads the model / downloads weights)
      - ears  (Whisper STT — model load + first forward pass)
    The brain (long pole) warms in the background while voice + ears warm sequentially (kept
    serial to avoid GPU contention). The bar fills as stages finish; the in-progress stage eases
    its slice forward over time so the bar keeps moving even though each step is opaque."""
    # weight = share of the bar; eta = soft estimate used ONLY to animate the in-progress fill
    # (the slice eases toward its boundary and snaps to full when the stage actually completes).
    engines = {
        "brain": {"weight": 0.50, "eta": 90.0, "done": False, "ok": False, "label": "language model"},
        "voice": {"weight": 0.25, "eta": 12.0, "done": False, "ok": False, "label": "voice engine"},
        "ears":  {"weight": 0.25, "eta": 8.0,  "done": False, "ok": False, "label": "speech recognition"},
    }
    starts = {}

    def warm_brain():
        starts["brain"] = time.time()
        try:
            prefrontal_cortex.warmup()
            engines["brain"]["ok"] = True
        except Exception as e:
            print(f"\n[brain] warmup error: {e}")
        engines["brain"]["done"] = True

    def warm_voice_then_ears():
        # voice first, then ears — serial so they don't fight over the GPU during load.
        starts["voice"] = time.time()
        def vprog(stage):
            engines["voice"]["label"] = stage
        try:
            engines["voice"]["ok"] = bool(brocas_area.warmup(progress=vprog, blocking=True))
        except Exception as e:
            print(f"\n[voice] warmup error: {e}")
        engines["voice"]["done"] = True
        starts["ears"] = time.time()
        try:
            adapter.warmup()             # LocalAdapter -> ears.warmup(); no-op for Discord
            engines["ears"]["ok"] = True
        except Exception as e:
            print(f"\n[ears] warmup error: {e}")
        engines["ears"]["done"] = True

    # Keep the bar on ONE continuously-redrawn line: send the warmup threads' background log
    # lines (whisper "loading...", Kokoro "server ready", "brain warmed") into a buffer so they
    # can't break the bar onto new lines. The bar writes straight to the REAL stdout; the captured
    # logs are replayed once the bar finishes, so nothing is lost.
    real_out = sys.stdout
    real_err = sys.stderr
    captured = io.StringIO()

    def _draw(text, *, newline=False):
        cols = shutil.get_terminal_size((100, 20)).columns
        real_out.write("\r" + text[:cols - 1] + " " * max(0, cols - len(text) - 1) + ("\n" if newline else ""))
        real_out.flush()

    t0 = time.time()
    sys.stdout = captured                          # warmup threads print here, not over the bar
    sys.stderr = captured                          # ...and their warnings too (HF/torch noise)
    try:
        threading.Thread(target=warm_brain, daemon=True).start()
        threading.Thread(target=warm_voice_then_ears, daemon=True).start()

        width = 30
        while not all(e["done"] for e in engines.values()):
            # overall fraction = finished slices + the eased fill of any in-progress slice
            frac = 0.0
            for name in ("brain", "voice", "ears"):
                e = engines[name]
                if e["done"]:
                    frac += e["weight"]
                elif name in starts:
                    el = time.time() - starts[name]
                    sub = 1.0 - 0.5 ** (el / e["eta"])  # 0 -> ~1 asymptotically; never quite reaches 1
                    frac += e["weight"] * min(sub, 0.97)
            # label: show whatever foreground stage is running (voice/ears), else the brain.
            if not engines["voice"]["done"]:
                label = "warming voice — " + engines["voice"]["label"]
            elif not engines["ears"]["done"]:
                label = "warming speech recognition"
            else:
                label = "warming language model"
            filled = int(round(width * frac))
            bar = "█" * filled + "░" * (width - filled)
            el = int(time.time() - t0)
            _draw(f"[{bar}] {int(round(frac * 100)):3d}%  {label} ({el}s)")
            time.sleep(0.1)

        el = int(time.time() - t0)
        _draw(f"[{'█' * width}] 100%  all engines warm in {el}s", newline=True)
    finally:
        sys.stdout = real_out
        sys.stderr = real_err

    # On success the captured output is just warmup noise (HF/torch warnings, "ready"
    # confirmations) — discard it so the bar stays the only thing on screen. Surface it ONLY
    # if an engine failed, so the actual error is visible; then note what'll load lazily.
    failed = [name for name in ("brain", "voice", "ears") if not engines[name]["ok"]]
    if failed:
        logs = captured.getvalue().strip()
        if logs:
            print("[warmup] details:")
            for ln in logs.splitlines():
                print("  " + ln)
        for name in failed:
            print(f"[{name}] warmup didn't complete; it'll load on first use.")


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

    # Warm all three engines (brain LLM, voice TTS, ears STT) behind one progress bar BEFORE we
    # open the ears, so you never talk into a cold pipeline. Until this returns the mic is closed
    # and there's nothing to type into, so an early word can't get half-transcribed (which is what
    # made that first transcript lag).
    _warm_all()

    # Everything's hot — NOW open the ears (the model is already loaded, so the mic opens at once).
    adapter.start(on_event)

    # Bring her subconscious online (only with --subconscious): it listens to everything
    # she overhears, decides when to chime in, and lets her mind wander when it's quiet.
    if USE_SUBCONSCIOUS:
        subconscious.start(speak=speak_reply, session_recap=previous_session)
        print("[subconscious: ON — drafting, chime-ins, mind-wandering]\n")
    elif DRAFTING:
        subconscious.start(speak=speak_reply, session_recap=previous_session, draft_only=True)
        print("[drafting: ON — she drafts a reply while you talk, answers the instant you stop; "
              "no chime-ins or daydreams]\n")
    else:
        print("[subconscious: OFF — plain STT -> Mira -> TTS, replies when addressed]\n")

    # Everything is loaded and listening — this is the one authoritative "you may talk" signal,
    # printed only after the brain, voice, and ears are all warm and the mic is open.
    print("=" * 56)
    print("  ✓ All set — go ahead and speak or type to Mira now.")
    print("    (Type 'quit' + Enter to exit.)")
    print("=" * 56 + "\n")

    while True:
        typed = input().strip()
        if typed.lower() in ("quit", "exit"):
            scribe.finalize_if_active(notify=adapter.notify)  # save any open note session
            game_master.finalize_if_active(notify=adapter.notify)  # save an in-progress Deep IQ game
            if DRAFTING:
                subconscious.stop()                   # quiet her mind / stop the drafter first
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