import argparse
import sys
import threading
import time

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think, judge_relevance
from brain.forebrain.subcortical_structures.thalamus import receive, remember_reply
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


def run_consolidation():
    try:
        stored = consolidate()
        if stored:
            print(f"[Mira tucked away {len(stored)} memory(ies)]\n")
    except Exception as e:
        print(f"[consolidation skipped: {e}]\n")


def handle_message(event, interrupting=False):
    global message_count, last_consolidation
    with turn_lock:
        # She follows the whole room: every message updates short-term context,
        # whether or not she ends up replying to it.
        context = receive(event.text)

        chan_key = getattr(event.raw, "id", None)
        chan_key = str(chan_key) if chan_key is not None else event.channel

        # Basal ganglia decides whether this one is hers to answer.
        if not interrupting:
            decision = should_respond(
                event.text,
                mentioned=getattr(event, "mentioned", False),
                reply_to_her=getattr(event, "reply_to_her", False),
                channel=chan_key,
            )
            respond = decision.respond
            # un-addressed but conversation is live -> judge if it's a continuation
            if not respond and decision.reason == "consider":
                respond = judge_relevance(context)
            if not respond:
                if LOG_HEARD:
                    print(f"[heard] {event.speaker}: {event.text}")
                return

        print(f"\n{event.speaker}: {event.text}")

        feel(event.text)                              # amygdala updates mood
        memories = recall(event.text)                 # hippocampus -> long-term recall
        if previous_session:
            memories = [f"From our last session: {previous_session}"] + memories

        flavor = color() + (INTERRUPT_NOTE if interrupting else "")
        reply = think(context, flavor, memories)
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
        print(f"\r[mic] {ev.text}{' ' * 8}", end="", flush=True)
    elif ev.kind == INTERRUPT:
        handle_message(ev, interrupting=True)
    else:  # FINAL
        handle_message(ev)


try:
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