import threading
import time

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think
from brain.forebrain.subcortical_structures.thalamus import receive, remember_reply
from brain.forebrain.subcortical_structures.limbic_system.amygdala import feel, color
import brain.forebrain.subcortical_structures.limbic_system.amygdala as amygdala
from brain.forebrain.subcortical_structures.limbic_system.hippocampus import (
    recall, observe, consolidate, summarize_session, last_session,
)
from brain.forebrain.cerebrum.frontal_lobe import brocas_area
from brain.forebrain.cerebrum.temporal_lobe import wernickes_area as ears

CONSOLIDATE_EVERY_MESSAGES = 95       # consolidate after this many messages...
CONSOLIDATE_EVERY_SECONDS = 30 * 60   # ...or this long, whichever comes first

INTERRUPT_NOTE = (
    " The viewer has been talking for a long time without pausing. "
    "Playfully interrupt them mid-thought with a short reaction to what "
    "they've said so far — they are still mid-ramble."
)

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


def handle_turn(user_input, interrupting=False):
    global message_count, last_consolidation
    with turn_lock:
        print(f"\nYou: {user_input}")

        feel(user_input)                              # amygdala updates mood
        context = receive(user_input)                 # thalamus -> working memory
        memories = recall(user_input)                 # hippocampus -> long-term recall
        if previous_session:
            memories = [f"From our last session: {previous_session}"] + memories

        flavor = color() + (INTERRUPT_NOTE if interrupting else "")
        reply = think(context, flavor, memories)
        remember_reply(reply)
        observe(user_input, reply)
        print(f"Mira ({amygdala.mood}): {reply}\n")

        ears.pause()                                  # don't let her hear herself
        brocas_area.say(reply)
        brocas_area.wait_until_done()
        if interrupting:
            ears.flush()                              # drop the ramble she just answered
        ears.resume()

        message_count += 2
        if (message_count >= CONSOLIDATE_EVERY_MESSAGES
                or time.time() - last_consolidation >= CONSOLIDATE_EVERY_SECONDS):
            run_consolidation()
            message_count = 0
            last_consolidation = time.time()


ears.start(
    on_partial=lambda t: print(f"\r[mic] {t}{' ' * 8}", end="", flush=True),
    on_final=handle_turn,
    on_interrupt=lambda t: handle_turn(t, interrupting=True),
)

brocas_area.warmup()    # heat up TTS so her first real reply is fast

while True:
    typed = input().strip()
    if typed.lower() in ("quit", "exit"):
        ears.stop()
        run_consolidation()                           # flush atomic facts
        try:
            if summarize_session():                   # write the session recap
                print("[Mira saved a recap of this session]\n")
        except Exception as e:
            print(f"[session summary skipped: {e}]\n")
        brocas_area.wait_until_done()                 # let her finish her last line
        break
    if typed:
        handle_turn(typed)                            # typed chat still works