import time

from brain.forebrain.cerebrum.frontal_lobe.prefrontal_cortex import think
from brain.forebrain.subcortical_structures.thalamus import receive, remember_reply
from brain.forebrain.subcortical_structures.limbic_system.amygdala import feel, color
import brain.forebrain.subcortical_structures.limbic_system.amygdala as amygdala
from brain.forebrain.subcortical_structures.limbic_system.hippocampus import (
    recall, observe, consolidate, summarize_session, last_session,
)

CONSOLIDATE_EVERY_MESSAGES = 95       # consolidate after this many messages...
CONSOLIDATE_EVERY_SECONDS = 30 * 60   # ...or this long, whichever comes first

print("Chat with Mira (type 'quit' to exit)\n\n")

message_count = 0
last_consolidation = time.time()

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

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        run_consolidation()                       # flush atomic facts
        try:
            if summarize_session():               # write the session recap
                print("[Mira saved a recap of this session]\n")
        except Exception as e:
            print(f"[session summary skipped: {e}]\n")
        break

    feel(user_input)                              # amygdala updates mood
    context = receive(user_input)                 # thalamus -> working memory
    memories = recall(user_input)                 # hippocampus -> relevant long-term memories
    if previous_session:                          # keep last-session context handy
        memories = [f"From our last session: {previous_session}"] + memories
    reply = think(context, color(), memories)     # prefrontal cortex, mood + memory aware
    remember_reply(reply)
    observe(user_input, reply)                    # feed consolidation buffer + session log
    print(f"Mira ({amygdala.mood}): {reply}\n")

    message_count += 2
    if (message_count >= CONSOLIDATE_EVERY_MESSAGES
            or time.time() - last_consolidation >= CONSOLIDATE_EVERY_SECONDS):
        run_consolidation()
        message_count = 0
        last_consolidation = time.time()