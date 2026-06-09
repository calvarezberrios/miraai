from brain.forebrain.prefrontal_cortex import think
from brain.forebrain.thalamus import receive, remember_reply
import brain.forebrain.amygdala as amygdala

print("Chat with Mira (type 'quit' to exit)\n\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break
    amygdala.feel(user_input)                    # amygdala updates mood
    context = receive(user_input)       # thalamus -> working memory
    reply = think(context, amygdala.color())     # prefrontal cortex, mood-colored
    remember_reply(reply)
    print(f"Mira ({amygdala.mood}): {reply}\n")