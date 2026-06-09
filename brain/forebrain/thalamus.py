from collections import deque

MAX_TURNS = 100 # how many recenet messages Mira remembers

working_memory = deque(maxlen = MAX_TURNS) # Mira's working memory, stores recent messages for context

def receive(user_message: str) -> list[dict]:
    """Route incoming message into working memory, return current context"""
    working_memory.append({"role": "user", "content": user_message})
    return list(working_memory)

def remember_reply(reply: str) -> None:
    working_memory.append({"role": "assistant", "content": reply})