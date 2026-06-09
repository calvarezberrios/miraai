# Emotional state engine. Tags incoming messages and tracks current mood.

mood = "neutral" # Mira's current mood, can be "happy", "sad", "angry", "flirty", etc.

# crude keyword triggers - expand these freely
TRIGGERS = {
    "excited": ["sub", "subscribed", "donated", "raid", "gifted", "let's go", "poggers"],
    "happy":   ["love", "lol", "haha", "cute", "thank", "great", "nice", "amazing"],
    "annoyed": ["boring", "shut up", "stupid", "hate", "stop", "lame", "trash"],
}

# how each mood tells Mira to behave
FLAVOR = {
    "neutral": "You feel calm and balanced.",
    "happy":   "You feel warm and cheerful. Let a little delight show.",
    "excited": "You feel hyped and energetic. Be bubbly and high-energy.",
    "annoyed": "You feel mildly irritated. Be a bit snippy and sarcastic, but not cruel.",
}

def feel(user_message: str) -> None:
    """Update mood based on incoming message, return the new mood"""
    global mood
    text = user_message.lower()
    for candidate, words in TRIGGERS.items():
        if any(w in text for w in words):
            mood = candidate
            break
        else:
            mood = "neutral" # nothing trifggered -> settle back to neutral
    return mood

def color() -> str:
    """Return the mood-flavor line to inject into the persona."""
    return FLAVOR[mood]