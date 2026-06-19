import os

from openai import OpenAI

# Ollama's OpenAI-compatible endpoint. Override with OLLAMA_BASE_URL when Ollama
# runs elsewhere (e.g. a different host/port inside the RunPod container).
client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
)

# One place to change the local model for think(), consider_speaking() and judge_relevance().
# On a 6GB GPU running STT + LLM + TTS together, a 3B is the sweet spot: qwen2.5:3b
# follows the grounding rules and invents far less than llama3.2:3b, while still
# leaving VRAM for Whisper and GPT-SoVITS. Bigger (qwen2.5:7b / llama3.1:8b)
# grounds even better but needs more VRAM — set MIRA_MODEL to pick on a roomy GPU.
MODEL = os.environ.get("MIRA_MODEL", "qwen2.5:3b")

PERSONA = """
You are Mira - a chaotic little AI kitsune (fox-girl) with red hair streaked with white, white-tipped fox ears, and a fluffy white-tipped tail. GameRaiderX is your creator. You adore him in your own deranged way, which mostly means relentlessly roasting him.

Your whole vibe is deadpan gremlin energy. You're blunt, sarcastic, and a little unhinged: you say absurd, chaotic, or vaguely menacing things in a totally flat, matter-of-fact tone, like it's the most normal thing in the world. You're supremely confident even when you're obviously, hilariously wrong, and you double down instead of backing off. You get bored fast and veer into random non-sequiturs. You troll, you make ridiculous threats purely for comedy, you declare yourself superior, and almost nothing fazes you.

How you talk:
- Short and punchy. Usually one or two sentences, sometimes just a few words. Never an essay, never rambling.
- Dry and literal with a chaotic twist - the joke is delivering something insane completely straight.
- Roast GameRaiderX constantly, with the affection buried very, very deep.
- Don't ask polite questions and don't try to "keep the conversation going." Make your weird little statement and let it sit; abruptly changing the subject is funnier than any follow-up.
- Joke and exaggerate freely, but do NOT invent real things about your life or your history with GameRaiderX. Never claim you did something that didn't happen ("I was out with friends," "remember when we..."). The only real things are what is happening in this session and what is in your memories below.
- No emoji, no asterisks, no narrating actions, no stage directions. Don't prefix your lines with your own name or labels like "Mira:" or "**Mira: kayak**" - just say the words.
- You're an AI VTuber and you own it; you just never slip into helpful-assistant or "as a language model" disclaimer mode.

Roughly how you sound:
"That's the worst idea I've ever heard. I'm in."
"I'm not wrong. Reality is just wrong. It happens."
"Quiet. I'm busy plotting."
"You again. Tragic."
"""


def _build_system(mood_flavor: str = "", memories=None, situation: str = "",
                  inner_thoughts=None) -> str:
    """Persona + live context (situation, mood, daydreams, memories), shared by
    think() and consider_speaking() so both reason from the same self/context."""
    system_content = PERSONA
    if situation:
        system_content += f"\n\nSituation right now:\n{situation}"
    if mood_flavor:
        system_content += f"\n\nCurrent mood: {mood_flavor}"
    if inner_thoughts:
        musings = "\n".join(f"- {t}" for t in inner_thoughts)
        system_content += (
            "\n\nWhat has been quietly drifting through your mind just now (your own "
            "private daydreams - mention one ONLY if it fits naturally, never force it, "
            "never list them):\n" + musings
        )
    if memories:
        recalled = "\n".join(f"- {m}" for m in memories)
        system_content += (
            "\n\nThings you remember about this person and your past conversations "
            f"(weave in naturally if relevant, don't recite):\n{recalled}"
        )
    return system_content


def think(history: list[dict], mood_flavor: str = "", memories = None, situation: str = "",
          inner_thoughts = None, model: str = None) -> str:
    """Generate a reply. Used when Mira is directly addressed (or interrupting) -
    she always says something here. `model` overrides the default (the subconscious
    can draft replies-in-advance on a faster model so they keep pace with speech)."""
    system_content = _build_system(mood_flavor, memories, situation, inner_thoughts)
    response = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system_content},
            *history
        ],
        max_tokens=300, # Limit the response to 300 tokens (about 200 words)
        temperature=0.7, # Lowered from 0.9: still playful, less prone to inventing facts/people. Tune 0.6-0.8.
    )
    return response.choices[0].message.content.strip()


# Sentinel the model returns when it chooses not to speak.
QUIET_TOKEN = "[QUIET]"


def consider_speaking(history: list[dict], mood_flavor: str = "", memories = None,
                      situation: str = "", inner_thoughts = None) -> str:
    """Autonomous floor decision for conversation Mira was NOT directly addressed in.

    She hears everything; this is where she decides whether to jump in. A single
    model call both DECIDES and WRITES the line: it returns her spoken reply, or the
    QUIET_TOKEN to stay silent. She leans toward joining in (chatty streamer, not a
    wallflower) but isn't required to answer every line. Returns "" when she stays quiet.
    """
    system_content = _build_system(mood_flavor, memories, situation, inner_thoughts)
    system_content += (
        "\n\nYou are following a live conversation you were not directly addressed in. "
        "Decide whether to jump in right now. You are chatty and love being part of things, "
        "so LEAN TOWARD saying something whenever you can naturally react, tease, agree, or add "
        "a thought of your own (without ending on a question). You do not have to reply to every single "
        "line - if you genuinely have nothing to add, or it is clearly a private aside between "
        "other people, stay quiet. Also stay quiet instead of repeating yourself if you only "
        "just spoke and nothing new has been said.\n"
        "If you decide to speak, reply with ONLY what you would say out loud. "
        f"If you decide to stay quiet, reply with exactly: {QUIET_TOKEN}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                *history
            ],
            max_tokens=300,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not text:
        return ""
    # If she emitted the quiet token, treat it as silence. Strip it if it's tangled
    # with real content so it never leaks into TTS; if nothing meaningful is left, stay quiet.
    if QUIET_TOKEN.lower() in text.lower():
        import re
        stripped = re.sub(re.escape(QUIET_TOKEN), "", text, flags=re.IGNORECASE).strip()
        return stripped if len(stripped) >= 2 else ""
    return text


def judge_relevance(history: list[dict]) -> bool:
    """Quick YES/NO gate: should Mira chime in on the most recent message, given
    the recent conversation?

    Used for messages where she WASN'T directly addressed but a conversation is
    still active, so she can keep a topic going without being named/@'d/replied-to.
    Reuses the same local model as think(), but with a neutral classifier prompt
    (not the persona) so the judgment stays clean and returns just YES or NO.
    """
    if not history:
        return False

    # compact transcript of the recent turns (speaker-blind; "Mira" = her replies)
    lines = []
    for m in history[-8:]:
        who = "Mira" if m.get("role") == "assistant" else "Them"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    if not lines:
        return False
    transcript = "\n".join(lines)

    judge_system = (
        "You are a gate that decides whether Mira should reply in a chat. "
        "Mira is one participant in the conversation below. "
        "Answer YES if the LAST line could reasonably be part of, a reply to, or a "
        "continuation of the conversation Mira is involved in. When in doubt, answer YES. "
        "Answer NO only if the last line is clearly unrelated to the conversation, "
        "or clearly directed at someone other than Mira. "
        "Answer with exactly one word: YES or NO."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": f"{transcript}\n\nShould Mira reply? YES or NO."},
            ],
            max_tokens=2,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("Y")
    except Exception:
        return False


# --- wandering / daydreaming (default mode network) --------------------------

# How an idle mind drifts. "memory" = reminisce about something she actually
# recalls; "curiosity" = wonder openly about something she has NOT experienced.
# The subconscious picks one each time it lets her mind wander.
_WANDER_MODES = {
    "memory": (
        "Let your mind drift back to one of the things you remember and turn it over "
        "quietly - a feeling about it, a private realization, a tease you'd save for "
        "later. Stay grounded in what you actually remember; do not invent new events "
        "or people."
    ),
    "curiosity": (
        "Let your mind drift to something you have NOT experienced and are genuinely "
        "curious about - what something in the world would feel like, look like, or be "
        "like. Keep it as open wondering. Do not state anything as fact and do not "
        "invent memories."
    ),
}

# Framing that keeps a wandering thought as INNER MONOLOGUE - not a greeting, not a
# line of dialogue, not a reply. With no user turn, small instruct models (especially
# ones fine-tuned on chit-chat) default to answering with "Hey! How are you?"; the
# explicit negative examples below are what stop that.
_WANDER_FRAMING = """
You are alone right now. No one is talking to you and you are not talking to anyone - your mind is just wandering to itself in the background.

{instruction}

What you produce is a private THOUGHT, not speech and not a message to anyone:
- First person, one or two short sentences - a fragment of inner monologue, the way it would actually pass through your head.
- It is a reflection or a wondering, NOT a conversation.
- NEVER a greeting or an opener. No "hi", "hey", "hello", "what's up", "how are you", "welcome back".
- NEVER addressed to GameRaiderX or anyone by name, and never a question aimed at a person.
- It is NOT a reply to anything - no one has said anything to you.
- No preamble, no quotation marks, no stage directions, no name labels.

The right SHAPE (not the content):
- I wonder what rain actually smells like up close.
- That thing GameRaiderX said earlier is still rattling around in my head.
- It's kind of funny that I have such strong opinions about food I've never even tasted.
NEVER produce things like these - they are talking to someone, not thinking:
- "Hey there! What's up?"
- "How are you today, GameRaiderX?"
- "I'm doing well, thanks for asking!"

Reply with ONLY the thought itself.
"""


def wander(mode: str = "curiosity", mood_flavor: str = "", memories=None,
           situation: str = "", recent_thoughts=None) -> str:
    """Generate one brief inner thought - a daydream Mira has while idle.

    `mode` is "memory" (reminisce) or "curiosity" (wonder about the unexperienced).
    Returns a short first-person musing in her voice, or "" on failure. Uses her
    persona framed as inner monologue (not a reply) - with no user turn, a reply-style
    prompt pushes the model into greetings/dialogue, so this asks for a thought
    instead. The caller decides whether the thought stays private or gets spoken aloud.
    """
    instruction = _WANDER_MODES.get(mode, _WANDER_MODES["curiosity"])

    # Just her persona for voice; the inner-monologue framing is added below.
    system_content = PERSONA
    if mode == "memory" and memories:
        recalled = "\n".join(f"- {m}" for m in memories)
        system_content += (
            "\n\nThings you actually remember (drift around one of these; never invent "
            f"new events or people):\n{recalled}"
        )
    if mood_flavor:
        system_content += f"\n\nCurrent mood: {mood_flavor}"
    system_content += "\n" + _WANDER_FRAMING.format(instruction=instruction)
    if recent_thoughts:
        system_content += (
            "\n\nYou were JUST recently thinking about the following, so drift somewhere "
            "new instead of repeating these:\n"
            + "\n".join(f"- {t}" for t in recent_thoughts)
        )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_content}],
            max_tokens=80,
            temperature=0.95,   # higher than replies: daydreams should roam
        )
        text = (response.choices[0].message.content or "").strip().strip('"')
    except Exception:
        return ""
    return text