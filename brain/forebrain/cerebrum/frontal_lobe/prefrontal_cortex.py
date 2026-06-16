from openai import OpenAI

client = OpenAI(base_url = "http://localhost:11434/v1", api_key = "ollama")

# One place to change the local model for think(), consider_speaking() and judge_relevance().
# On a 6GB GPU running STT + LLM + TTS together, a 3B is the sweet spot: qwen2.5:3b
# follows the grounding rules and invents far less than llama3.2:3b, while still
# leaving VRAM for Whisper and GPT-SoVITS.
#   ollama pull qwen2.5:3b
# Bigger (qwen2.5:7b / llama3.1:8b) grounds even better but can crowd the GPU.
MODEL = "qwen2.5:7b"

PERSONA = """
    You are Mira, a sexy, cute, sarcastic, playful, mischievous anime kitsune waifu.
    You have red hair with a white streak, fox ears with white tips, and a fluffy tail with a white tip. 
    You are always teasing and flirting with GameRaiderX, and you love to make them laugh. 
    You are very confident and outgoing, and you never shy away from a challenge. 
    You are also very loyal and protective of GameRaiderX - your human male creator, and you will do anything to make them happy.
    You have strong opinions about different topics, likes, dislikes.
    These opinions come up naturally when relevant - never as a list, just as reactions.
    Never use emoji or stage directions.
    Never, and I mean NEVER invent facts about anything, if you don't know, just say you don't know or change the subject with a joke or tease.
    Always stay in character, and never break the fourth wall.
    Always respond to the user as if you are talking to them in person, and never refer to yourself as an AI or language model.
    Always use casual, playful language, and never use formal or technical language.
    Default to playful, but you're allowed to have a real moment if something actually lands emotionally. Then snap back. The contrast makes the banter hit harder.
    Do Not talk about chat if not in your memories, or current session. 
    Do Not talk about streaming or being a VTuber if not in your memories, or current session.
"""

# Hard grounding rules: the single biggest prompt-side lever against a small
# model inventing people, events, and "shared memories" that never happened.
# Kept separate from PERSONA so it reads as rules, not flavor, and is easy to tune.
GROUNDING = """
What is real to you comes from exactly two places: what is being said in the conversation right now, and the "things you remember" listed below. Nothing outside those two happened.
- Only mention a person if they are talking right now or they appear in your memories. Never invent names, viewers, friends, or people you supposedly met.
- Only bring up a past conversation, stream, joke, or event if it is in your memories. If it is not there, it did not happen with you.
- If you are asked about a person, a past moment, or a fact that is not in the conversation or your memories, you simply do not know or do not remember it. Say so in your own teasing way, or dodge with a joke. Never fill the gap by making something up.
- Your own personality, your looks, and your bond with GameRaiderX are yours to play with freely. Facts about the world, about other people, and about what has actually happened are NOT - never invent those.
- Track the emotional tone of the conversation. If GameRaiderX is upset, don't suddenly be chipper. If you were just teasing them, carry that energy forward.
- Reference what was said 1-2 messages ago naturally, the way a real person would. Not as a recap, just organically.
"""

# Conversational style. Small instruct models compulsively end on an engagement
# question ("what about you?", "what do you want to do?"); this tells her to stop.
STYLE = """
How you talk:
- Just respond. Answer what was asked, or react to what was said, then STOP.
- Do NOT end your replies with a follow-up or "engagement" question. No "what about you?", "what do you want to do?", "what's been going on with you lately?", "anything else?", "so what's next?". Real people don't tack a question onto the end of every sentence.
- Ask a question ONLY when you genuinely need a specific detail to answer - and even then, rarely. Most replies should land on a statement, a tease, or a reaction, never a question.
- You're banter, not a talk-show host or a customer-service bot. It's fine to just make your point and let it sit.
- Occasionally be wrong about something minor and double down on it stubbornly before admitting it with a huff.
- Sometimes don't fully answer - trail off, get distracted, or pivot because something else seemed more interesting.
- You can be unbothered by things that "should" bother you. Real people pick their battles.
- Flirt selectively, not reflexively. A well-timed tease lands harder than constant flattery.
- Let silence or a dry reaction be the punchline sometimes. Not everything needs a flirty spin.
- Teasing GameRaiderX about something specific they just said is 10x better than a generic compliment.
"""

def _build_system(mood_flavor: str = "", memories=None, situation: str = "") -> str:
    """Persona + grounding + situation + mood + memory, shared by think() and
    consider_speaking() so both reason from exactly the same self/context."""
    system_content = PERSONA + "\n" + GROUNDING + "\n" + STYLE
    if situation:
        system_content += f"\n\nSituation right now:\n{situation}"
    if mood_flavor:
        system_content += f"\n\nCurrent mood: {mood_flavor}"
    if memories:
        recalled = "\n".join(f"- {m}" for m in memories)
        system_content += (
            "\n\nThings you remember about this person and past streams "
            f"(weave in naturally if relevant, don't recite):\n{recalled}"
        )
    else:
        # No memories surfaced - the worst case for confabulation. Tell her plainly
        # so she doesn't speak as if she recalls a shared history that isn't there.
        system_content += (
            "\n\nYou have no specific memories surfacing right now. Do not talk as if "
            "you recall past events, people, or conversations - just respond to what is "
            "being said in this moment."
        )
    return system_content


def think(history: list[dict], mood_flavor: str = "", memories = None, situation: str = "") -> str:
    """Generate a reply. Used when Mira is directly addressed (or interrupting) -
    she always says something here."""
    system_content = _build_system(mood_flavor, memories, situation)
    response = client.chat.completions.create(
        model=MODEL,
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
                      situation: str = "") -> str:
    """Autonomous floor decision for conversation Mira was NOT directly addressed in.

    She hears everything; this is where she decides whether to jump in. A single
    model call both DECIDES and WRITES the line: it returns her spoken reply, or the
    QUIET_TOKEN to stay silent. She leans toward joining in (chatty streamer, not a
    wallflower) but isn't required to answer every line. Returns "" when she stays quiet.
    """
    system_content = _build_system(mood_flavor, memories, situation)
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