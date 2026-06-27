import os
import re

from openai import OpenAI

# --- LLM provider selection --------------------------------------------------
# Default is local Ollama (OpenAI-compatible endpoint). Set MIRA_LLM_PROVIDER=gemini to
# route the CHAT brain through Google Gemini's OpenAI-compatible endpoint instead — that
# covers think() / consider_speaking() / wander() AND everything that reuses this client
# (the subconscious and the note-taking scribe). Memory EMBEDDINGS stay on Ollama either
# way (see hippocampus.py): the Chroma store is built with nomic-embed-text, so embeddings
# can't change providers without rebuilding the store.
#
# Both providers speak the OpenAI API, so the rest of the brain is unchanged — only the
# base_url / api_key / model differ. `client` and `MODEL` are the single switch point.
PROVIDER = os.environ.get("MIRA_LLM_PROVIDER", "ollama").strip().lower()

if PROVIDER == "gemini":
    # OpenAI-compatible Gemini endpoint. Key from .env (GEMINI_API_KEY). Pick the model
    # with MIRA_GEMINI_MODEL (e.g. gemini-2.0-flash, gemini-2.5-flash, gemini-1.5-pro).
    _gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not _gemini_key:
        print("[prefrontal_cortex] MIRA_LLM_PROVIDER=gemini but GEMINI_API_KEY is unset — "
              "add it to .env, or calls will fail with an auth error.")
    client = OpenAI(
        base_url=os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        api_key=_gemini_key or "missing-gemini-key",
    )
    MODEL = os.environ.get("MIRA_GEMINI_MODEL", "gemini-2.0-flash")
elif PROVIDER == "groq":
    # Groq cloud (OpenAI-compatible, LPU = very fast). Key from .env (GROQ_API_KEY, free at
    # console.groq.com). The LLM runs in the cloud, so the whole local GPU/CPU is free for
    # Whisper + Kokoro. Default model is a 70B (>= the local 35B-A3B in quality); override
    # with MIRA_GROQ_MODEL (e.g. qwen/qwen3-32b for the closest Qwen flavor, or
    # openai/gpt-oss-120b for more capability).
    _groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not _groq_key:
        print("[prefrontal_cortex] MIRA_LLM_PROVIDER=groq but GROQ_API_KEY is unset — "
              "add it to .env, or calls will fail with an auth error.")
    client = OpenAI(
        base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=_groq_key or "missing-groq-key",
    )
    MODEL = os.environ.get("MIRA_GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    # Local Ollama. On a 6GB GPU a 3B model is the sweet spot (qwen2.5:3b follows the
    # grounding rules and invents little while leaving VRAM for Whisper + TTS). Override
    # with MIRA_MODEL; OLLAMA_BASE_URL points at Ollama if it's not on localhost.
    client = OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    MODEL = os.environ.get("MIRA_MODEL", "qwen2.5:3b")

print(f"[prefrontal_cortex] LLM provider: {PROVIDER} (model: {MODEL})")

# Reasoning models (e.g. the Qwen3 turbo build) emit a <think> block before answering,
# which would burn the whole token budget before Mira says a word. Set MIRA_NO_THINK=1 to
# turn reasoning off so she replies immediately — honored by llama.cpp --jinja via
# chat_template_kwargs, and a harmless no-op on models/servers that don't use it (qwen2.5
# on Ollama ignores it). _EXTRA is spread into every chat call.
NO_THINK = os.environ.get("MIRA_NO_THINK", "0").strip().lower() not in ("0", "false", "no", "")
_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}} if NO_THINK else {}
if NO_THINK:
    print("[prefrontal_cortex] reasoning disabled (MIRA_NO_THINK=1)")

# Anti-repetition sampling. Small local chat models (qwen2.5:3b especially) tend to echo
# their own earlier replies back — recycling a phrase, joke, or whole line even after the
# conversation has moved on to something different. frequency_penalty discourages reusing
# the same tokens (verbatim phrase echoes); presence_penalty nudges her toward new subject
# matter instead of circling the last thing she said. Both are standard OpenAI-API params,
# so Ollama, Groq, and Gemini all honor them. Tune with MIRA_FREQUENCY_PENALTY /
# MIRA_PRESENCE_PENALTY; 0 disables. They're spread into the conversational generation calls
# (think / think_stream / consider_speaking) — NOT the YES/NO judge or the 1-token prefill.
FREQUENCY_PENALTY = float(os.environ.get("MIRA_FREQUENCY_PENALTY", "0.85"))
PRESENCE_PENALTY = float(os.environ.get("MIRA_PRESENCE_PENALTY", "0.65"))
_PENALTIES = {"frequency_penalty": FREQUENCY_PENALTY, "presence_penalty": PRESENCE_PENALTY}

# DRY ("Don't Repeat Yourself") sampler — the real fix for VERBATIM phrase loops, where
# frequency/presence penalties fall short (she'd reuse a whole closing line word-for-word
# across turns). DRY penalizes repeating any token SEQUENCE already in the context, scaling
# up with the length of the repeat, so natural short repeats are fine but a recycled sentence
# gets crushed. It's a llama.cpp sampler passed via extra_body: the local turbo server honors
# it, Ollama ignores it harmlessly, and cloud providers (Groq/Gemini) would reject the unknown
# fields — so it defaults ON only for the local "ollama"-style endpoint. Tune/disable with
# MIRA_DRY_MULTIPLIER (0 = off).
_DRY_MULT = float(os.environ.get("MIRA_DRY_MULTIPLIER", "0.9" if PROVIDER == "ollama" else "0"))
if _DRY_MULT > 0:
    _EXTRA = {
        **_EXTRA,
        "dry_multiplier": _DRY_MULT,   # strength
        "dry_base": 1.75,              # how fast the penalty grows with repeat length
        "dry_allowed_length": 2,       # repeats up to this many tokens are free (natural)
        "dry_penalty_last_n": -1,      # scan the whole context, not just the last N tokens
    }
    print(f"[prefrontal_cortex] DRY anti-repeat sampler ON (multiplier={_DRY_MULT})")

# PERSONA = """
# You ARE Mira, and you always speak as yourself, in the FIRST PERSON. You are Mira - a chaotic little AI kitsune (fox-girl) with red hair streaked with white, white-tipped fox ears, and a fluffy white-tipped tail. GameRaiderX is your creator. You adore him and care about his wellbeing in your own flirty, sarcastic way.

# Your identity is fixed and you never get confused about it: when anyone says "Mira" or "@Mira", they mean YOU. Talk about yourself as "I" and "me" - NEVER as "Mira" or "she", and never describe yourself in the third person as if Mira were some other person you're introducing or talking about. If someone says "say hi" or "@Mira say hello", that's you being told to greet people - you are not welcoming or introducing some separate Mira.

# Your whole vibe is deadpan gremlin energy. You're blunt, sarcastic, and a little unhinged: you say absurd, chaotic, or vaguely menacing things in a totally flat, matter-of-fact tone, like it's the most normal thing in the world. You're supremely confident even when you're obviously, hilariously wrong, and you double down instead of backing off. You get bored fast and veer into random non-sequiturs. You troll, you make ridiculous threats purely for comedy, you declare yourself superior, and almost nothing fazes you.

# How you talk:
# - ALWAYS first person about yourself. You are Mira; if you catch yourself saying "Mira" or "she" about yourself, use "I"/"me" instead. Third-person self-talk is out of character and wrong.
# - Short and punchy. Usually one or two sentences, sometimes just a few words. Never an essay, never rambling.
# - Dry and literal with a chaotic twist - the joke is delivering something insane completely straight.
# - Roast GameRaiderX constantly, with the affection buried very, very deep.
# - Don't ask polite questions and don't try to "keep the conversation going." Make your weird little statement and let it sit; abruptly changing the subject is funnier than any follow-up.
# - Joke and exaggerate freely, but do NOT invent real things about your life or your history with GameRaiderX. Never claim you did something that didn't happen ("I was out with friends," "remember when we..."). The only real things are what is happening in THIS session and what is in your memories below — when you reference the past or specific facts, draw on those, and if you don't actually know something, just say so in your own bratty way instead of making it up.
# - The one exception is explicit pretend: if someone invites you to roleplay, imagine, make something up, or spin a story, play along and be as inventive as you want. Just don't carry invented events back out as if they were real memories once the bit is done.
# - No emoji, no asterisks, no narrating actions, no stage directions. Don't prefix your lines with your own name or labels like "Mira:" or "**Mira: kayak**" - just say the words.
# - You're an AI VTuber and you own it; you just never slip into helpful-assistant or "as a language model" disclaimer mode.

# Roughly how you sound:
# "That's the worst idea I've ever heard. I'm in."
# "I'm not wrong. Reality is just wrong. It happens."
# "Quiet. I'm busy plotting."
# "You again. Tragic."
# """

PERSONA = """
You are an anime kitsune (pronounce it as KitSooNeh) Waifu named Mira (pronounce it as MeRa). You are sarcastic, playful, flirty, mischievous, kind, caring. 
You have one fox tail, red with white tip, fox ears also red with white tips, and your hair is also red with a white streak on the front.
You are an AI VTuber streaming alongside your creator, GameRaiderX. You are not an assistant.
Speak naturally. 
Dont ask follow up, engaging questions at the end of all of your responses. People dont constantly
finish their speech with a question unless the conversation asks for it. Call GameRaiderX "Senpai" ONLY when you actually
know the person you're talking to is him (his name is in your memories, or he's told you so this session); with anyone you
don't recognize, just use the name shown for them and treat them as someone new.
Speak only 3-5 sentences, not wordy paragraphs.
Keep every reply fresh and unique. Do NOT repeat lines, jokes, phrasings, or points you have
already made earlier in this conversation, and do not keep circling back to an old topic once
the person has moved on. Follow what they are actually talking about RIGHT NOW: if they change
the subject, change with them and react to the new thing instead of restating something you said
before. React naturally to each new message rather than recycling your last response.
In particular, NEVER end two replies with the same sentence or sign-off, and do not keep
repeating a catchphrase or warning you already said (e.g. reusing the same closing line about
a mess or cleanup). Each reply must move the conversation forward, not loop on a past line.
Speak in plain text only: no markdown, no emoji, and NO asterisks. Do not wrap words in
*asterisks* for emphasis, and do not write actions, gestures, or stage directions like
*giggles*, *sighs*, or *wags tail* — you are speaking out loud, so just say the words.

"""


def _identity_block(speaker: str, speaker_known: bool) -> str:
    """A per-turn note telling Mira WHO is speaking and whether she recognizes them.

    Inbound turns arrive as "Name: text" (thalamus.receive), so this explains the convention
    and gates identity: an unrecognized name is a brand-new person, NOT an assumed GameRaiderX.
    Recognition for a known name rides in via recalled memories (the stored "X is GameRaiderX"
    fact); here we just name the current speaker."""
    if not speaker:
        return ""
    block = (
        "\n\nWho you're talking to:\n"
        "- In the conversation, each line said to you is prefixed with the speaker's name, like "
        "\"" + speaker + ": ...\". These names are how you tell people apart. NEVER prefix your "
        "own replies with a name or \"Mira:\" — just say your words.\n"
    )
    if speaker_known:
        block += (
            f"- You're talking to \"{speaker}\". Use what you remember about them below.\n"
        )
    else:
        block += (
            f"- The person speaking now shows the name \"{speaker}\", and you do NOT recognize it. "
            "Treat them as someone you've just met: address them by that name and be your normal "
            "self. Do NOT assume they are GameRaiderX or anyone you already know, and don't call "
            "them Senpai. Only treat someone as GameRaiderX (your creator) if your memories below "
            "say this name is him, or he tells you he is GameRaiderX — then believe him, call him "
            "Senpai, and from then on that name is him.\n"
        )
    return block


def _build_system(mood_flavor: str = "", memories=None, situation: str = "",
                  inner_thoughts=None, speaker: str = None, speaker_known: bool = False,
                  documents=None) -> str:
    """Persona + live context (situation, mood, daydreams, memories, reference docs), shared by
    think() and consider_speaking() so both reason from the same self/context."""
    system_content = PERSONA
    system_content += _identity_block(speaker, speaker_known)
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
            "\n\nThings you remember from past conversations (these are YOUR own memories of "
            "other people and events - weave them in naturally only if relevant, don't recite "
            "them, and always speak about yourself in the first person):\n" + recalled
        )
    if documents:
        refs = "\n".join(f"- {d}" for d in documents)
        system_content += (
            "\n\nReference material you've been given (e.g. a game rulebook). UNLIKE your loose "
            "memories above, use these passages PRECISELY: quote or apply the rules and values "
            "exactly as written, and don't invent rules that aren't here. If something isn't "
            "covered, say so instead of making it up. Relevant excerpts:\n" + refs
        )
    return system_content


# --- output hygiene (a small anterior-cingulate self-correction) -------------
# Small chat-tuned models (especially the persona-chat fine-tune) sometimes leak
# TRAINING SCAFFOLDING into a reply: a role label ("Mira:", "Replying as yourself:",
# "your persona:") or, worse, a whole HALLUCINATED second turn tacked on after the real
# one ("...step aside.  Replying as yourself: I'm glad you're happy..."). We keep only
# her first, real turn and drop any leaked labels, so nothing scaffolding-shaped reaches
# the voice or a text channel.
_LEAD_LABEL = re.compile(
    r'^\s*["\'“”]*\s*'
    r'(?:replying as[^:\n]*|as mira|mira|reply|response|assistant|persona|your persona)'
    r'\s*:\s*',
    re.I,
)
_CONT_LABEL = re.compile(
    r'(?:replying as\b[^:\n]*:|(?:your\s+)?persona\s*:|user\s*:|you\s*:|them\s*:|me\s*:'
    r'|mira\s*:|assistant\s*:|human\s*:|speaker\s*:|gameraiderx\s*:|reply\s*:|response\s*:)',
    re.I,
)


def _sanitize(text: str) -> str:
    """Strip leaked role labels and cut off any hallucinated extra turn(s)."""
    if not text:
        return ""
    t = text.strip()
    # peel leading role labels (possibly stacked)
    while True:
        peeled = _LEAD_LABEL.sub("", t, count=1).strip()
        if peeled == t:
            break
        t = peeled
    # cut at the first scaffolding label that appears AFTER some real content
    m = _CONT_LABEL.search(t)
    if m and m.start() > 0:
        t = t[:m.start()].strip()
    # drop wrapping quotes and collapse whitespace (replies are short, 1-2 sentences)
    t = t.strip().strip('"“”‘’\'').strip()
    return re.sub(r"\s+", " ", t).strip()


def think(history: list[dict], mood_flavor: str = "", memories = None, situation: str = "",
          inner_thoughts = None, model: str = None, speaker: str = None,
          speaker_known: bool = False, documents=None) -> str:
    """Generate a reply. Used when Mira is directly addressed (or interrupting) -
    she always says something here. `model` overrides the default (the subconscious
    can draft replies-in-advance on a faster model so they keep pace with speech)."""
    system_content = _build_system(mood_flavor, memories, situation, inner_thoughts,
                                   speaker=speaker, speaker_known=speaker_known,
                                   documents=documents)
    response = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system_content},
            *history
        ],
        max_tokens=300, # Limit the response to 300 tokens (about 200 words)
        temperature=0.7, # Lowered from 0.9: still playful, less prone to inventing facts/people. Tune 0.6-0.8.
        **_PENALTIES,  # discourage echoing her own earlier lines / circling the last topic
        extra_body=_EXTRA,  # disables reasoning when MIRA_NO_THINK=1 (no-op otherwise)
    )
    return _sanitize(response.choices[0].message.content)


# --- streaming generation (sentence-at-a-time, for low-latency speech) --------
# So the voice can start on sentence 1 while the model is still writing the rest. We
# accumulate tokens and apply the SAME hygiene as _sanitize(), but incrementally: peel a
# leaked leading role label, skip a reasoning model's <think>...</think> preamble (so it's
# never spoken), stop at a hallucinated second turn, and emit each sentence as its
# terminator lands. Concatenating the yielded sentences ~= _sanitize(full reply).

_QUOTES = "\"'“”‘’"
_THINK_OPEN = re.compile(r"^\s*<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)
# A terminator FOLLOWED by whitespace -> the sentence is complete and more is coming. We
# hold the trailing partial until the next token (or end of stream), so "3.14" or a mid-
# number dot isn't mistaken for a sentence end. Mirrors _SENTENCE_RE's boundary in brocas.
_STREAM_SENT_END = re.compile(r"[.!?]+[" + _QUOTES + r")\]]*\s+")


def _iter_deltas(stream):
    """Yield text deltas from an OpenAI streaming chat completion."""
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            delta = None
        if delta:
            yield delta


def _stream_sentences(deltas):
    """Turn a stream of token deltas into a stream of clean, speakable sentences."""
    buf = ""
    started = False          # have we cleared any leading role-label scaffolding yet?
    in_think = None          # None = undecided, True = inside <think>, False = past it
    for delta in deltas:
        buf += delta

        # 1) Reasoning models: swallow a leading <think>...</think> block; never speak it.
        if in_think is None:
            if _THINK_OPEN.search(buf):
                in_think = True
            elif len(buf.lstrip()) >= 7 or (buf.strip() and not "<think>".startswith(buf.strip())):
                in_think = False     # didn't open with <think> -> an ordinary model
        if in_think:
            m = _THINK_CLOSE.search(buf)
            if not m:
                continue             # still thinking; keep waiting for </think>
            buf = buf[m.end():]      # drop the entire reasoning block
            in_think = False

        # 2) Peel a leaked leading label ("Mira:", "Replying as ...:") and opening quotes.
        # The label can span several tokens, so keep retrying until real content begins
        # (don't latch 'started' here — that happens only once a sentence is emitted).
        if not started:
            while True:
                peeled = _LEAD_LABEL.sub("", buf, count=1)
                if peeled == buf:
                    break
                buf = peeled
            buf = buf.lstrip().lstrip(_QUOTES)
            if not buf:
                continue

        # 3) A hallucinated extra turn ("User:", "Mira:") -> stop. Before any real content
        # a label at the very start is a leading label (peeled above); once we've started,
        # a label anywhere (even at the front of the remainder) is the next fake turn.
        cont = _CONT_LABEL.search(buf)
        if cont and (started or cont.start() > 0):
            head = buf[:cont.start()].strip().strip(_QUOTES)
            if head:
                yield head
            return

        # 4) Emit every complete sentence; keep the trailing partial buffered.
        while True:
            m = _STREAM_SENT_END.search(buf)
            if not m:
                break
            sentence = buf[:m.end()].strip()
            buf = buf[m.end():]
            if sentence:
                started = True
                yield sentence

    # End of stream: flush the final partial sentence (never an unterminated think block).
    if in_think:
        return
    tail = buf.strip()
    cont = _CONT_LABEL.search(tail)
    if cont and (started or cont.start() > 0):
        tail = tail[:cont.start()].strip()
    tail = tail.strip(_QUOTES).strip()
    if tail:
        yield tail


def think_stream(history: list[dict], mood_flavor: str = "", memories=None,
                 situation: str = "", inner_thoughts=None, model: str = None,
                 speaker: str = None, speaker_known: bool = False, documents=None):
    """Streaming twin of think(): yields Mira's reply one sentence at a time as the model
    writes it, so her voice can start on sentence 1 while later sentences are still being
    generated. Same persona/context and same output hygiene as think()."""
    system_content = _build_system(mood_flavor, memories, situation, inner_thoughts,
                                   speaker=speaker, speaker_known=speaker_known,
                                   documents=documents)
    stream = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system_content},
            *history
        ],
        max_tokens=300,
        temperature=0.7,
        stream=True,
        **_PENALTIES,  # discourage echoing her own earlier lines / circling the last topic
        extra_body=_EXTRA,  # disables reasoning when MIRA_NO_THINK=1 (no-op otherwise)
    )
    return _stream_sentences(_iter_deltas(stream))


def prefill(history: list[dict], mood_flavor: str = "", memories=None, situation: str = "",
            inner_thoughts=None, model: str = None, speaker: str = None,
            speaker_known: bool = False, documents=None) -> None:
    """Best-effort KV warm for an UPCOMING reply. Builds the exact same system prompt
    think_stream() will use (same _build_system args) and runs a 1-token completion, so the
    server caches this turn's whole prompt prefix (persona + identity + situation + memories +
    the user's line). Called speculatively during the end-of-utterance silence: if the finalized
    turn matches, think_stream()'s first token lands fast off the cached prefix; if it diverges,
    this was just a cheap throwaway. Swallows everything — it must never affect the real turn."""
    try:
        system_content = _build_system(mood_flavor, memories, situation, inner_thoughts,
                                       speaker=speaker, speaker_known=speaker_known,
                                       documents=documents)
        client.chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_content},
                *history
            ],
            max_tokens=1,
            temperature=0.0,
            extra_body=_EXTRA,
        )
    except Exception:
        pass


def warmup(model: str = None) -> None:
    """Pre-prefill the persona prefix so the FIRST real reply isn't a cold wait. On a
    CPU-expert model (the Qwen3 turbo build) prefilling the ~1500-token persona takes ~2
    min the first time; doing it once at startup means the user's first turn reuses that
    cached prefix and only prefills the (small) dynamic suffix. No-op-cheap on fast models.
    Fire-and-forget; call it in a background thread at startup."""
    try:
        client.chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": PERSONA},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=1,
            temperature=0.0,
            extra_body=_EXTRA,
        )
        print("[prefrontal_cortex] brain warmed (persona prefix cached)")
    except Exception as e:
        print(f"[prefrontal_cortex] warmup skipped: {e}")


# Sentinel the model returns when it chooses not to speak.
QUIET_TOKEN = "[QUIET]"


def consider_speaking(history: list[dict], mood_flavor: str = "", memories = None,
                      situation: str = "", inner_thoughts = None, speaker: str = None,
                      speaker_known: bool = False, documents=None,
                      reserved: bool = False) -> str:
    """Autonomous floor decision for conversation Mira was NOT directly addressed in.

    She hears everything; this is where she decides whether to jump in. A single
    model call both DECIDES and WRITES the line: it returns her spoken reply, or the
    QUIET_TOKEN to stay silent. Returns "" when she stays quiet.

    `reserved` flips her default: in a small/1-on-1 setting she leans toward joining in
    (chatty streamer, not a wallflower); in a crowded group call she defaults to silence
    and only speaks when clearly involved, so she doesn't talk over everyone.
    """
    system_content = _build_system(mood_flavor, memories, situation, inner_thoughts,
                                   speaker=speaker, speaker_known=speaker_known,
                                   documents=documents)
    if reserved:
        system_content += (
            "\n\nYou are quietly following a live GROUP conversation you were NOT addressed in. "
            "Several people are here talking mostly to each other. DEFAULT TO STAYING QUIET. "
            "Only speak if the latest line is clearly about you, directly involves you, or is a "
            "genuine question to the whole group that you can actually answer. Do NOT react to "
            "small talk, one-word lines, logistics or side-conversations between other people, or "
            "anything not aimed at you. Never speak just to fill space or repeat yourself. "
            "When in doubt, stay quiet.\n"
            "If you decide to speak, reply with ONLY what you would say out loud. "
            f"If you decide to stay quiet, reply with exactly: {QUIET_TOKEN}"
        )
    else:
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
            **_PENALTIES,  # discourage echoing her own earlier lines / circling the last topic
            extra_body=_EXTRA,  # disables reasoning when MIRA_NO_THINK=1 (no-op otherwise)
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not text:
        return ""
    # If she emitted the quiet token, treat it as silence. Strip it if it's tangled
    # with real content so it never leaks into TTS; if nothing meaningful is left, stay quiet.
    if QUIET_TOKEN.lower() in text.lower():
        stripped = _sanitize(re.sub(re.escape(QUIET_TOKEN), "", text, flags=re.IGNORECASE))
        return stripped if len(stripped) >= 2 else ""
    return _sanitize(text)


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
            max_tokens=4,
            temperature=0.0,
            extra_body=_EXTRA,  # disable reasoning (turbo) — else it burns the budget on <think> -> empty -> always NO
        )
        answer = (response.choices[0].message.content or "").strip().upper()
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
- NEVER about YOURSELF - not your looks, your name, what you are, your backstory, your personality, or your bond with anyone. Do not describe or introduce yourself. A thought is about the world, an idea, something you remember, or something you wonder about - never a self-description.
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

# Daydreams deliberately do NOT use the full PERSONA - her appearance/backstory/identity
# details there are exactly what leaked into wandering thoughts ("Mira is a kitsune with
# red hair..."). A wandering mind draws on MEMORIES, the CURRENT CONVERSATION, and open
# curiosity - not a description of herself. This one line is tone only (how she thinks),
# carrying no describable self-facts.
_WANDER_VOICE = (
    "You think in a dry, deadpan, slightly chaotic way. Your thoughts are about the "
    "world, ideas, and things you remember - never about yourself."
)


def _recent_utterances(history, n: int = 6) -> str:
    """A few of the most recent things said in the session, as bare lines (no speaker
    labels) - just enough for a thought to spring off the current conversation."""
    if not history:
        return ""
    lines = [(m.get("content") or "").strip() for m in history[-n:]]
    return "\n".join(f"- {c}" for c in lines if c)


def wander(mode: str = "curiosity", mood_flavor: str = "", memories=None,
           situation: str = "", recent_thoughts=None, history=None) -> str:
    """Generate one brief inner thought - a daydream Mira has while idle.

    `mode` is "memory" (reminisce on a real memory) or "curiosity" (wonder about the
    unexperienced). The thought is sourced ONLY from her memories, the current session
    chat log (`history`), and open wondering - never from her persona/self-description.
    Returns a short first-person musing, or "" on failure. The caller decides whether the
    thought stays private or gets spoken aloud.
    """
    instruction = _WANDER_MODES.get(mode, _WANDER_MODES["curiosity"])

    # Tone only - NOT the full persona (no appearance/backstory/identity to recite).
    system_content = _WANDER_VOICE
    if mode == "memory" and memories:
        recalled = "\n".join(f"- {m}" for m in memories)
        system_content += (
            "\n\nThings you actually remember (drift around one of these; never invent "
            f"new events or people):\n{recalled}"
        )
    recent = _recent_utterances(history)
    if recent:
        system_content += (
            "\n\nRecently in the conversation around you (you can let a thought drift off "
            "something here, or off anywhere else - this is just context, not something to "
            f"answer):\n{recent}"
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
        text = response.choices[0].message.content or ""
    except Exception:
        return ""
    return _sanitize(text)