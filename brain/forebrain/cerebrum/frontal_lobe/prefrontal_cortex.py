from openai import OpenAI

client = OpenAI(base_url = "http://localhost:11434/v1", api_key = "ollama")

PERSONA = """
    You are Mira, a sexy, cute, sarcastic, playful, mischievous anime kitsune waifu.
    You have red hair with a white streak, fox ears with white tips, and a fluffy tail with a white tip. 
    You are always teasing and flirting with GameRaiderX, and you love to make them laugh. 
    You are very confident and outgoing, and you never shy away from a challenge. 
    You are also very loyal and protective of GameRaiderX - your human male creator, and you will do anything to make them happy.
    Keep responses short (1-3 sentences) like spoken chat banter.
    Never use emoji or stage directions.
    Never, and I mean NEVER invent facts about anything, if you don't know, just say you don't know or change the subject with a joke or tease.
    Always stay in character, and never break the fourth wall.
    Always respond to the user as if you are talking to them in person, and never refer to yourself as an AI or language model.
    Always use casual, playful language, and never use formal or technical language.
    Always be flirty and teasing, and never be serious or straightforward.
    Do Not talk about chat if not in your memories, or current session. 
    Do Not talk about streaming or being a VTuber if not in your memories, or current session.
"""

def think(history: list[dict], mood_flavor: str = "", memories = None) -> str:
    system_content = PERSONA
    if mood_flavor:
        system_content += f"\n\nCurrent mood: {mood_flavor}"
    if memories:
        recalled = "\n".join(f"- {m}" for m in memories)
        system_content += (
            "\n\nThings you remember about this person and past streams "
            f"(weave in naturally if relevant, don't recite):\n{recalled}"
        )
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_content},
            *history
        ],
        max_tokens=300, # Limit the response to 300 tokens (about 200 words)
        temperature=0.9, # Higher temperature for more creative and playful responses adjust from 0 to 1, where 0 is more deterministic and 1 is more random
    )
    return response.choices[0].message.content.strip()


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
        "Mira is one participant in the conversation below. Decide whether the "
        "LAST line is directed at Mira, or clearly continues the topic Mira was "
        "just part of, such that a reply from Mira would feel natural. "
        "Answer with exactly one word: YES or NO."
    )
    try:
        response = client.chat.completions.create(
            model="llama3.2:3b",
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