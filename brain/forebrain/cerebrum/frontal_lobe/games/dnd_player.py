"""
dnd_player.py — Mira plays Dungeons & Dragons as a PARTY MEMBER (a human DMs).

Mirrors the deep_iq/game_master pattern: `is_active()` + `intercept(event, *, notify, speak)`
gated in main.handle_message, engine truth in dnd_engine (real dice, real sheet), and the LLM
only NARRATES what the engine reports — in her own dandere voice, as her character.

Commands (ALL must be addressed by name, e.g. "mira roll perception" — so the table's normal
D&D chatter, "roll for initiative everyone", never triggers her; only honored from Discord
voice/text or the local console, never Twitch chat or game audio):

  start   : "mira let's play dnd" / "mira join the party" / "mira make a character"
  create  : "mira be an elf wizard" (race optional) · "mira reroll your stats"
            · "mira name yourself Melisande" / "mira your name is X"
  rolls   : "mira roll perception" (any skill) · "mira roll a dex save" · "mira roll initiative"
            · "mira roll a d20 / 2d6+3" · "mira attack" · "mira cast magic missile"
  hp      : "mira take 5 damage" · "mira heal 3" · "mira long rest"
  sheet   : "mira show your sheet"
  end     : "mira leave the party" / "mira stop playing dnd"

Everything else flows to normal conversation — with a situation note telling her she's at the
table as her character, so ordinary "Mira, what do you do?" turns stay in-game and in-character.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Callable, Optional

from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex as _pfc
from brain.forebrain.subcortical_structures import thalamus
from brain.forebrain.subcortical_structures.basal_ganglia.action_selector import NAME, ALIASES
from . import dnd_engine as eng

SAVE_DIR = os.environ.get("MIRA_GAMES_DIR", "games")
SAVE_PATH = os.path.join(SAVE_DIR, "dnd_character.json")
LEVEL = int(os.environ.get("MIRA_DND_LEVEL", "3"))

_lock = threading.RLock()
_char: Optional[eng.Character] = None
_active = False

_NAMES = tuple(dict.fromkeys(n.lower() for n in (NAME, *ALIASES) if n and n.strip()))


def is_active() -> bool:
    with _lock:
        return _active


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------
def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9+/' ]+", " ", (t or "").lower()).strip()


def _strip_name(c: str) -> Optional[str]:
    c = re.sub(r"^(hey|ok|okay|alright|so)\s+", "", c).strip()
    for n in _NAMES:
        if c == n:
            return ""
        if c.startswith(n + " "):
            return re.sub(r"^[,!:\s]+", "", c[len(n) + 1:])
    return None


_START_RE = re.compile(
    r"\b(play|playing|join)\b.*\b(dnd|d ?and ?d|dungeons?( and dragons?)?|the (party|campaign))\b"
    r"|^(make|create|roll)( up)? (a|your) character\b|^dnd mode\b")
_END_RE = re.compile(
    r"\b(leave|stop|end|quit|done)\b.*\b(dnd|d ?and ?d|dungeons?( and dragons?)?|party|campaign)\b")
_SHEET_RE = re.compile(r"\b(show|read|whats|what's|check)\b.*\b(sheet|character|stats)\b"
                       r"|^(your )?(sheet|character sheet)$")
_BE_RE = re.compile(
    r"^(be|as|become|play as) (a |an )?"
    r"(?P<race>human|elf|halfling|dwarf|tiefling|gnome|half elf|half-elf)? ?"
    r"(?P<clazz>wizard|cleric|rogue|fighter|ranger|bard)\b")
_REROLL_RE = re.compile(r"^reroll\b")
_NAME_RE = re.compile(r"^(name yourself|your name is|call yourself|rename( yourself)? to?)\s+(?P<nm>.+)$")
_INIT_RE = re.compile(r"^roll (for )?initiative\b")
_SAVE_RE = re.compile(
    r"^(roll|make) (a |an )?"
    r"(?P<ab>str(ength)?|dex(terity)?|con(stitution)?|int(elligence)?|wis(dom)?|cha(risma)?)"
    r"[ -]?sav(e|ing)( throw)?\b")
_CHECK_RE = re.compile(r"^(roll|make) (a |an )?(?P<skill>[a-z' ]+?)( check| roll)?$")
_DICE_RE = re.compile(r"^roll (a |an )?(?P<expr>\d*d\d+([+-]\d+)?)\b")
_ATTACK_RE = re.compile(r"^(attack|roll to hit|make an attack|roll an attack)\b")
_CAST_RE = re.compile(r"^cast (?P<spell>[a-z' ]+)$")
_DMG_RE = re.compile(r"^(you )?take[sn]? (?P<n>\d+)( points of)?( damage)?\b")
_HEAL_RE = re.compile(r"^(you( a|')?re )?heal(ed)?( for)? (?P<n>\d+)\b")
_REST_RE = re.compile(r"^(take a )?long rest\b")
_ADV_RE = re.compile(r"\bwith advantage\b")
_DIS_RE = re.compile(r"\bwith disadvantage\b")
_AB_FULL = {"str": "STR", "dex": "DEX", "con": "CON", "int": "INT", "wis": "WIS", "cha": "CHA"}


def _streamer_channel(event) -> bool:
    ch = str(getattr(event, "channel", "") or "")
    return ch.startswith("discord") or ch == "local"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save():
    if _char is None:
        return
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(_char.to_json())
    except Exception as e:
        print(f"[dnd] save failed: {e}")


def _load() -> Optional[eng.Character]:
    try:
        if os.path.isfile(SAVE_PATH):
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                return eng.Character.from_json(f.read())
    except Exception as e:
        print(f"[dnd] load failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Narration — her dandere voice, reciting ONLY what the engine reports
# ---------------------------------------------------------------------------
# A TRIMMED voice, deliberately not the full persona: deep_iq learned that the full persona
# "fights the 'just narrate the engine' job" — the model starts roleplaying the scene and
# re-rolling dice in its head. Low temperature + recite-only rules keep the numbers true.
_RULES = (
    "HARD RULES — follow exactly:\n"
    "- The game engine ALREADY rolled the real dice. The RESULT lines are EXACTLY what "
    "happened — final and true.\n"
    "- Your ONLY job is to announce that result to the table in your quiet voice: state the "
    "final total (and natural 20 / natural 1 if present) EXACTLY as written.\n"
    "- NEVER roll, re-roll, change, or invent any number, die, HP value, or spell effect. Do "
    "not write out dice notation like 'd20(17)' — just say the totals in words.\n"
    "- ONE or TWO short sentences, softly, in character. No markdown, no emoji, at most one "
    "small *action* wrapped in asterisks, never end on a question.")


def _narrate(event, result_lines, speak, *, flavor_note: str = ""):
    """Speak the engine's result in character. Falls back to reading the raw line if the LLM
    is unreachable — the numbers must always reach the table."""
    log = "\n".join(l for l in result_lines if l)
    thalamus.receive(event.text, speaker=getattr(event, "speaker", None))
    chan = getattr(getattr(event, "raw", None), "id", None)
    chan = str(chan) if chan is not None else getattr(event, "channel", "local")
    who = _char.short_summary() if _char else "your character"
    system = ("You are Mira, a shy, quiet, soft-spoken girl (dandere) sitting at a D&D table, "
              f"playing your character: {who}. A human DM runs the game.\n" + _RULES
              + (f"\nThis moment: {flavor_note}" if flavor_note else ""))
    user = ("RESULT (already rolled by the engine — announce it faithfully, totals exactly "
            "as written):\n" + log + "\n\nAnnounce this to the table now.")
    try:
        stream = _pfc.client.chat.completions.create(
            model=_pfc.MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=110, temperature=0.35, stream=True, extra_body=_pfc._EXTRA,
        )
        sentences = _pfc._no_trailing_question(
            _pfc._no_bare_actions(_pfc._stream_sentences(_pfc._iter_deltas(stream))))
        speak(sentences, user_text=event.text, channel=chan, speaker=getattr(event, "speaker", None))
    except Exception as e:
        print(f"[dnd] narration error ({e}); speaking the raw result")
        speak(log, user_text=event.text, channel=chan, speaker=getattr(event, "speaker", None))


def _pick_identity() -> dict:
    """Let HER pick race/class/name (one tightly-constrained LLM call); deterministic
    fallback if the model is down or answers garbage."""
    fallback = {"race": "elf", "clazz": "wizard", "name": "Melisande"}
    try:
        resp = _pfc.client.chat.completions.create(
            model=_pfc.MODEL,
            messages=[
                {"role": "system", "content":
                    "Pick a D&D character for a shy, quiet, bookish girl to play. Reply with ONLY "
                    "a JSON object: {\"race\": one of [human, elf, halfling, dwarf, tiefling, "
                    "gnome, half-elf], \"clazz\": one of [wizard, cleric, rogue, fighter, ranger, "
                    "bard], \"name\": a pretty fantasy first name}. JSON only."},
                {"role": "user", "content": "Choose."},
            ],
            max_tokens=60, temperature=0.9, extra_body=_pfc._EXTRA,
        )
        import json as _json
        m = re.search(r"\{.*\}", (resp.choices[0].message.content or ""), re.S)
        d = _json.loads(m.group(0)) if m else {}
        race = str(d.get("race", "")).lower().replace(" ", "-")
        clazz = str(d.get("clazz", "")).lower()
        name = re.sub(r"[^A-Za-z' -]", "", str(d.get("name", "")).strip())[:24]
        return {"race": race if race in eng.RACES else fallback["race"],
                "clazz": clazz if clazz in eng.CLASSES else fallback["clazz"],
                "name": name or fallback["name"]}
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def intercept(event, *, notify: Callable[[str], None], speak: Callable) -> bool:
    """Top of handle_message. Consumes only NAME-ADDRESSED D&D commands; everything else
    returns False so normal conversation (and her in-character replies) proceed."""
    global _char, _active
    if not _streamer_channel(event):
        return False
    rest = _strip_name(_norm(getattr(event, "text", "") or ""))
    if rest is None or not rest:
        return False

    adv = 1 if _ADV_RE.search(rest) else (-1 if _DIS_RE.search(rest) else 0)
    rest_base = _ADV_RE.sub("", _DIS_RE.sub("", rest)).strip()

    with _lock:
        active = _active

    # ---- start / resume ----
    if not active:
        if _START_RE.search(rest_base):
            with _lock:
                _active = True
                _char = _load()
                if _char is not None:
                    notify(f"[D&D] {NAME} rejoins the party as her saved character.]")
                    notify(_char.sheet_panel())
                    _narrate(event, [f"You sit down at the table again as {_char.short_summary()}."],
                             speak, flavor_note="you're happy to quietly rejoin the game")
                    return True
                ident = _pick_identity()
                _char, report = eng.create_character(
                    clazz=ident["clazz"], race=ident["race"], name=ident["name"], level=LEVEL)
                _save()
            notify(f"[D&D] {NAME} joins the party. The engine rolled her character:]")
            notify(_char.sheet_panel())
            _narrate(event, [f"Character created: {_char.short_summary()}.", report],
                     speak, flavor_note=("you just rolled up this character and are shyly "
                                         "introducing her to the table — say her name"))
            return True
        return False

    # ---- active: end ----
    if _END_RE.search(rest_base):
        with _lock:
            _active = False
            _save()
        notify(f"[D&D] {NAME} steps away from the table (character saved).]")
        _narrate(event, ["You are stepping away from the game; your character is saved."],
                 speak, flavor_note="a quiet goodbye to the table")
        return True

    # ---- active: creation adjustments ----
    m = _BE_RE.match(rest_base)
    if m:
        clazz = m.group("clazz")
        race = (m.group("race") or "").replace(" ", "-") or None
        with _lock:
            keep_name = _char.name if _char else "Melisande"
            _char, report = eng.create_character(
                clazz=clazz, race=race or "elf", name=keep_name, level=LEVEL)
            _save()
        notify(_char.sheet_panel())
        _narrate(event, [f"Rebuilt as {_char.short_summary()}.", report], speak)
        return True

    if _REROLL_RE.match(rest_base) and _char is not None:
        with _lock:
            _char, report = eng.create_character(
                clazz=_char.clazz, race=_char.race, name=_char.name, level=LEVEL)
            _save()
        notify(_char.sheet_panel())
        _narrate(event, [f"Stats rerolled: now {_char.short_summary()}.", report], speak)
        return True

    m = _NAME_RE.match(rest_base)
    if m and _char is not None:
        nm = m.group("nm").strip().strip(".!\"'").title()[:30]
        with _lock:
            _char.name = nm
            _save()
        notify(f"[D&D] Her character is now named {nm}.]")
        _narrate(event, [f"Your character's name is now {nm}."], speak,
                 flavor_note="you quietly try out the new name")
        return True

    if _char is None:
        return False

    # ---- active: dice ----
    result = None
    if _INIT_RE.match(rest_base):
        result = _char.initiative()
    elif _SAVE_RE.match(rest_base):
        result = _char.save(_AB_FULL[_SAVE_RE.match(rest_base).group("ab")[:3]], adv)
    elif _DICE_RE.match(rest_base):
        total, detail = eng.roll_expr(_DICE_RE.match(rest_base).group("expr"))
        result = _char._log(f"Roll: {detail}")
    elif _ATTACK_RE.match(rest_base):
        result = _char.attack(adv)
    elif _CAST_RE.match(rest_base):
        result = _char.cast(_CAST_RE.match(rest_base).group("spell"), adv)
        _save()
    elif _DMG_RE.match(rest_base):
        result = _char.take_damage(int(_DMG_RE.match(rest_base).group("n")))
        _save()
    elif _HEAL_RE.match(rest_base):
        result = _char.heal(int(_HEAL_RE.match(rest_base).group("n")))
        _save()
    elif _REST_RE.match(rest_base):
        result = _char.long_rest()
        _save()
    elif _SHEET_RE.search(rest_base):
        notify(_char.sheet_panel())
        _narrate(event, [f"Your sheet was just shown to the table: {_char.short_summary()}."],
                 speak, flavor_note="a small shy comment about your character")
        return True
    else:
        m = _CHECK_RE.match(rest_base)
        if m:
            skill = m.group("skill").strip()
            if skill in eng.SKILLS:
                result = _char.check(skill, adv)
            elif skill[:3] in _AB_FULL and skill in (
                    "str", "dex", "con", "int", "wis", "cha", "strength", "dexterity",
                    "constitution", "intelligence", "wisdom", "charisma"):
                result = _char.ability_check(_AB_FULL[skill[:3]], adv)

    if result is None:
        return False       # not a D&D command — normal conversation handles it

    notify(f"[dice] {result}")
    _narrate(event, [result], speak)
    return True


# ---------------------------------------------------------------------------
# Situation note — keeps her ordinary replies in-game while the mode is on
# ---------------------------------------------------------------------------
def situation_note() -> str:
    with _lock:
        if not _active or _char is None:
            return ""
        recent = " | ".join(_char.log[-3:]) if _char.log else "none yet"
    return (f"You are at the table playing Dungeons & Dragons as your character: "
            f"{_char.short_summary()}. A human DM runs the game — react and roleplay as your "
            f"character in your own quiet way, but NEVER invent dice results, damage, or rules "
            f"outcomes: the game engine rolls for you when someone says '{NAME.lower()} roll "
            f"...'. Your recent engine results: {recent}.")


def finalize_if_active(*, notify: Callable[[str], None]) -> None:
    with _lock:
        if _active and _char is not None:
            _save()
            notify(f"[D&D] {NAME}'s character was saved.]")
