"""
dnd_player.py — Mira plays Dungeons & Dragons as a PARTY MEMBER (a human DMs).

Engine truth (dnd_engine) owns every die and every number on the sheet; the LLM only NARRATES
what the engine reports, in her dandere voice, as her character. She NEVER invents a roll.

Entering the mode does NOT make a character. You either build one step by step or load a saved
one. All commands are name-addressed ("mira ...") from a streamer channel (voice/text/local),
so ordinary table chatter never triggers her.

  MODE
    "mira let's play dnd" / "mira join the party" / "mira dnd mode"   enter (no character yet)
    "mira leave the party" / "mira stop playing dnd"                 exit (saves)

  CREATE A CHARACTER — step by step (say the pieces in any order)
    "mira create a character"            begin building
      "mira be an elf" / "mira race tiefling"          set race
      "mira be a wizard" / "mira class rogue"          set class
      "mira name yourself Lyra"                        set name
      "mira level 3"                                   set level
      "mira roll your stats" / "mira use standard array"   choose stat method
      "mira set int 16"                                assign one score by hand
      "mira show the character" / "mira cancel"        review / abort
    "mira finish the character"          finalize (needs at least a class)

  LOAD A SAVED CHARACTER
    "mira use character Lyra"            load by name   ·   "mira list characters"

  PLAY (once she has a character) — the engine rolls, she announces the real result
    "mira roll perception" / "mira make a stealth check" / "mira perception check"
    "mira roll a dex save" · "mira roll initiative" · "mira roll 2d6+3" / "mira roll a d20"
    "mira attack" / "mira attack the goblin" / "mira roll to hit"
    "mira cast magic missile" · "mira take 5 damage" · "mira heal 3" · "mira long rest"
    "mira show your sheet"      (add "with advantage" / "with disadvantage" to any roll)

Between rolls she's a normal party member — but while the mode is on she must never STATE a die
number or outcome in ordinary chat; scrub_stream/scrub_text (used by main.py) strip any she slips.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Callable, List, Optional

from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex as _pfc
from brain.forebrain.subcortical_structures import thalamus
from brain.forebrain.subcortical_structures.basal_ganglia.action_selector import NAME, ALIASES
from . import dnd_engine as eng

CHARS_DIR = os.path.join(os.environ.get("MIRA_GAMES_DIR", "games"), "dnd_characters")
DEFAULT_LEVEL = int(os.environ.get("MIRA_DND_LEVEL", "1"))

_lock = threading.RLock()
_active = False
_char: Optional[eng.Character] = None
_draft: Optional[dict] = None       # step-by-step creation in progress (None = not creating)

_NAMES = tuple(dict.fromkeys(n.lower() for n in (NAME, *ALIASES) if n and n.strip()))


def is_active() -> bool:
    with _lock:
        return _active


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+/' ]+", " ", (t or "").lower())).strip()


def _strip_name(c: str) -> Optional[str]:
    c = re.sub(r"^(hey|ok|okay|alright|so|um|uh)\s+", "", c).strip()
    for n in _NAMES:
        if c == n:
            return ""
        if c.startswith(n + " "):
            return re.sub(r"^[,!:\s]+", "", c[len(n) + 1:])
    return None


def _streamer_channel(event) -> bool:
    ch = str(getattr(event, "channel", "") or "")
    return ch.startswith("discord") or ch == "local"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "char").lower()).strip("-") or "char"


def _adv(rest: str) -> int:
    if re.search(r"\b(with )?advantage\b", rest):
        return 1
    if re.search(r"\b(with )?disadvantage\b", rest):
        return -1
    return 0


_RACE_RE = "|".join(sorted((r.replace("-", "[- ]?") for r in eng.RACES), key=len, reverse=True))
_CLASS_RE = "|".join(eng.CLASSES)
_AB_WORD = {"str": "STR", "strength": "STR", "dex": "DEX", "dexterity": "DEX",
            "con": "CON", "constitution": "CON", "int": "INT", "intelligence": "INT",
            "wis": "WIS", "wisdom": "WIS", "cha": "CHA", "charisma": "CHA"}


# ---------------------------------------------------------------------------
# Persistence — one JSON per named character
# ---------------------------------------------------------------------------
def _save(c: eng.Character) -> None:
    try:
        os.makedirs(CHARS_DIR, exist_ok=True)
        with open(os.path.join(CHARS_DIR, _slug(c.name) + ".json"), "w", encoding="utf-8") as f:
            f.write(c.to_json())
    except Exception as e:
        print(f"[dnd] save failed: {e}")


def _list_saved() -> List[str]:
    try:
        out = []
        for fn in sorted(os.listdir(CHARS_DIR)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(CHARS_DIR, fn), encoding="utf-8") as f:
                        out.append(eng.Character.from_json(f.read()).name)
                except Exception:
                    pass
        return out
    except Exception:
        return []


def _load_by_name(name: str) -> Optional[eng.Character]:
    want = _slug(name)
    try:
        for fn in os.listdir(CHARS_DIR):
            if fn.endswith(".json") and _slug(fn[:-5]) == want:
                with open(os.path.join(CHARS_DIR, fn), encoding="utf-8") as f:
                    return eng.Character.from_json(f.read())
        # fuzzy: first character whose name contains the query
        for fn in sorted(os.listdir(CHARS_DIR)):
            if fn.endswith(".json"):
                with open(os.path.join(CHARS_DIR, fn), encoding="utf-8") as f:
                    c = eng.Character.from_json(f.read())
                if want in _slug(c.name):
                    return c
    except Exception as e:
        print(f"[dnd] load failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Narration — trimmed dandere voice, RECITE ONLY (low temp so numbers stay true)
# ---------------------------------------------------------------------------
_RULES = (
    "HARD RULES — follow exactly:\n"
    "- The game engine ALREADY rolled the real dice. The RESULT line is EXACTLY what happened — "
    "final and true.\n"
    "- Announce that result to the table in your quiet voice: SAY THE FINAL TOTAL out loud (e.g. "
    "'that's a 13 to hit, and 11 piercing if it lands'), including natural 20 / natural 1 if noted.\n"
    "- Do NOT decide whether an attack HIT or MISSED, or whether a check SUCCEEDED or FAILED — the "
    "DM decides that. Just report your number and let them call it.\n"
    "- NEVER roll, re-roll, change, or invent any number, die, HP, or spell effect. Don't write "
    "dice notation like 'd20(17)'; say totals in words.\n"
    "- ONE or TWO short sentences, softly, in character. No markdown, no emoji, at most one small "
    "*action* in asterisks, never end on a question.")


def _narrate(event, result_lines, speak, *, flavor_note: str = ""):
    log = "\n".join(l for l in result_lines if l)
    thalamus.receive(event.text, speaker=getattr(event, "speaker", None))
    chan = getattr(getattr(event, "raw", None), "id", None)
    chan = str(chan) if chan is not None else getattr(event, "channel", "local")
    who = _char.short_summary() if _char else "your character"
    system = ("You are Mira, a shy, quiet, soft-spoken girl (dandere) at a D&D table, playing "
              f"your character: {who}. A human DM runs the game.\n" + _RULES
              + (f"\nThis moment: {flavor_note}" if flavor_note else ""))
    user = ("RESULT (already rolled by the engine — announce it faithfully, totals exactly as "
            "written):\n" + log + "\n\nAnnounce this to the table now.")
    try:
        stream = _pfc.client.chat.completions.create(
            model=_pfc.MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=110, temperature=0.35, stream=True, extra_body=_pfc._EXTRA,
        )
        sentences = _pfc._no_trailing_question(
            _pfc._no_bare_actions(_pfc._stream_sentences(_pfc._iter_deltas(stream))))
        speak(sentences, user_text=event.text, channel=chan, speaker=getattr(event, "speaker", None))
    except Exception as e:
        print(f"[dnd] narration error ({e}); speaking the raw result")
        speak(log, user_text=event.text, channel=chan, speaker=getattr(event, "speaker", None))


def _ddb_import(event, url_or_id, notify, speak):
    global _active, _char, _draft
    from . import dnd_beyond as _ddb
    notify("[D&D] Importing from D&D Beyond...")
    try:
        c, imp_notes = _ddb.import_character(url_or_id)
    except Exception as e:
        notify(f"[D&D] Import failed: {e}")
        with _lock:
            _active = True
        return
    with _lock:
        _active = True
        _char = c
        _draft = None
        _save(c)
    notify("[D&D] Imported from D&D Beyond — compare this to your sheet on the site:")
    notify(c.sheet_panel())
    for n in imp_notes:
        notify("  note: " + n)
    _narrate(event, [f"You loaded your real character from D&D Beyond: {c.short_summary()}."],
             speak, flavor_note="quietly pleased to have your actual character ready")


# ---------------------------------------------------------------------------
# Step-by-step character creation
# ---------------------------------------------------------------------------
def _draft_status() -> str:
    d = _draft or {}
    need = []
    if not d.get("clazz"):
        need.append("class (wizard/cleric/rogue/fighter/ranger/bard)")
    have = []
    if d.get("race"):
        have.append(f"race {d['race']}")
    if d.get("clazz"):
        have.append(f"class {d['clazz']}")
    if d.get("name"):
        have.append(f"name {d['name']}")
    have.append(f"level {d.get('level', DEFAULT_LEVEL)}")
    if d.get("manual"):
        have.append("scores " + ", ".join(f"{a} {v}" for a, v in d["manual"].items()))
    elif d.get("method"):
        have.append(f"stats: {d['method']}")
    s = "[creating] " + (", ".join(have) if have else "nothing set yet") + "."
    if need:
        s += " Still needed: " + "; ".join(need) + "."
    s += " Say 'mira finish the character' when ready (or 'mira cancel')."
    return s


def _handle_creation(event, rest: str, notify, speak) -> bool:
    """Consume a creation-step command. Returns True (always consumed while creating, unless it's
    a finish/cancel handled here). Assumes _draft is not None and _lock is NOT held."""
    global _draft, _char, _active

    if re.search(r"\b(cancel|abort|nevermind|never mind|stop creating|forget it)\b", rest):
        with _lock:
            _draft = None
        notify("[D&D] Character creation cancelled.")
        return True

    if re.search(r"\b(finish|done|finalize|complete|that'?s it|build it|create it|ready)\b", rest):
        with _lock:
            d = _draft
            if not d.get("clazz"):
                notify("[D&D] I still need a class first — say e.g. 'mira be a wizard'.")
                return True
            scores = d["manual"] if len(d.get("manual") or {}) == 6 else None
            array = eng.STANDARD_ARRAY if d.get("method") == "standard array" else None
            _char, report = eng.build_character(
                clazz=d["clazz"], race=d.get("race", "human"),
                name=d.get("name") or "Adventurer", level=d.get("level", DEFAULT_LEVEL),
                scores=scores, array=array)
            _draft = None
            _save(_char)
        notify("[D&D] Character created:")
        notify(_char.sheet_panel())
        _narrate(event, [f"Character finished: {_char.short_summary()}.", report], speak,
                 flavor_note="you just finished making your character and shyly show it to the table")
        return True

    changed = False
    with _lock:
        d = _draft
        # A name command only sets the name (so "name yourself Ranger" can't set the class).
        m = re.search(r"\b(?:name(?: yourself| is| me)?|call yourself|named)\s+(?P<nm>[a-z' -]+)$", rest)
        if m:
            d["name"] = m.group("nm").strip().title()[:30]; changed = True
        else:
            # race/class by bare word anywhere ("be an elf wizard" sets both; "wizard" alone works)
            mr = re.search(r"\b(?P<race>" + _RACE_RE + r")\b", rest)
            if mr:
                d["race"] = mr.group("race").replace(" ", "-"); changed = True
            mc = re.search(r"\b(?P<clazz>" + _CLASS_RE + r")\b", rest)
            if mc:
                d["clazz"] = mc.group("clazz"); changed = True
        m = re.search(r"\blevel\s+(\d{1,2})\b", rest)
        if m:
            d["level"] = max(1, min(20, int(m.group(1)))); changed = True
        if re.search(r"\b(roll|reroll)\b.*\bstat", rest) or re.search(r"\broll (my|your|the) stats?\b", rest):
            d["method"] = "roll"; d["manual"] = {}; changed = True
        if re.search(r"\bstandard array\b", rest):
            d["method"] = "standard array"; d["manual"] = {}; changed = True
        m = re.search(r"\bset\s+(?P<ab>str|dex|con|int|wis|cha|strength|dexterity|constitution|"
                      r"intelligence|wisdom|charisma)\w*\s+(?:to |= )?(?P<v>\d{1,2})\b", rest)
        if m:
            d.setdefault("manual", {})[_AB_WORD[m.group("ab")]] = max(1, min(20, int(m.group("v"))))
            changed = True
        show = re.search(r"\b(show|review|status|what)\b", rest)

    if changed:
        notify(_draft_status())
        return True
    if show:
        notify(_draft_status())
        return True
    # Not a recognized creation step. Consume it anyway (we're mid-creation) with a nudge,
    # so a stray sentence doesn't leak into normal chat and derail the build.
    notify("[D&D] (creating) set race/class/name/level/stats, or 'mira finish the character'.")
    return True


# ---------------------------------------------------------------------------
# Roll-intent detection during play — SEARCH the whole message (so "mira, go
# ahead and attack the goblin" routes to the engine instead of free chat).
# ---------------------------------------------------------------------------
def _do_roll(rest: str, adv: int):
    """Return an engine result string for a play command in `rest`, or None if it isn't one."""
    if re.search(r"\binitiative\b", rest):
        return _char.initiative()
    m = re.search(r"\broll\s+(?:a |an )?(\d*d\d+\s*(?:[+-]\s*\d+)?)\b", rest)
    if m:
        total, detail = eng.roll_expr(m.group(1))
        return _char._log(f"Roll: {detail}")
    m = re.search(r"\b(str|dex|con|int|wis|cha|strength|dexterity|constitution|intelligence|"
                  r"wisdom|charisma)\w*\s+(?:saving throw|save)\b", rest)
    if m:
        return _char.save(_AB_WORD[m.group(1)], adv)
    # skill check: a skill name plus a check/roll cue
    if re.search(r"\b(check|roll|make|attempt|try)\b", rest):
        for skill in eng.SKILLS:
            if re.search(r"\b" + re.escape(skill) + r"\b", rest):
                return _char.check(skill, adv)
        m = re.search(r"\b(strength|dexterity|constitution|intelligence|wisdom|charisma)\b", rest)
        if m and re.search(r"\bcheck\b", rest):
            return _char.ability_check(_AB_WORD[m.group(1)], adv)
    m = re.search(r"\bcast\s+(?P<spell>[a-z' ]+?)\s*(?:\bat\b|\bon\b|\bspell\b|$)", rest)
    if m:
        r = _char.cast(m.group("spell").strip(), adv)
        _save(_char)
        return r
    if re.search(r"\b(attack|roll to hit|make an attack|swing (?:my|her|at)|fire (?:my|at|an "
                 r"arrow|the bow)|shoot(?:s|ing)?\b)", rest):
        wm = re.search(r"\bwith (?:my |her |the |a |an )?(?P<wep>[a-z'+ ]+?)(?:\s+(?:at|on|against|"
                       r"toward|towards)\b|$)", rest)
        return _char.attack_with(wm.group("wep").strip() if wm else None, adv)
    m = re.search(r"\btake[sn]?\s+(\d+)(?:\s+points?)?(?:\s+of)?\s+damage\b", rest) \
        or re.search(r"\b(\d+)\s+(?:points? of\s+)?damage\b", rest)
    if m:
        r = _char.take_damage(int(m.group(1)))
        _save(_char)
        return r
    m = re.search(r"\bheal(?:ed|s)?(?:\s+for)?\s+(\d+)\b", rest) or re.search(r"\bgains?\s+(\d+)\s*(?:hp|health)\b", rest)
    if m:
        r = _char.heal(int(m.group(1)))
        _save(_char)
        return r
    if re.search(r"\blong rest\b", rest):
        r = _char.long_rest()
        _save(_char)
        return r
    return None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def intercept(event, *, notify: Callable[[str], None], speak: Callable) -> bool:
    global _active, _char, _draft
    if not _streamer_channel(event):
        return False
    rest = _strip_name(_norm(getattr(event, "text", "") or ""))
    if rest is None:
        return False   # not addressed by name — normal conversation

    with _lock:
        active, creating, has_char = _active, (_draft is not None), (_char is not None)

    # ---- leave (works anytime) ----
    if active and re.search(r"\b(leave|stop|end|quit|done with|exit)\b.*\b(party|dnd|d and d|"
                            r"dungeons|campaign|game)\b", rest):
        with _lock:
            if _char is not None:
                _save(_char)
            _active = False
            _draft = None
        notify(f"[D&D] {NAME} steps away from the table (character saved).")
        _narrate(event, ["You are stepping away from the game; your character is saved."],
                 speak, flavor_note="a quiet goodbye to the table")
        return True

    # ---- create a character (also enters mode) ----
    if re.search(r"\b(create|make|build|roll up)\b.*\bcharacter\b", rest) \
            or re.search(r"\blet'?s create\b", rest):
        with _lock:
            _active = True
            _draft = {"race": None, "clazz": None, "name": None,
                      "level": DEFAULT_LEVEL, "method": "roll", "manual": {}}
        notify("[D&D] Let's build your character, step by step.")
        notify(_draft_status())
        _narrate(event, ["You're about to make a new D&D character together, step by step."],
                 speak, flavor_note="you're shyly excited to build a character")
        return True

    # ---- import from / sync with D&D Beyond (also enters mode) ----
    if re.search(r"\bsync\b", rest) and re.search(r"\b(character|sheet|dndbeyond|ddb|beyond)\b", rest):
        with _lock:
            cur = _char
        if cur is None or not cur.ddb_id:
            notify("[D&D] Nothing to sync — first load one with "
                   "'mira use your dndbeyond character <url>'.")
            with _lock:
                _active = True
            return True
        _ddb_import(event, cur.ddb_id, notify, speak)
        return True
    if re.search(r"\b(dndbeyond|d and d beyond|d&d ?beyond|ddb|beyond20)\b", rest) \
            or (re.search(r"\bimport\b", rest) and re.search(r"\bcharacter\b", rest)):
        from . import dnd_beyond as _ddb
        cid = _ddb.parse_character_id(getattr(event, "text", "") or "")
        if not cid:
            notify("[D&D] Give me your D&D Beyond character URL, e.g. 'mira use your dndbeyond "
                   "character https://www.dndbeyond.com/characters/12345678' (set it to Public first).")
            with _lock:
                _active = True
            return True
        _ddb_import(event, cid, notify, speak)
        return True

    # ---- use / load a saved character (also enters mode) ----
    m = re.search(r"\buse (?:the )?character\s+(?P<nm>[a-z' -]+)$", rest) \
        or re.search(r"\b(?:load|play as|be)\s+(?P<nm>[a-z' -]+?)\s*$", rest) if \
        re.search(r"\buse (?:the )?character\b|\bload\b", rest) else None
    if re.search(r"\buse (?:the )?character\b|\bload character\b", rest):
        mm = re.search(r"character\s+(?P<nm>[a-z0-9' -]+)$", rest)
        who = mm.group("nm").strip() if mm else ""
        c = _load_by_name(who) if who else None
        if c is None:
            saved = _list_saved()
            notify("[D&D] No saved character by that name. "
                   + (f"Saved: {', '.join(saved)}." if saved else "None saved yet — say 'mira create a character'."))
            with _lock:
                _active = True
            return True
        with _lock:
            _active = True
            _char = c
            _draft = None
        notify(f"[D&D] Loaded {c.name}.")
        notify(c.sheet_panel())
        _narrate(event, [f"You take up your character again: {c.short_summary()}."], speak,
                 flavor_note="quietly glad to play this character again")
        return True

    if re.search(r"\blist (my |saved )?characters?\b", rest):
        saved = _list_saved()
        notify("[D&D] Saved characters: " + (", ".join(saved) if saved else "none yet."))
        return True

    # ---- enter mode (no character, no auto-create) ----
    if not active and (re.search(r"\b(play|playing|start)\b.*\b(dnd|d and d|dungeons|the party)\b", rest)
                       or re.search(r"\bjoin (the )?party\b", rest) or rest in ("dnd mode", "dnd")):
        with _lock:
            _active = True
        saved = _list_saved()
        notify("[D&D] " + NAME + " is at the table. She has no character yet — "
               "say 'mira create a character' to build one"
               + (f", or 'mira use character <name>' (saved: {', '.join(saved)})." if saved else "."))
        _narrate(event, ["You sit down at the D&D table, but you don't have a character yet."],
                 speak, flavor_note=("you're willing to play but need to make or pick a character "
                                     "first; say so shyly"))
        return True

    if not active:
        return False   # not in D&D mode and not a start/create/load command

    # ---- in mode ----
    if creating:
        return _handle_creation(event, rest, notify, speak)

    if re.search(r"\b(show|read|check)\b.*\b(sheet|character|stats)\b", rest) or rest in ("sheet", "character sheet"):
        if not has_char:
            notify("[D&D] No character yet — 'mira create a character' or 'mira use character <name>'.")
            return True
        notify(_char.sheet_panel())
        _narrate(event, [f"Your sheet was shown to the table: {_char.short_summary()}."],
                 speak, flavor_note="a small shy comment about your character")
        return True

    if re.search(r"\breroll\b.*\bstat", rest) and has_char:
        with _lock:
            _char, report = eng.build_character(clazz=_char.clazz, race=_char.race,
                                                name=_char.name, level=_char.level)
            _save(_char)
        notify(_char.sheet_panel())
        _narrate(event, [f"Stats rerolled: now {_char.short_summary()}.", report], speak)
        return True

    if not has_char:
        # In mode but no character: only creation/load commands (handled above) do anything.
        if re.search(r"\b(roll|attack|cast|initiative|save|check|damage|heal|turn|options)\b", rest):
            notify("[D&D] No character yet — say 'mira create a character', "
                   "'mira use character <name>', or 'mira use your dndbeyond character <url>'.")
            return True
        return False

    # ---- combat turn / action economy ----
    if re.search(r"\b(start|begin|it'?s)\b.*\bturn\b", rest) or rest in ("my turn", "your turn"):
        r = _char.start_turn()
        notify("[turn] " + r)
        notify(_char.turn_options())
        _narrate(event, [r, _char.turn_options()], speak,
                 flavor_note="it's your turn in combat; softly note what you're weighing up")
        return True
    if re.search(r"\b(end|finish|done with)\b.*\bturn\b", rest) or rest in ("end turn", "pass"):
        notify("[turn] " + _char.end_turn())
        _narrate(event, ["Your turn is over."], speak, flavor_note="quietly pass the turn on")
        return True
    if re.search(r"\bwhat can (you|i) do\b", rest) or re.search(r"\bwhat (are|s) (my|your) options\b", rest) \
            or re.search(r"^(my |your )?options$", rest) or re.search(r"\bwhat'?s available\b", rest):
        notify(_char.turn_options())
        _narrate(event, [_char.turn_options()], speak,
                 flavor_note="softly think aloud about what you could do this turn")
        return True
    m = re.search(r"\b(?:move|walk|run|step)\s+(\d+)\s*(?:ft|feet|foot)?\b", rest) \
        or re.search(r"\b(\d+)\s*(?:ft|feet|foot)\b", rest)
    if m and re.search(r"\b(move|walk|run|step|feet|ft|foot)\b", rest):
        r = _char.spend_move(int(m.group(1)))
        notify("[turn] " + r)
        _narrate(event, [r], speak, flavor_note="quietly move into position")
        return True
    if re.search(r"\b(use|take)\b.*\breaction\b", rest) or rest.startswith("reaction"):
        what = re.sub(r"^.*?reaction\s*(?:to |for |and )?", "", rest).strip() or "a reaction"
        r = _char.use_reaction(what)
        notify("[turn] " + r)
        _narrate(event, [r], speak)
        return True
    m = re.search(r"^(?:i |i'?ll |we )?(?:take the |use )?(?P<act>dash|dodge|disengage)\b", rest)
    if m:
        r = _char.use_action(m.group("act").title())
        notify("[turn] " + r)
        _narrate(event, [r], speak, flavor_note="do it quietly, in character")
        return True

    # play: engine rolls, she announces
    result = _do_roll(rest, _adv(rest))
    if result is None:
        return False   # not a mechanical command -> normal in-character conversation
    notify(f"[dice] {result}")
    _narrate(event, [result], speak)
    return True


# ---------------------------------------------------------------------------
# Dice-claim scrub — used by main.py on her NORMAL (non-engine) replies while
# the mode is active, so she can't state a fabricated roll/number/outcome.
# ---------------------------------------------------------------------------
_DICE_CLAIM_RE = re.compile(
    r"\bnat(?:ural)?\s*(?:20|1|twenty|one)\b"
    r"|\b(?:roll(?:ed|s)?|got|scored)\s+(?:a |an |for )?\d+\b"
    r"|\broll\s+of\s+\d+\b"
    r"|\b\d+\s*(?:to\s*hit|to\s*the\s*hit)\b"
    r"|\b(?:hit|hits|deal|deals|dealt|do|does|did|take|takes|for)\s+\d+\s*(?:damage|points|hp)\b"
    r"|\b\d+\s*(?:damage|points\s+of\s+damage|hp)\b"
    r"|\btotal(?:s|ing|ling)?\s+(?:of\s+)?\d+\b"
    r"|\bthat'?s\s+(?:a |an )?\d+\b"
    r"|\b(?:critical\s+hit|crit(?:s|ical)?)\b"
    r"|\bi\s+(?:hit|miss(?:ed)?|crit|succeed|fail(?:ed)?)\b",
    re.I)


def _has_dice_claim(sentence: str) -> bool:
    return bool(_DICE_CLAIM_RE.search(sentence or ""))


def scrub_text(text: str) -> str:
    if not text or not is_active():
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts if p.strip() and not _has_dice_claim(p)]
    return " ".join(kept).strip()


def scrub_stream(sentences):
    """Wrap a sentence generator: drop any sentence that states a die result/number/outcome
    (she must never fake a roll — the engine does that). Pass-through when the mode is off."""
    if not is_active():
        yield from sentences
        return
    for s in sentences:
        if not _has_dice_claim(s):
            yield s


def scrub(reply):
    """Scrub either a plain string reply or a streamed sentence generator."""
    if isinstance(reply, str):
        return scrub_text(reply)
    return scrub_stream(reply)


# ---------------------------------------------------------------------------
# Situation note — keeps ordinary replies in-game AND forbids inventing dice
# ---------------------------------------------------------------------------
def situation_note() -> str:
    with _lock:
        if not _active:
            return ""
        if _draft is not None:
            return ("You are building a new D&D character together, step by step. Talk it through "
                    "shyly in character, but do not roll or state any numbers — the tools handle "
                    "stats when the human confirms them.")
        if _char is None:
            return ("You are at a D&D table but you do NOT have a character yet. If asked to do "
                    "anything, gently say you still need to make or pick a character first.")
        recent = " | ".join(_char.log[-3:]) if _char.log else "none yet"
        who = _char.short_summary()
        weps = ", ".join(w["name"] for w in _char.weapons) or "unarmed"
        spells = ", ".join(s["name"] for s in _char.spell_list) or "none"
        turn = ("\nIt is your turn. " + _char.turn_options().replace("\n", " ")) if _char.in_turn else ""
    return (f"You are playing Dungeons & Dragons at the table as your character: {who}. A human DM "
            f"runs the game. Your weapons: {weps}. Your spells: {spells}. On your turn you have an "
            f"action, a bonus action, a reaction, and movement (per 5e). CRITICAL: you do NOT roll "
            f"dice and you must NEVER say a die number, a to-hit or damage amount, or whether you "
            f"hit, missed, succeeded, or failed — the engine rolls everything. When you act, "
            f"describe ONLY what your character ATTEMPTS ('she swings her sword at it', 'I cast fire "
            f"bolt at the goblin') and let the roll happen." + turn
            + f"\nRecent engine results (already announced): {recent}.")


def finalize_if_active(*, notify: Callable[[str], None]) -> None:
    with _lock:
        if _active and _char is not None:
            _save(_char)
            notify(f"[D&D] {NAME}'s character was saved.")
