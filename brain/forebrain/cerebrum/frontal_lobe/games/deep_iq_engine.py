"""
deep_iq_engine.py - the authoritative rules engine for "Phill's Modified Deep IQ 2017",
a solo Magic: The Gathering AI opponent. PURE engine: real dice + tracked state + the
tables encoded as data. No LLM, no I/O - so it's deterministic and unit-testable, and Mira
(the LLM) only ever *narrates* what this engine actually did. That's the whole point: the
model kept faking dice ("rolled a 4" every time) and forgetting state; here the dice are
real (random) and the state (table, life, both battlefields, modifiers) lives in code.

Deep IQ has no deck. Each of its turns is a d10 roll on its current Table (I-VI); the result
can make tokens, remove the player's stuff, change life, move tables, or roll on the Token
Chart (creature abilities) or the Spooky Chart (big effects). After the turn it rolls to
advance to the next table. Effects that touch the PLAYER's cards are emitted as instructions
(with the engine picking the target from the tracked player board); the player resolves them
with their physical cards and reports the outcome.

Result text comes straight from the PDF; `effects` are the structured, engine-applied parts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Colour codes. WUBRG plus the two non-colour identities.
COLOURS = ["W", "U", "B", "R", "G"]
COLOUR_NAME = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green",
               "Artifact": "Artifact", "Colourless": "Colourless"}

# How high you can roll on the per-table ADVANCEMENT roll and still advance one table.
# (Table VI has none - Deep IQ stays there.)
ADVANCE_MAX = {1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 0}
START_LIFE = 20


# ---------------------------------------------------------------------------
# Dice - the ONLY source of randomness. Everything random goes through here.
# ---------------------------------------------------------------------------
def roll(sides: int = 10, mod: int = 0) -> int:
    """Roll one die (default d10) and add a modifier. Real RNG."""
    return random.randint(1, sides) + mod


# ---------------------------------------------------------------------------
# Effect helpers - small structured ops the engine applies deterministically.
# A table/chart "entry" is {"text": <PDF wording>, "effects": [<ops>]}.
# ---------------------------------------------------------------------------
def E(text: str, *effects) -> dict:
    return {"text": text, "effects": list(effects)}


# effect op constructors (kept as tuples: (op_name, *args))
def nothing():           return ("nothing",)
def token(p, t, mod=0):  return ("token", p, t, mod)          # create token, roll Token Chart at +mod
def move(n):             return ("move", n)                   # move Deep IQ up to Table n
def drop(n):             return ("drop", n)                   # send Deep IQ back to Table n
def diq_life(d):         return ("diq_life", d)
def player_life(d):      return ("player_life", d)
def free_roll(n):        return ("free_roll", n)              # extra roll on Table n this turn
def spooky(mod=0):       return ("spooky", mod)
def do_nothing_next():   return ("do_nothing_next",)
def roll_bonus(d):       return ("roll_bonus", d)             # persistent +d to all Deep IQ rolls
def note(text):          return ("note", text)               # freeform instruction for the player
def target(verb, kind, count=1):  return ("target", verb, kind, count)  # hits the player's board
def dmg_creature(n, count=1):     return ("dmg_creature", n, count)     # N damage to player's best creature(s)


# ---------------------------------------------------------------------------
# Colour-branched results. The PDF gives many results per single colour; some
# group colours ("Red / Black / Green"). We store a branch as a list of
# (colours, entry) and resolve by Deep IQ's *active colour* for the turn.
# ---------------------------------------------------------------------------
def branch(*pairs, default=None) -> dict:
    """pairs: (colour_string_or_tuple, entry). e.g. branch(("R","B","G"), E(...))."""
    norm = []
    for cols, entry in pairs:
        if isinstance(cols, str):
            cols = (cols,)
        norm.append((tuple(cols), entry))
    return {"_branch": norm, "_default": default}


# A small "removal" entry generator (most colour rows are just a kind of removal,
# sometimes with a Deep IQ side effect appended).
def kill(verb, kind="creature", count=1, *extra):
    label = {"sacrifice": "Sacrifice", "destroy": "Destroy", "bounce": "Bounce",
             "exile": "Exile", "arrest": "Arrest", "fight": "Deep IQ Fights"}[verb]
    plural = "s" if count > 1 else ""
    txt = f"{label} your {'two best ' if count == 2 else 'best '}target {kind}{plural}."
    return E(txt, target(verb, kind, count), *extra)


# ===========================================================================
# TABLES I-VI  (roll d10 -> result). Keys are the roll value (1..10). A value
# can be an entry E(...) or a colour branch(...). Special multi-step rows
# (Table V #9, Table VI #4) are handled in code.
# ===========================================================================
def _table_I():
    eight = branch(
        ("B", kill("sacrifice", "creature")),
        ("R", E("Lightning Bolt your best target creature (3 damage).", dmg_creature(3))),
        ("W", kill("arrest", "creature")),
        ("U", kill("bounce", "creature")),
        ("G", E("Deep IQ one-way Fights your best target creature.", target("fight", "creature"))),
        ("Artifact", E("Sacrifice your best target creature. Deep IQ stays on Table I.",
                       target("sacrifice", "creature"))),
        ("Colourless", E("Sacrifice your best target creature. Deep IQ loses 1 life.",
                         target("sacrifice", "creature"), diq_life(-1))),
    )
    return {r: E("Do nothing.", nothing()) for r in range(1, 8)} | {
        8: eight,
        9: E("Put a 1/1 token on the battlefield (token roll -4).", token(1, 1, -4)),
        10: E("Put a 1/1 token on the battlefield (token roll -4).", token(1, 1, -4)),
    }


def _table_II():
    nine_ten = branch(
        ("W", kill("exile", "creature")),
        ("R", E("4 damage to your best target creature, exile it if it dies.", dmg_creature(4))),
        ("B", kill("sacrifice", "creature")),
        ("U", E("Bounce your best target creature to the top of your deck.", target("bounce", "creature"))),
        ("G", E("Deep IQ one-way Fights your best target creature with 2 creatures.",
                target("fight", "creature"))),
        ("Artifact", E("Exile your best target creature. Deep IQ has -2 on its next roll.",
                       target("exile", "creature"), ("next_minus", 2))),
        ("Colourless", E("Exile your best target creature. Deep IQ loses 1 life.",
                         target("exile", "creature"), diq_life(-1))),
    )
    return {1: E("Do nothing.", nothing()), 2: E("Do nothing.", nothing()),
            3: E("Do nothing.", nothing()), 4: E("Do nothing.", nothing()),
            5: E("Put a 2/2 token on the battlefield (+0).", token(2, 2, 0)),
            6: E("Put a 2/2 token on the battlefield (+0).", token(2, 2, 0)),
            7: E("Put a 2/2 token on the battlefield (+0).", token(2, 2, 0)),
            8: E("Move Deep IQ up to Table IV.", move(4)),
            9: nine_ten, 10: nine_ten}


def _table_III():
    six = branch(
        (("R", "B", "G"), E("Destroy your best target land.", target("destroy", "land"))),
        ("U", E("Bounce your best target land.", target("bounce", "land"))),
        ("W", E("Exile your best target land; reveal until a basic land, put it into play tapped, "
                "shuffle the revealed cards into your library.", target("exile", "land"))),
        ("Artifact", E("Destroy your best target land. Deep IQ stays on Table III.",
                       target("destroy", "land"))),
        ("Colourless", E("Destroy your best target land. Deep IQ loses 1 life.",
                         target("destroy", "land"), diq_life(-1))),
    )
    nine = branch(
        ("B", kill("sacrifice", "creature")),
        ("R", E("4 damage to your best creature.", dmg_creature(4))),
        ("W", kill("arrest", "creature")),
        ("U", kill("bounce", "creature")),
        ("G", E("Deep IQ one-way Fights your best creature with 2 creatures.", target("fight", "creature"))),
        ("Artifact", E("Sacrifice your best creature. Deep IQ moves back to Table II.",
                       target("sacrifice", "creature"), drop(2))),
        ("Colourless", E("Sacrifice your best creature. Deep IQ loses 2 life.",
                         target("sacrifice", "creature"), diq_life(-2))),
    )
    ten = branch(
        (("R", "G", "W"), E("Destroy your best target artifact.", target("destroy", "artifact"))),
        ("U", E("Bounce your best target artifact.", target("bounce", "artifact"))),
        ("Artifact", E("Destroy your best target artifact. Deep IQ moves back to Table II.",
                       target("destroy", "artifact"), drop(2))),
        ("Colourless", E("Destroy your best target artifact. Deep IQ loses 1 life.",
                         target("destroy", "artifact"), diq_life(-1))),
        ("B", E("Roll on the Spooky Chart (-2).", spooky(-2))),
    )
    return {1: E("Do nothing.", nothing()), 2: E("Do nothing.", nothing()), 3: E("Do nothing.", nothing()),
            4: E("Put a 2/2 token on the battlefield (+2).", token(2, 2, 2)),
            5: E("Put a 2/1 token on the battlefield (+4).", token(2, 1, 4)),
            6: six,
            7: E("Move Deep IQ up to Table V and put a 1/1 token on the battlefield (+0).",
                 move(5), token(1, 1, 0)),
            8: E("Put a 1/1 token on the battlefield (+1) and Deep IQ gets a free roll on Table II.",
                 token(1, 1, 1), free_roll(2)),
            9: nine, 10: ten}


def _table_IV():
    five = branch(
        ("B", kill("sacrifice", "creature")),
        ("R", E("4 damage to your best target creature.", dmg_creature(4))),
        ("W", kill("arrest", "creature")),
        ("U", kill("bounce", "creature")),
        ("G", E("Deep IQ one-way Fights your best target creature.", target("fight", "creature"))),
        ("Artifact", E("Sacrifice your best target creature. Deep IQ stays on Table IV.",
                       target("sacrifice", "creature"))),
        ("Colourless", E("Sacrifice your best target creature. Deep IQ loses 1 life.",
                         target("sacrifice", "creature"), diq_life(-1))),
    )
    six = branch(
        (("G", "W"), E("Destroy your best target artifact or enchantment.",
                       target("destroy", "artifact or enchantment"))),
        ("R", E("Destroy your best target artifact.", target("destroy", "artifact"))),
        ("U", E("Bounce your best target artifact or enchantment.",
                target("bounce", "artifact or enchantment"))),
        ("Artifact", E("Destroy your best target artifact or enchantment. Deep IQ moves back to Table III.",
                       target("destroy", "artifact or enchantment"), drop(3))),
        ("Colourless", E("Destroy your best target artifact or enchantment. Deep IQ loses 1 life.",
                         target("destroy", "artifact or enchantment"), diq_life(-1))),
        ("B", E("You lose 1 life, Deep IQ gains 1 life.", player_life(-1), diq_life(1))),
    )
    seven = branch(
        ("W", kill("exile", "creature")),
        ("R", E("4 damage to your best creature, exile it if it dies.", dmg_creature(4))),
        ("B", kill("sacrifice", "creature")),
        ("U", E("Bounce your best creature to the top of your deck.", target("bounce", "creature"))),
        ("G", E("Deep IQ one-way Fights your best creature with 2 creatures.", target("fight", "creature"))),
        ("Artifact", E("Exile your best creature. Deep IQ moves back to Table III.",
                       target("exile", "creature"), drop(3))),
        ("Colourless", E("Exile your best creature. Deep IQ loses 2 life.",
                         target("exile", "creature"), diq_life(-2))),
    )
    # Row 8: "If you have no creatures take 4 damage, otherwise <branch on 2 best creatures>"
    eight = branch(
        ("B", kill("sacrifice", "creature", 2)),
        ("R", E("4 damage to your best creature and 4 damage to you.", dmg_creature(4), player_life(-4))),
        ("W", kill("arrest", "creature", 2)),
        ("U", kill("bounce", "creature", 2)),
        ("G", E("Deep IQ one-way Fights your 2 best creatures with 2 creatures.",
                target("fight", "creature", 2))),
        ("Artifact", E("Sacrifice your 2 best creatures. Deep IQ moves back to Table III.",
                       target("sacrifice", "creature", 2), drop(3))),
        ("Colourless", E("Sacrifice your 2 best creatures. Deep IQ loses 4 life.",
                         target("sacrifice", "creature", 2), diq_life(-4))),
    )
    eight["_no_creature"] = E("You have no creatures: take 4 damage.", player_life(-4))
    nine = branch(
        (("U", "Artifact", "Colourless"), E("Roll on the Spooky Chart (-1).", spooky(-1))),
        default=E("Put a 2/4 token on the battlefield (+7).", token(2, 4, 7)),
    )
    return {1: E("Do nothing.", nothing()), 2: E("Do nothing.", nothing()), 3: E("Do nothing.", nothing()),
            4: E("Put a 4/4 token on the battlefield (+3).", token(4, 4, 3)),
            5: five, 6: six, 7: seven, 8: eight, 9: nine,
            10: E("Roll on the Spooky Chart (+0).", spooky(0))}


def _table_V():
    six = E("Destroy your best target creature, enchantment, or artifact.",
            target("destroy", "creature, enchantment, or artifact"))
    return {1: E("Do nothing.", nothing()), 2: E("Do nothing.", nothing()), 3: E("Do nothing.", nothing()),
            4: E("Put a 3/4 token on the battlefield (+4).", token(3, 4, 4)),
            5: E("Put a 2/2 token on the battlefield (+2) and Deep IQ gets a free roll on Table III.",
                 token(2, 2, 2), free_roll(3)),
            6: six,
            7: E("Put a 4/4 token on the battlefield (+1).", token(4, 4, 1)),
            8: E("Destroy all lands OR put a 4/1 token on the battlefield (+3) - Deep IQ chooses randomly.",
                 ("choice_V8",)),
            9: ("special_V9",),
            10: E("Roll on the Spooky Chart (+2).", spooky(2))}


def _table_VI():
    return {1: E("Do nothing.", nothing()), 2: E("Do nothing.", nothing()), 3: E("Do nothing.", nothing()),
            4: ("special_VI4",),
            5: E("Put a 4/5 token on the battlefield (+6).", token(4, 5, 6)),
            6: E("Destroy your best target creature.", target("destroy", "creature")),
            7: E("You take 6 target damage.", player_life(-6)),
            8: E("Destroy your best target artifact, enchantment, or land.",
                 target("destroy", "artifact, enchantment, or land")),
            9: E("Exile your best creature OR roll on the Spooky Chart (+3) - Deep IQ chooses randomly.",
                 ("choice_VI9",)),
            10: E("Roll on the Spooky Chart (+4).", spooky(4))}


TABLES = {1: _table_I(), 2: _table_II(), 3: _table_III(), 4: _table_IV(), 5: _table_V(), 6: _table_VI()}
TABLE_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


# ---------------------------------------------------------------------------
# Token Chart (creature abilities) and Spooky Chart - resolved in code.
# ---------------------------------------------------------------------------
def token_chart(active: str, modded_roll: int) -> List[str]:
    """Return the list of ability strings for a token, from a (modified) d10 roll.
    Rolls can exceed 10 via modifiers; 10 means 'roll twice more', handled by caller."""
    r = modded_roll
    if r <= 1:
        return []
    if r == 2:
        if active in ("B", "R"): return ["+2/+0", "first strike"]
        if active in ("W", "Artifact", "Colourless"): return ["+1/+0", "first strike"]
        if active == "G": return ["+1/+1", "trample"]
        if active == "U": return ["+1/+0", "flying"]
    if r == 3:
        return ["regeneration (once/turn; if used, -2 on Deep IQ's next roll)"]
    if r == 4:
        if active in ("W", "U"): return ["+0/+3", "defender"]
        if active == "G": return ["+0/+2", "reach"]
        return ["+0/+2", "defender"]
    if r == 5:
        if active == "U": return ["flying"]
        if active == "G": return ["+1/+1"]
        return ["first strike"]
    if r == 6:
        c = roll(10)
        prot = ("black" if c <= 3 else "white" if c <= 6 else "red" if c <= 8 else "blue" if c == 9 else "green")
        return [f"protection from {prot}"]
    if r == 7:
        if active in ("B", "G", "Artifact", "Colourless"): return ["deathtouch"]
        if active == "U": return ["flying"]
        if active == "R": return ["+2/+0"]
        if active == "W": return ["+0/+2"]
    if r == 8:
        out = ["+2/+2"]
        if active in ("B", "W"): out += ["flying", "lifelink"]
        elif active == "G": out += ["+1/+1", "trample"]
        elif active in ("U", "R", "Artifact"): out += ["flying"]
        elif active == "Colourless": out += ["annihilator 1"]
        return out
    if r == 9:
        return ["haste", "trample"]
    if r == 11:
        return ["flying", "trample"]
    if r == 12:
        return ["protection from a colour", "vigilance"]
    if r == 13:
        return ["ETB: sacrifice one of your creatures at random"]
    if r == 14:
        return ["first strike", "shroud"]
    if r == 15:
        return ["protection from a colour", "deathtouch", "your weakest creature becomes unblockable"]
    if r >= 16:
        return ["ETB: exile target permanent you control"]
    return []


SPOOKY = {
    1: E("Deep IQ plays an enchantment token: while it's out, all its creature tokens get +1/+1.",
         ("diq_token", "enchantment", "all Deep IQ creature tokens get +1/+1")),
    2: E("Deep IQ plays an artifact token: while it's out, reroll the first 'Do nothing' each turn.",
         ("diq_token", "artifact", "reroll the first 'Do nothing' result each turn")),
    3: E("Deep IQ plays an enchantment token: while it's out, Deep IQ gets +1 to all die rolls.",
         ("diq_token", "enchantment", "+1 to all Deep IQ die rolls"), roll_bonus(1)),
    4: E("Destroy all your creatures (W/B), or all your artifacts (R/G), or all your enchantments "
         "(G/W) - or Bounce them (U). Treat Deep IQ's next roll as 'Do nothing'.",
         note("mass removal vs the player (choose per Deep IQ's colour)"), do_nothing_next()),
    5: E("Deep IQ gains 5 life and moves up to Table VI if it isn't already there.",
         diq_life(5), move(6)),
    6: E("You take 10 damage.", player_life(-10)),
    7: E("Deep IQ plays an artifact token: while it's out, it gets two table rolls every turn and "
         "takes the best one.", ("diq_token", "artifact", "two table rolls per turn, take the best")),
    8: E("Destroy all your lands of one basic type (the most inconvenient). Treat Deep IQ's next "
         "roll as 'Do nothing'.", note("destroy all of one basic land type"), do_nothing_next()),
    9: E("Exile the top twenty cards of your library.", note("exile your top 20 cards")),
    10: E("Deep IQ plays an artifact token: when it enters, tap your best creature; it stays tapped "
          "while this artifact is out (next best if you lose it).",
          ("diq_token", "artifact", "keeps your best creature tapped")),
    11: E("All of Deep IQ's tokens get a free, permanent roll on the Token Chart (+0).",
          ("retoken_all",)),
    12: E("Deep IQ plays an enchantment token: free roll on Table II whenever one of its permanents "
          "is destroyed or exiled.", ("diq_token", "enchantment", "free Table II roll when a Deep IQ permanent dies")),
    13: E("Deep IQ gains 20 life.", diq_life(20)),
    14: E("Destroy all your permanents. Treat Deep IQ's next roll as 'Do nothing'.",
          note("destroy ALL your permanents"), do_nothing_next()),
}

MONO_BONUS = {
    "W": "alternates +0/+1 / Deep IQ gains 2 life",
    "U": "alternates flying / tap your best untapped creature",
    "B": "you lose 1 life",
    "R": "alternates +1/+0 / 1 damage to any target",
    "G": "alternates +1/+1 / make a 1/1 Saproling",
}


# ===========================================================================
# Game state
# ===========================================================================
@dataclass
class Token:
    power: int
    toughness: int
    kind: str = "creature"          # creature | enchantment | artifact
    abilities: List[str] = field(default_factory=list)
    note: str = ""                  # persistent effect text (for enchant/artifact tokens)
    tapped: bool = False

    def label(self) -> str:
        if self.kind == "creature":
            extra = (" [" + ", ".join(self.abilities) + "]") if self.abilities else ""
            return f"{self.power}/{self.toughness}{extra}"
        return f"{self.kind} token ({self.note})" if self.note else f"{self.kind} token"


@dataclass
class Permanent:
    name: str = "permanent"
    kind: str = "creature"          # creature | artifact | enchantment | land | planeswalker
    power: int = 0
    toughness: int = 0
    tapped: bool = False
    abilities: List[str] = field(default_factory=list)   # flying/first strike/deathtouch/trample/reach

    def label(self) -> str:
        if self.kind == "creature":
            extra = (" [" + ", ".join(self.abilities) + "]") if self.abilities else ""
            return f"{self.power}/{self.toughness} {self.name}{extra}".strip()
        return f"{self.name} ({self.kind})"


@dataclass
class GameState:
    colour_identity: List[str] = field(default_factory=list)   # e.g. ["R","G"] or ["Artifact"]
    identity_label: str = ""
    table: int = 1
    diq_life: int = START_LIFE
    player_life: int = START_LIFE
    turn: int = 0
    diq_board: List[Token] = field(default_factory=list)
    player_board: List[Permanent] = field(default_factory=list)
    # pending / persistent modifiers
    roll_bonus: int = 0                 # persistent +N to all Deep IQ rolls (Spooky #3)
    next_minus: int = 0                 # one-shot -N on the next roll (Annihilate/Artifact rows)
    do_nothing_next: bool = False       # treat next upkeep roll as 'Do nothing'
    free_rolls: List[int] = field(default_factory=list)   # queued extra table rolls this turn
    colour_cycle: int = 0               # round-robin index for multi-colour branched results
    started: bool = False

    # ---- serialization ----
    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "GameState":
        d = json.loads(s)
        d["diq_board"] = [Token(**t) for t in d.get("diq_board", [])]
        d["player_board"] = [Permanent(**p) for p in d.get("player_board", [])]
        return GameState(**d)


# ===========================================================================
# Setup
# ===========================================================================
def _roll_identity() -> (List[str], str):
    """Roll Deep IQ's colour identity per the PDF."""
    n_roll = roll(10)
    if n_roll <= 3:
        kind = "mono"
    elif n_roll <= 7:
        kind = "two"
    elif n_roll <= 9:
        kind = "three"
    else:
        sub = roll(10)
        if sub <= 2:   kind = "four"
        elif sub <= 6: kind = "five"
        elif sub <= 8: return (["Colourless"], "Colourless")
        else:          return (["Artifact"], "Artifact")
    count = {"mono": 1, "two": 2, "three": 3, "four": 4, "five": 5}[kind]

    def one_colour():
        c = roll(10)
        return "W" if c <= 2 else "U" if c <= 4 else "B" if c <= 6 else "R" if c <= 8 else "G"

    cols: List[str] = []
    guard = 0
    while len(cols) < count and guard < 50:
        c = one_colour()
        if c not in cols:
            cols.append(c)
        guard += 1
    label = {1: "Mono", 2: "Two-colour", 3: "Three-colour", 4: "Four-colour", 5: "Five-colour"}[count]
    label += " " + "/".join(COLOUR_NAME[c] for c in cols)
    return cols, label


def setup(colour_identity: Optional[List[str]] = None) -> GameState:
    """Start a new game. Pass an explicit identity (e.g. ['R','G'] / ['Artifact']) or roll it."""
    st = GameState()
    if colour_identity:
        st.colour_identity = list(colour_identity)
        st.identity_label = ("Artifact" if colour_identity == ["Artifact"]
                             else "Colourless" if colour_identity == ["Colourless"]
                             else "/".join(COLOUR_NAME.get(c, c) for c in colour_identity))
    else:
        st.colour_identity, st.identity_label = _roll_identity()
    st.table = 1
    st.diq_life = st.player_life = START_LIFE
    st.turn = 0
    st.started = True
    return st


def _active_colour(st: GameState) -> str:
    """Deep IQ's colour for a branched result this turn. Mono/Artifact/Colourless -> that one;
    multi-colour -> round-robin through its colours (PDF: 'cycle through the colour combinations')."""
    ids = st.colour_identity
    if not ids:
        return "Colourless"
    if len(ids) == 1:
        return ids[0]
    c = ids[st.colour_cycle % len(ids)]
    return c


# ===========================================================================
# Turn execution
# ===========================================================================
def _best_player_target(st: GameState, kind: str):
    """Pick the player's 'best' permanent of a kind (highest power, then toughness for creatures)."""
    kinds = [k.strip() for part in kind.replace(" or ", ",").split(",") for k in [part]]
    kinds = [("creature" if "creature" in k else "artifact" if "artifact" in k
              else "enchantment" if "enchantment" in k else "land" if "land" in k else k)
             for k in kinds]
    cands = [p for p in st.player_board if p.kind in kinds]
    if not cands:
        return None
    if "creature" in kinds:
        cands.sort(key=lambda p: (p.power, p.toughness), reverse=True)
    return cands[0]


def _resolve_entry(st: GameState, entry: dict, log: List[str]):
    """Apply one resolved (non-branch) entry's effects, appending narration to log."""
    log.append("  " + entry["text"])
    for eff in entry["effects"]:
        op = eff[0]
        if op == "nothing":
            pass
        elif op == "token":
            _make_token(st, eff[1], eff[2], eff[3], log)
        elif op == "move":
            if eff[1] > st.table:
                st.table = eff[1]; log.append(f"  -> Deep IQ moves up to Table {TABLE_ROMAN[st.table]}.")
        elif op == "drop":
            st.table = eff[1]; log.append(f"  -> Deep IQ drops back to Table {TABLE_ROMAN[st.table]}.")
        elif op == "diq_life":
            st.diq_life += eff[1]; log.append(f"  -> Deep IQ life {eff[1]:+d} -> {st.diq_life}.")
        elif op == "player_life":
            st.player_life += eff[1]; log.append(f"  -> Your life {eff[1]:+d} -> {st.player_life}.")
        elif op == "free_roll":
            st.free_rolls.append(eff[1]); log.append(f"  -> Deep IQ queues a free roll on Table {TABLE_ROMAN[eff[1]]}.")
        elif op == "spooky":
            _resolve_spooky(st, eff[1], log)
        elif op == "do_nothing_next":
            st.do_nothing_next = True; log.append("  -> Deep IQ's next roll will be treated as 'Do nothing'.")
        elif op == "roll_bonus":
            st.roll_bonus += eff[1]; log.append(f"  -> Deep IQ now has {st.roll_bonus:+d} to all die rolls.")
        elif op == "next_minus":
            st.next_minus += eff[1]; log.append(f"  -> Deep IQ has -{eff[1]} on its next roll.")
        elif op == "note":
            log.append(f"  -> {eff[1]} (resolve with your cards).")
        elif op == "target":
            _resolve_target(st, eff[1], eff[2], eff[3], log)
        elif op == "dmg_creature":
            tgt = _best_player_target(st, "creature")
            if tgt:
                log.append(f"  -> {eff[1]} damage to your best creature: {tgt.label()} "
                           f"(you confirm if it dies; tell me '<that creature> dies').")
            else:
                log.append(f"  -> {eff[1]} damage to your best creature - you have none.")
        elif op == "diq_token":
            _make_special_token(st, eff[1], eff[2], log)
        elif op == "retoken_all":
            for tk in st.diq_board:
                if tk.kind == "creature":
                    extra = token_chart(_active_colour(st), roll(10))
                    tk.abilities += [a for a in extra if a not in tk.abilities]
            log.append("  -> all Deep IQ tokens gained a permanent extra Token Chart roll.")


def _resolve_target(st: GameState, verb: str, kind: str, count: int, log: List[str]):
    tgt = _best_player_target(st, kind)
    label = {"sacrifice": "sacrifice", "destroy": "destroy", "bounce": "bounce",
             "exile": "exile", "arrest": "tap down / Arrest", "fight": "fight"}.get(verb, verb)
    if count == 2:
        log.append(f"  -> {label} your TWO best {kind}s (you resolve with your cards).")
        return
    if tgt is not None:
        log.append(f"  -> {label} your best {kind}: **{tgt.label()}** - resolve it and tell me the result.")
    else:
        log.append(f"  -> {label} your best {kind} - you have none, so nothing happens.")


def _create_token(st: GameState, p: int, t: int, mod: int) -> str:
    """Roll the Token Chart, build the creature token, add it to Deep IQ's board. Returns a line."""
    active = _active_colour(st)
    r = roll(10, mod)
    abilities = token_chart(active, r)
    # Token Chart 10 = roll two more times at +0
    if r == 10:
        for _ in range(2):
            abilities += token_chart(active, roll(10))
    if len(st.colour_identity) == 1 and active in MONO_BONUS:
        abilities.append(f"mono bonus: {MONO_BONUS[active]}")
    # fold explicit +X/+Y ability strings into the token's P/T
    for a in list(abilities):
        if a and a[0] == "+" and "/" in a and a.replace("+", "").replace("/", "").isdigit():
            try:
                dp, dt = a.replace("+", "").split("/")
                p += int(dp); t += int(dt)
            except Exception:
                pass
    abilities = [a for a in abilities if not (a and a[0] == "+" and "/" in a
                 and a.replace("+", "").replace("/", "").isdigit())]
    tok = Token(power=max(0, p), toughness=max(1, t), kind="creature", abilities=abilities)
    st.diq_board.append(tok)
    if len(st.colour_identity) > 1:
        st.colour_cycle += 1
    return (f"  -> Deep IQ makes a {tok.label()} token (token roll {r}"
            + (f", as {COLOUR_NAME.get(active, active)}" if len(st.colour_identity) > 1 else "") + ").")


def _make_token(st: GameState, p: int, t: int, mod: int, log: List[str], depth: int = 0):
    log.append(_create_token(st, p, t, mod))


def _make_special_token(st: GameState, kind: str, effect_note: str, log: List[str]):
    st.diq_board.append(Token(power=0, toughness=0, kind=kind, note=effect_note))
    log.append(f"  -> Deep IQ plays a {kind} token: {effect_note}.")


def _resolve_spooky(st: GameState, mod: int, log: List[str]):
    r = roll(10, mod + st.roll_bonus)
    r = max(min(r, 14), 1)
    entry = SPOOKY[r]
    log.append(f"  -> SPOOKY CHART roll {r}:")
    _resolve_entry(st, entry, log)


def _lookup(st: GameState, table: int, r: int, log: List[str]) -> Optional[dict]:
    """Resolve a table roll value to a concrete entry (handling colour branches + specials)."""
    r = max(1, min(10, r))
    cell = TABLES[table][r]
    # specials
    if isinstance(cell, tuple):
        return _resolve_special(st, cell[0], log)
    if "_branch" in cell:
        active = _active_colour(st)
        if "_no_creature" in cell and not any(p.kind == "creature" for p in st.player_board):
            return cell["_no_creature"]
        for cols, entry in cell["_branch"]:
            if active in cols:
                if len(st.colour_identity) > 1:
                    st.colour_cycle += 1
                return entry
        if cell.get("_default"):
            return cell["_default"]
        # fall back to first branch entry
        return cell["_branch"][0][1]
    return cell


def _resolve_special(st: GameState, tag: str, log: List[str]) -> Optional[dict]:
    if tag == "choice_V8":
        if roll(2) == 1:
            return E("Destroy all your lands.", note("destroy ALL your lands"))
        return E("Put a 4/1 token on the battlefield (+3).", token(4, 1, 3))
    if tag == "special_V9":
        sub = roll(10)
        log.append(f"  -> Table V #9 sub-roll {sub}:")
        if sub <= 5:
            return _branch_to_entry(st, sub)
        return E("Roll on the Spooky Chart (+1).", spooky(1))
    if tag == "choice_VI9":
        if roll(2) == 1:
            return E("Exile your best creature.", target("exile", "creature"))
        return E("Roll on the Spooky Chart (+3).", spooky(3))
    if tag == "special_VI4":
        sub = roll(10)
        log.append(f"  -> Table VI #4 sub-roll {sub}:")
        if sub == 1:
            return E("Sacrifice all lands (Red), else put a 2/4 token (+3).",
                     note("if Deep IQ is Red: destroy all your lands; otherwise a 2/4 token"))
        if sub <= 3:
            return E("Destroy all creatures (Blue bounces; B/W/U only - else 2/4 token).",
                     note("if Deep IQ is B/W/U: destroy (U: bounce) all your creatures; otherwise a 2/4 token"))
        if sub == 4:
            return E("Sacrifice all artifacts (Red), else put a 2/4 token (+3).",
                     note("if Deep IQ is Red: destroy all your artifacts; otherwise a 2/4 token"))
        return E("Put a 2/4 token on the battlefield (+3).", token(2, 4, 3))
    return E("Do nothing.", nothing())


def _branch_to_entry(st: GameState, sub: int) -> dict:
    # helper for special_V9 1-5: rebuild branch and resolve by active colour
    b = branch(
        ("B", kill("sacrifice", "creature")),
        ("R", E("4 damage to your best target creature.", dmg_creature(4))),
        ("W", kill("arrest", "creature")),
        ("U", kill("bounce", "creature")),
        ("G", E("Deep IQ Fights your best target creature.", target("fight", "creature"))),
        ("Artifact", E("Sacrifice your best target creature. Deep IQ drops to Table IV.",
                       target("sacrifice", "creature"), drop(4))),
        ("Colourless", E("Sacrifice your best target creature. Deep IQ loses 1 life.",
                         target("sacrifice", "creature"), diq_life(-1))),
    )
    active = _active_colour(st)
    for cols, entry in b["_branch"]:
        if active in cols:
            return entry
    return b["_branch"][0][1]


def take_diq_turn(st: GameState) -> List[str]:
    """Run one full Deep IQ turn. Returns a structured turn log (lines)."""
    st.turn += 1
    log: List[str] = [f"=== Deep IQ turn {st.turn} (Table {TABLE_ROMAN[st.table]}, "
                      f"{st.identity_label}) ==="]
    # untap
    for tk in st.diq_board:
        tk.tapped = False
    # upkeep roll(s): the main roll plus any queued free rolls
    st.free_rolls = []
    rolls_to_do = [("main", st.table)]
    # main roll
    queue = list(rolls_to_do)
    while queue:
        kind, table = queue.pop(0)
        if st.do_nothing_next and kind == "main":
            st.do_nothing_next = False
            log.append(f"  Roll skipped - treated as 'Do nothing' (pending effect).")
            entry = E("Do nothing.", nothing())
        else:
            mod = st.roll_bonus - (st.next_minus if kind == "main" else 0)
            r = roll(10, mod)
            if kind == "main":
                st.next_minus = 0
            log.append(f"  {'Upkeep' if kind == 'main' else 'Free'} roll on Table "
                       f"{TABLE_ROMAN[table]}: {r}"
                       + (f" (incl. {mod:+d})" if mod else "") + ".")
            entry = _lookup(st, table, r, log)
        if entry is not None:
            _resolve_entry(st, entry, log)
        # fold any newly-queued free rolls in
        for fr in st.free_rolls:
            queue.append(("free", fr))
        st.free_rolls = []

    # attack step (heuristic; the player adjudicates blocks)
    attackers = [t for t in st.diq_board if t.kind == "creature" and t.power > 0]
    if attackers:
        total = sum(t.power for t in attackers)
        log.append(f"  Deep IQ has {len(attackers)} creature(s) (total power {total}). "
                   f"It attacks when it makes sense - declare blocks; unblocked damage hits your "
                   f"life (now {st.player_life}).")
    # advancement
    amax = ADVANCE_MAX[st.table]
    if amax > 0:
        ar = roll(10)
        if ar <= amax:
            st.table += 1
            log.append(f"  Advancement roll {ar} (<={amax}) -> Deep IQ advances to Table {TABLE_ROMAN[st.table]}.")
        else:
            log.append(f"  Advancement roll {ar} (>{amax}) -> Deep IQ stays on Table {TABLE_ROMAN[st.table]}.")
    else:
        log.append("  Table VI: no advancement (Deep IQ stays here).")

    # lethal check
    if st.player_life <= 0:
        log.append("  [X] Your life is 0 or less - Deep IQ wins!")
    if st.diq_life <= 0:
        log.append("  [WIN] Deep IQ's life is 0 or less - you win!")
    return log


# ===========================================================================
# Player board mutation
# ===========================================================================
def add_permanents(st: GameState, perms: List[dict]) -> List[str]:
    """Add permanents the player reports playing. Each: {name,kind,power,toughness}."""
    added = []
    for p in perms:
        ab = p.get("abilities") or []
        if isinstance(ab, str):
            ab = [ab]
        perm = Permanent(name=str(p.get("name", "permanent"))[:40],
                         kind=str(p.get("kind", "creature")),
                         power=int(p.get("power", 0) or 0),
                         toughness=int(p.get("toughness", 0) or 0),
                         abilities=[str(a) for a in ab][:6])
        st.player_board.append(perm)
        added.append(perm.label())
    return added


def remove_permanent(st: GameState, match: str) -> Optional[str]:
    """Remove the first player permanent whose label/name contains `match` (case-insensitive)."""
    m = (match or "").lower().strip()
    for i, p in enumerate(st.player_board):
        if m and (m in p.name.lower() or m in p.label().lower()):
            return st.player_board.pop(i).label()
    return None


# ===========================================================================
# Rendering
# ===========================================================================
def render_panel(st: GameState) -> str:
    if not st.started:
        return "No Deep IQ game in progress."
    lines = ["```",
             f"DEEP IQ - Table {TABLE_ROMAN[st.table]}   |   {st.identity_label}   |   turn {st.turn}",
             f"Life - Deep IQ: {st.diq_life}   You: {st.player_life}",
             "Deep IQ's battlefield:"]
    if st.diq_board:
        for tk in st.diq_board:
            lines.append(f"  * {tk.label()}")
    else:
        lines.append("  (empty)")
    lines.append("Your battlefield (as reported):")
    if st.player_board:
        for p in st.player_board:
            lines.append(f"  * {p.label()}" + ("  [tapped]" if p.tapped else ""))
    else:
        lines.append("  (none reported)")
    mods = []
    if st.roll_bonus:        mods.append(f"+{st.roll_bonus} to all rolls")
    if st.next_minus:        mods.append(f"-{st.next_minus} next roll")
    if st.do_nothing_next:   mods.append("next roll = Do nothing")
    if mods:
        lines.append("Modifiers: " + "; ".join(mods))
    lines.append("```")
    return "\n".join(lines)


# ===========================================================================
# STEPPED / INTERACTIVE turn (priority windows + a blocking step).
# plan_turn() rolls the whole turn and applies Deep IQ's own bookkeeping immediately, but
# returns the player-facing ACTIONS undone so the game master can give you priority to respond
# to each one before it resolves. Combat is then resolved by the engine from your declared blocks.
# ===========================================================================
# Which effect ops are mechanical (apply now) vs interactive (defer for player priority).
_MECHANICAL = {"nothing", "move", "drop", "diq_life", "free_roll", "do_nothing_next",
               "roll_bonus", "next_minus", "note", "retoken_all"}
_INTERACTIVE = {"token", "diq_token", "target", "dmg_creature", "player_life"}


def _apply_mechanical(st: GameState, eff, prelude: List[str]):
    op = eff[0]
    if op == "nothing":
        pass
    elif op == "move":
        if eff[1] > st.table:
            st.table = eff[1]; prelude.append(f"  -> Deep IQ moves up to Table {TABLE_ROMAN[st.table]}.")
    elif op == "drop":
        st.table = eff[1]; prelude.append(f"  -> Deep IQ drops back to Table {TABLE_ROMAN[st.table]}.")
    elif op == "diq_life":
        st.diq_life += eff[1]; prelude.append(f"  -> Deep IQ life {eff[1]:+d} -> {st.diq_life}.")
    elif op == "free_roll":
        st.free_rolls.append(eff[1])
    elif op == "do_nothing_next":
        st.do_nothing_next = True; prelude.append("  -> Deep IQ's next roll will be 'Do nothing'.")
    elif op == "roll_bonus":
        st.roll_bonus += eff[1]; prelude.append(f"  -> Deep IQ now has {st.roll_bonus:+d} to all rolls.")
    elif op == "next_minus":
        st.next_minus += eff[1]; prelude.append(f"  -> Deep IQ has -{eff[1]} on its next roll.")
    elif op == "note":
        prelude.append(f"  -> {eff[1]} (resolve with your cards).")
    elif op == "retoken_all":
        for tk in st.diq_board:
            if tk.kind == "creature":
                extra = token_chart(_active_colour(st), roll(10))
                tk.abilities += [a for a in extra if a not in tk.abilities]
        prelude.append("  -> all Deep IQ tokens gained a permanent extra Token Chart roll.")


def _action_text(st: GameState, eff) -> (str, Optional[str]):
    """Player-facing description of an interactive action + the chosen target label (if any)."""
    op = eff[0]
    if op == "token":
        return (f"put a {eff[1]}/{eff[2]} creature token onto the battlefield", None)
    if op == "diq_token":
        return (f"play a {eff[1]} token ({eff[2]})", None)
    if op == "target":
        tgt = _best_player_target(st, eff[2])
        label = tgt.label() if tgt else None
        verb = {"sacrifice": "make you sacrifice", "destroy": "destroy", "bounce": "bounce",
                "exile": "exile", "arrest": "Arrest", "fight": "fight"}.get(eff[1], eff[1])
        if eff[3] == 2:
            return (f"{verb} your two best {eff[2]}s", "TWO")
        return (f"{verb} your best {eff[2]}" + (f" ({label})" if label else ""), label)
    if op == "dmg_creature":
        tgt = _best_player_target(st, "creature")
        label = tgt.label() if tgt else None
        return (f"deal {eff[1]} damage to your best creature" + (f" ({label})" if label else ""), label)
    if op == "player_life":
        return (f"deal {-eff[1]} damage to YOU", None)
    return ("do something", None)


def _plan_effects(st: GameState, effects, prelude: List[str], actions: List[dict]):
    for eff in effects:
        op = eff[0]
        if op == "spooky":
            r = max(1, min(14, roll(10, eff[1] + st.roll_bonus)))
            entry = SPOOKY[r]
            prelude.append(f"  -> SPOOKY CHART roll {r}: {entry['text']}")
            _plan_effects(st, entry["effects"], prelude, actions)
        elif op in _MECHANICAL:
            _apply_mechanical(st, eff, prelude)
        elif op in _INTERACTIVE:
            text, target = _action_text(st, eff)
            actions.append({"op": eff, "kind": op, "text": text, "target": target})


def plan_turn(st: GameState) -> dict:
    """Roll Deep IQ's whole turn; apply its own bookkeeping; return {'prelude', 'actions'} where
    actions are the player-facing things still to resolve (each gets a priority window)."""
    st.turn += 1
    for tk in st.diq_board:
        tk.tapped = False
    prelude = [f"=== Deep IQ turn {st.turn} (Table {TABLE_ROMAN[st.table]}, {st.identity_label}) ==="]
    actions: List[dict] = []
    st.free_rolls = []
    queue = [("main", st.table)]
    while queue:
        kind, table = queue.pop(0)
        if st.do_nothing_next and kind == "main":
            st.do_nothing_next = False
            prelude.append("  Upkeep roll skipped - treated as 'Do nothing'.")
            entry = E("Do nothing.", nothing())
        else:
            mod = st.roll_bonus - (st.next_minus if kind == "main" else 0)
            r = roll(10, mod)
            if kind == "main":
                st.next_minus = 0
            prelude.append(f"  {'Upkeep' if kind == 'main' else 'Free'} roll on Table "
                           f"{TABLE_ROMAN[table]}: {r}" + (f" (incl. {mod:+d})" if mod else "") + ".")
            entry = _lookup(st, table, r, prelude)
        if entry is not None:
            prelude.append("  " + entry["text"])
            _plan_effects(st, entry["effects"], prelude, actions)
        for fr in st.free_rolls:
            queue.append(("free", fr))
        st.free_rolls = []
    return {"prelude": prelude, "actions": actions}


def resolve_step(st: GameState, action: dict, resolved: bool = True) -> str:
    """Apply one interactive action once the player has passed/responded. resolved=False = it was
    countered/prevented, so it does nothing."""
    if not resolved:
        return f"  -> Countered/prevented: Deep IQ doesn't {action['text']}."
    op = action["op"]; k = action["kind"]
    if k == "token":
        return _create_token(st, op[1], op[2], op[3])
    if k == "diq_token":
        st.diq_board.append(Token(power=0, toughness=0, kind=op[1], note=op[2]))
        return f"  -> Deep IQ's {op[1]} token resolves: {op[2]}."
    if k == "target":
        if action.get("target") == "TWO":
            return "  -> resolve it on your two best (your cards)."
        if action.get("target"):
            gone = remove_permanent(st, action["target"])
            return (f"  -> {action['target']} is gone." if gone
                    else f"  -> (no matching permanent to remove; you resolve it.)")
        return "  -> you had no valid target; it fizzles."
    if k == "dmg_creature":
        return f"  -> {op[1]} damage dealt (you confirm if it dies and report it)."
    if k == "player_life":
        st.player_life += op[1]
        return f"  -> You take {-op[1]} damage -> your life is now {st.player_life}."
    return "  -> resolved."


# ---- combat (engine-computed from declared blocks) ----
def combat_attackers(st: GameState) -> List[Token]:
    """Deep IQ's creatures that can attack (untapped, power > 0, not defenders)."""
    return [t for t in st.diq_board if t.kind == "creature" and t.power > 0 and not t.tapped
            and not _has(t.abilities, "defender", "can't attack", "cannot attack")]


def _has(abils, *keys) -> bool:
    s = " ".join(a.lower() for a in abils)
    return any(k in s for k in keys)


def apply_combat(st: GameState, blocks: dict) -> List[str]:
    """Resolve combat. `blocks` = {attacker_index(0-based): [player_board_index,...]}. Engine does
    the math (lethal, deathtouch, trample, flying legality, rough first strike), updates both boards
    + your life. Unlisted attackers are unblocked."""
    attackers = combat_attackers(st)
    lines: List[str] = []
    player_dmg = 0
    dead_diq: set = set()
    dead_player: set = set()
    for ai, atk in enumerate(attackers):
        a_ab = atk.abilities
        a_fly = _has(a_ab, "flying"); a_fs = _has(a_ab, "first strike", "double strike")
        a_dt = _has(a_ab, "deathtouch"); a_tr = _has(a_ab, "trample")
        idxs = [i for i in blocks.get(ai, []) if 0 <= i < len(st.player_board)
                and st.player_board[i].kind == "creature"]
        blockers = [(i, st.player_board[i]) for i in idxs]
        # flying legality
        legal = [(i, b) for (i, b) in blockers if not a_fly or _has(b.abilities, "flying", "reach")]
        illegal = [b for (i, b) in blockers if (i, b) not in legal]
        for b in illegal:
            lines.append(f"  (your {b.label()} can't block the flying {atk.power}/{atk.toughness}.)")
        blockers = legal
        if not blockers:
            player_dmg += atk.power
            lines.append(f"  {atk.power}/{atk.toughness} attacker is unblocked -> {atk.power} to you.")
            continue
        # attacker assigns its power across blockers (in order); deathtouch -> 1 is lethal
        dmg_left = atk.power
        killed_blockers = []
        for (i, b) in blockers:
            need = 1 if a_dt else b.toughness
            if dmg_left >= need:
                dead_player.add(i); killed_blockers.append((i, b)); dmg_left -= need
        # trample: only damage BEYOND lethal-to-all-blockers spills to the player (none if it
        # can't even get through them).
        if a_tr:
            needed = sum(1 if a_dt else b.toughness for (i, b) in blockers)
            spill = max(0, atk.power - needed)
            if spill > 0:
                player_dmg += spill
                lines.append(f"  trample: {spill} spills over to you.")
        # blockers deal back. with first strike, blockers killed before they swing deal nothing.
        if a_fs:
            back = sum(b.power for (i, b) in blockers if i not in dead_player)
            back_dt = _has([x for (i, b) in blockers if i not in dead_player for x in b.abilities], "deathtouch")
        else:
            back = sum(b.power for (i, b) in blockers)
            back_dt = any(_has(b.abilities, "deathtouch") for (i, b) in blockers)
        if back >= atk.toughness or (back_dt and back > 0):
            dead_diq.add(ai)
            lines.append(f"  your blockers kill Deep IQ's {atk.power}/{atk.toughness}.")
        else:
            kb = ", ".join(b.label() for (i, b) in killed_blockers)
            lines.append(f"  Deep IQ's {atk.power}/{atk.toughness} survives"
                         + (f"; it kills your {kb}." if kb else " the block."))
    # apply
    st.player_life -= player_dmg
    for i in sorted(dead_player, reverse=True):
        if 0 <= i < len(st.player_board):
            st.player_board.pop(i)
    for ai in sorted(dead_diq, reverse=True):
        tok = attackers[ai]
        if tok in st.diq_board:
            st.diq_board.remove(tok)
    lines.append(f"  Combat damage to you this turn: {player_dmg} -> your life is {st.player_life}.")
    if st.player_life <= 0:
        lines.append("  [X] You're at 0 or less - Deep IQ wins!")
    return lines


def do_advancement(st: GameState) -> List[str]:
    """End-of-turn advancement roll (Deep IQ may climb a table)."""
    amax = ADVANCE_MAX[st.table]
    if amax <= 0:
        return ["  Table VI: no advancement (Deep IQ stays here)."]
    ar = roll(10)
    if ar <= amax:
        st.table += 1
        return [f"  Advancement roll {ar} (<={amax}) -> Deep IQ advances to Table {TABLE_ROMAN[st.table]}."]
    return [f"  Advancement roll {ar} (>{amax}) -> Deep IQ stays on Table {TABLE_ROMAN[st.table]}."]
