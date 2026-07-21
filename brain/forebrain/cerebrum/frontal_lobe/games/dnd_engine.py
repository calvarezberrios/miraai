"""
dnd_engine.py — the TRUTH for Mira-as-a-D&D-player: real dice and a real character sheet.

Mirrors deep_iq_engine's division of labour: this module owns everything numeric — ability
scores, modifiers, proficiency, HP/AC, spell slots, and every die that gets rolled. The LLM
NEVER invents numbers; it narrates what this engine reports. 5e-lite (SRD basics): enough
rules to play a real session as a party member — checks, saves, attacks, cantrips/1st-level
spells, initiative, damage/healing — without modeling every subsystem.

A human DM runs the game; this engine is only HER character.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ABILITIES = ("STR", "DEX", "CON", "INT", "WIS", "CHA")

# skill -> governing ability (5e SRD)
SKILLS = {
    "acrobatics": "DEX", "animal handling": "WIS", "arcana": "INT", "athletics": "STR",
    "deception": "CHA", "history": "INT", "insight": "WIS", "intimidation": "CHA",
    "investigation": "INT", "medicine": "WIS", "nature": "INT", "perception": "WIS",
    "performance": "CHA", "persuasion": "CHA", "religion": "INT", "sleight of hand": "DEX",
    "stealth": "DEX", "survival": "WIS",
}

RACES = {
    "human":    {"bonus": {a: 1 for a in ABILITIES}, "speed": 30},
    "elf":      {"bonus": {"DEX": 2, "WIS": 1}, "speed": 30},
    "halfling": {"bonus": {"DEX": 2, "CHA": 1}, "speed": 25},
    "dwarf":    {"bonus": {"CON": 2, "WIS": 1}, "speed": 25},
    "tiefling": {"bonus": {"CHA": 2, "INT": 1}, "speed": 30},
    "gnome":    {"bonus": {"INT": 2, "DEX": 1}, "speed": 25},
    "half-elf": {"bonus": {"CHA": 2, "DEX": 1, "WIS": 1}, "speed": 30},
}

# class -> mechanics. stat_priority drives auto-assignment of the rolled array.
# weapon: (name, damage_expr, ability, ranged). AC formulas are starting-gear approximations.
CLASSES = {
    "wizard": {
        "hit_die": 6, "stat_priority": ["INT", "DEX", "CON", "WIS", "CHA", "STR"],
        "saves": ("INT", "WIS"), "skills": ("arcana", "investigation", "insight"),
        "ac": lambda dex: 10 + dex,   # unarmored (she can cast mage armor for 13+DEX)
        "weapon": ("fire bolt", "1d10", "INT", True),
        "caster": True, "cast_ability": "INT",
    },
    "cleric": {
        "hit_die": 8, "stat_priority": ["WIS", "CON", "STR", "DEX", "CHA", "INT"],
        "saves": ("WIS", "CHA"), "skills": ("medicine", "insight", "religion"),
        "ac": lambda dex: 18,          # chain mail + shield
        "weapon": ("mace", "1d6", "STR", False),
        "caster": True, "cast_ability": "WIS",
    },
    "rogue": {
        "hit_die": 8, "stat_priority": ["DEX", "CON", "WIS", "INT", "CHA", "STR"],
        "saves": ("DEX", "INT"), "skills": ("stealth", "sleight of hand", "perception", "acrobatics"),
        "ac": lambda dex: 11 + dex,    # leather
        "weapon": ("rapier", "1d8", "DEX", False),
        "caster": False,
    },
    "fighter": {
        "hit_die": 10, "stat_priority": ["STR", "CON", "DEX", "WIS", "CHA", "INT"],
        "saves": ("STR", "CON"), "skills": ("athletics", "perception", "intimidation"),
        "ac": lambda dex: 16,          # chain mail
        "weapon": ("longsword", "1d8", "STR", False),
        "caster": False,
    },
    "ranger": {
        "hit_die": 10, "stat_priority": ["DEX", "WIS", "CON", "STR", "INT", "CHA"],
        "saves": ("STR", "DEX"), "skills": ("survival", "perception", "stealth", "nature"),
        "ac": lambda dex: 12 + min(dex, 2),   # studded leather-ish (medium cap)
        "weapon": ("longbow", "1d8", "DEX", True),
        "caster": True, "cast_ability": "WIS",
    },
    "bard": {
        "hit_die": 8, "stat_priority": ["CHA", "DEX", "CON", "WIS", "INT", "STR"],
        "saves": ("DEX", "CHA"), "skills": ("performance", "persuasion", "deception", "insight"),
        "ac": lambda dex: 11 + dex,    # leather
        "weapon": ("rapier", "1d8", "DEX", False),
        "caster": True, "cast_ability": "CHA",
    },
}

# spell -> (level, kind, expr, note). kind: "attack" (spell attack roll), "save" (target saves,
# ability in note), "auto" (no roll), "heal". Cantrips are level 0 (never consume a slot).
SPELLS = {
    "wizard": {
        "fire bolt":     (0, "attack", "1d10", "fire"),
        "ray of frost":  (0, "attack", "1d8", "cold, target speed -10ft"),
        "magic missile": (1, "auto",   "3*(1d4+1)", "3 darts, auto-hit force"),
        "burning hands": (1, "save",   "3d6", "DEX save, fire, 15ft cone, half on save"),
        "shield":        (1, "auto",   "", "reaction: +5 AC until next turn"),
        "sleep":         (1, "auto",   "5d8", "HP of creatures affected, lowest first"),
    },
    "cleric": {
        "sacred flame":  (0, "save",   "1d8", "DEX save, radiant, no cover"),
        "guiding bolt":  (1, "attack", "4d6", "radiant; next attack on target has advantage"),
        "cure wounds":   (1, "heal",   "1d8", "touch"),
        "healing word":  (1, "heal",   "1d4", "60ft, bonus action"),
        "bless":         (1, "auto",   "", "3 allies +1d4 on attacks and saves"),
    },
    "bard": {
        "vicious mockery": (0, "save", "1d4", "WIS save, psychic, disadvantage on next attack"),
        "healing word":    (1, "heal", "1d4", "60ft, bonus action"),
        "faerie fire":     (1, "auto", "", "DEX save or outlined: attacks vs target have advantage"),
        "thunderwave":     (1, "save", "2d8", "CON save, thunder, 15ft cube, pushed 10ft"),
    },
    "ranger": {
        "hunter's mark": (1, "auto", "", "+1d6 on weapon hits vs marked target"),
        "cure wounds":   (1, "heal", "1d8", "touch"),
    },
}

# 1st-level spell slots by character level (5e full/half-caster-lite; rangers get fewer)
def _slots_for(clazz: str, level: int) -> int:
    if not CLASSES[clazz].get("caster"):
        return 0
    if clazz == "ranger":
        return 0 if level < 2 else (2 if level < 5 else 3)
    return {1: 2, 2: 3}.get(level, 4)


def prof_bonus(level: int) -> int:
    return 2 + (max(1, level) - 1) // 4


def mod(score: int) -> int:
    return (score - 10) // 2


def fmt_mod(m: int) -> str:
    return f"+{m}" if m >= 0 else str(m)


# ---------------------------------------------------------------------------
# Dice — the only dice in the room. Every roll returns (total, detail_string).
# ---------------------------------------------------------------------------
_DICE_RE = re.compile(r"^\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$", re.I)
_MULT_RE = re.compile(r"^\s*(\d+)\*\((.+)\)\s*$")   # e.g. 3*(1d4+1) for magic missile


def roll_die(sides: int) -> int:
    return random.randint(1, max(2, sides))


def roll_expr(expr: str) -> Tuple[int, str]:
    """Roll '2d6+3' / 'd20' / '3*(1d4+1)'. Returns (total, '2d6(4,5)+3=12')."""
    expr = (expr or "").strip().lower()
    m = _MULT_RE.match(expr)
    if m:
        times, inner = int(m.group(1)), m.group(2)
        total, parts = 0, []
        for _ in range(times):
            t, d = roll_expr(inner)
            total += t
            parts.append(str(t))
        return total, f"{times}x({inner}) = {'+'.join(parts)} = {total}"
    m = _DICE_RE.match(expr)
    if not m:
        raise ValueError(f"bad dice expression: {expr!r}")
    n = int(m.group(1) or 1)
    sides = int(m.group(2))
    bonus = int((m.group(3) or "0").replace(" ", ""))
    rolls = [roll_die(sides) for _ in range(min(n, 40))]
    total = sum(rolls) + bonus
    detail = f"{n}d{sides}({','.join(map(str, rolls))})" + (fmt_mod(bonus) if bonus else "") + f" = {total}"
    return total, detail


def d20(modifier: int = 0, adv: int = 0) -> Tuple[int, str, bool, bool]:
    """One d20 roll with a modifier. adv: +1 advantage, -1 disadvantage, 0 straight.
    Returns (total, detail, nat20, nat1)."""
    r1, r2 = roll_die(20), roll_die(20)
    if adv > 0:
        used, tag = max(r1, r2), f"adv d20({r1},{r2})->{max(r1, r2)}"
    elif adv < 0:
        used, tag = min(r1, r2), f"dis d20({r1},{r2})->{min(r1, r2)}"
    else:
        used, tag = r1, f"d20({r1})"
    total = used + modifier
    return total, f"{tag} {fmt_mod(modifier)} = {total}", used == 20, used == 1


def roll_stat_array() -> List[int]:
    """Six ability scores, 4d6 drop lowest, sorted high->low."""
    out = []
    for _ in range(6):
        rolls = sorted(roll_die(6) for _ in range(4))
        out.append(sum(rolls[1:]))
    return sorted(out, reverse=True)


# Spell casting times (5e SRD common cases). Everything else defaults to an Action.
BONUS_ACTION_SPELLS = {"healing word", "misty step", "spiritual weapon", "hunter's mark",
                       "hex", "shield of faith", "sanctuary", "expeditious retreat",
                       "healing spirit", "sanctuary"}
REACTION_SPELLS = {"shield", "feather fall", "counterspell", "hellish rebuke", "absorb elements"}


def cast_time(spell: str) -> str:
    s = (spell or "").lower()
    if s in BONUS_ACTION_SPELLS:
        return "bonus"
    if s in REACTION_SPELLS:
        return "reaction"
    return "action"


# Every SRD spell we have mechanics for, indexed by name (across all classes) so an imported
# D&D Beyond character can roll a spell we recognize regardless of class.
SPELL_INDEX = {}
for _cls, _sp in SPELLS.items():
    for _n, _d in _sp.items():
        SPELL_INDEX.setdefault(_n, _d)

# A real starting weapon per class (the class "weapon" field is a cast-attack for casters, which
# isn't a weapon — casters instead get a mundane sidearm, with their cantrips in the spell list).
CLASS_DEFAULT_WEAPON = {
    "wizard":  {"name": "dagger",     "dice": "1d4", "ability": "DEX", "finesse": True,  "ranged": False, "type": "piercing"},
    "cleric":  {"name": "mace",       "dice": "1d6", "ability": "STR", "finesse": False, "ranged": False, "type": "bludgeoning"},
    "rogue":   {"name": "rapier",     "dice": "1d8", "ability": "DEX", "finesse": True,  "ranged": False, "type": "piercing"},
    "fighter": {"name": "longsword",  "dice": "1d8", "ability": "STR", "finesse": False, "ranged": False, "type": "slashing"},
    "ranger":  {"name": "longbow",    "dice": "1d8", "ability": "DEX", "finesse": False, "ranged": True,  "type": "piercing"},
    "bard":    {"name": "rapier",     "dice": "1d8", "ability": "DEX", "finesse": True,  "ranged": False, "type": "piercing"},
}

# The 5e Action options that don't need a listed weapon/spell (for turn_options()).
BASIC_ACTIONS = ("Dash", "Disengage", "Dodge", "Help", "Hide", "Ready", "Search", "Use an Object")


# ---------------------------------------------------------------------------
# The character sheet
# ---------------------------------------------------------------------------
@dataclass
class Character:
    name: str = "Melisande"
    race: str = "human"
    clazz: str = "wizard"
    level: int = 3
    scores: dict = field(default_factory=dict)          # ability -> score (racial bonus applied)
    hp: int = 0
    max_hp: int = 0
    slots: int = 0                                       # 1st-level slots remaining
    max_slots: int = 0
    unconscious: bool = False
    log: List[str] = field(default_factory=list)         # recent engine results (rolling window)

    # sheet data (defaults set by build_character; overwritten by a D&D Beyond import)
    speed: int = 30
    weapons: List[dict] = field(default_factory=list)    # each: name/dice/ability/prof/finesse/ranged/bonus/type
    spell_list: List[dict] = field(default_factory=list) # each: {"name": str, "level": int}
    inventory: List[str] = field(default_factory=list)
    prof_skills: set = field(default_factory=set)
    prof_saves: set = field(default_factory=set)
    expertise: set = field(default_factory=set)
    ac_override: int = 0                                  # 0 = use class formula
    spell_dc_override: int = 0
    spell_atk_override: int = 0
    class_name: str = ""                                 # display class (DDB may be non-SRD)
    source: str = "generated"                            # "generated" | "dndbeyond"
    ddb_id: str = ""

    # action economy (one combat turn)
    in_turn: bool = False
    action_used: bool = False
    bonus_used: bool = False
    reaction_used: bool = False
    move_left: int = 0

    # ---- derived ----
    def m(self, ability: str) -> int:
        return mod(self.scores.get(ability, 10))

    @property
    def prof(self) -> int:
        return prof_bonus(self.level)

    @property
    def ac(self) -> int:
        return self.ac_override or CLASSES[self.clazz]["ac"](self.m("DEX"))

    @property
    def _cast_ability(self):
        return CLASSES[self.clazz].get("cast_ability")

    @property
    def cast_mod(self) -> int:
        if self.spell_atk_override:
            return self.spell_atk_override
        ab = self._cast_ability
        return (self.m(ab) + self.prof) if ab else 0

    @property
    def spell_dc(self) -> int:
        if self.spell_dc_override:
            return self.spell_dc_override
        ab = self._cast_ability
        return (8 + self.m(ab) + self.prof) if ab else 0

    @property
    def caster(self) -> bool:
        return bool(self.spell_list) or bool(CLASSES[self.clazz].get("caster"))

    def skill_mod(self, skill: str) -> int:
        base = self.m(SKILLS[skill])
        if skill in self.expertise:
            return base + 2 * self.prof
        if skill in self.prof_skills:
            return base + self.prof
        return base

    def save_mod(self, ability: str) -> int:
        return self.m(ability) + (self.prof if ability in self.prof_saves else 0)

    def known_spells(self) -> set:
        return {s["name"] for s in self.spell_list}

    def weapon_by_name(self, name: str):
        if not name:
            return self.weapons[0] if self.weapons else None
        name = name.strip().lower()
        for w in self.weapons:
            if name == w["name"].lower() or name in w["name"].lower() or w["name"].lower() in name:
                return w
        return None

    def _wep_ability(self, w: dict) -> str:
        """Which ability a weapon uses: ranged -> DEX; finesse -> the better of STR/DEX."""
        if w.get("ranged"):
            return "DEX"
        if w.get("finesse"):
            return "DEX" if self.m("DEX") >= self.m("STR") else "STR"
        return w.get("ability", "STR")

    # ---- engine actions (each returns a human-readable result line, also logged) ----
    def _log(self, line: str) -> str:
        self.log.append(line)
        del self.log[:-8]
        return line

    def _spend(self, kind: str) -> str:
        """Mark an action-economy resource used during a turn; note if over-spent. Off turn = ''."""
        if not self.in_turn:
            return ""
        used = getattr(self, f"{kind}_used")
        setattr(self, f"{kind}_used", True)
        return f"  [!! {kind} already used this turn]" if used else f"  [{kind} used]"

    def check(self, skill: str, adv: int = 0) -> str:
        total, detail, n20, n1 = d20(self.skill_mod(skill), adv)
        extra = " — NATURAL 20!" if n20 else (" — natural 1..." if n1 else "")
        return self._log(f"{skill.title()} check: {detail}{extra}")

    def ability_check(self, ability: str, adv: int = 0) -> str:
        total, detail, n20, n1 = d20(self.m(ability), adv)
        return self._log(f"{ability} check: {detail}" + (" — NATURAL 20!" if n20 else (" — natural 1..." if n1 else "")))

    def save(self, ability: str, adv: int = 0) -> str:
        total, detail, n20, n1 = d20(self.save_mod(ability), adv)
        return self._log(f"{ability} save: {detail}" + (" — NATURAL 20!" if n20 else (" — natural 1..." if n1 else "")))

    def initiative(self) -> str:
        total, detail, _, _ = d20(self.m("DEX"))
        return self._log(f"Initiative: {detail}")

    def attack(self, adv: int = 0) -> str:
        return self.attack_with(None, adv)

    def attack_with(self, name: Optional[str] = None, adv: int = 0) -> str:
        w = self.weapon_by_name(name)
        if w is None:
            have = ", ".join(x["name"] for x in self.weapons) or "no weapons on the sheet"
            return self._log(f"No weapon matching '{name}'. Weapons: {have}.")
        ab = self._wep_ability(w)
        atk_mod = self.m(ab) + (self.prof if w.get("prof", True) else 0) + w.get("bonus", 0)
        total, detail, n20, n1 = d20(atk_mod, adv)
        dmg_mod = self.m(ab) + w.get("bonus", 0)
        expr = w["dice"] + (fmt_mod(dmg_mod) if dmg_mod else "")
        dmg_total, dmg_detail = roll_expr(expr)
        if n20:
            crit_total, crit_detail = roll_expr(w["dice"])
            dmg_total += crit_total
            dmg_detail += f" + CRIT {crit_detail}"
        dtype = f" {w['type']}" if w.get("type") else ""
        line = (f"Attack ({w['name']}, {ab}): to hit {detail}"
                + (" — NATURAL 20, CRIT!" if n20 else (" — natural 1, miss." if n1 else ""))
                + ((" | damage if it hits: " + (f"{dmg_detail} -> total {dmg_total}" if n20 else dmg_detail)
                    + dtype) if not n1 else ""))
        return self._log(line + self._spend("action"))

    def cast(self, spell: str, adv: int = 0) -> str:
        spell = spell.strip().lower()
        entry = next((s for s in self.spell_list if s["name"] == spell), None)
        if entry is None:
            entry = next((s for s in self.spell_list
                          if spell in s["name"] or s["name"] in spell), None)
        if entry is None:
            known = ", ".join(sorted(self.known_spells())) or "none"
            return self._log(f"Cast failed: {self.name} doesn't have '{spell}'. Known: {known}.")
        name, lvl = entry["name"], int(entry.get("level", 0))
        ct = cast_time(name)
        econ = self._spend("bonus" if ct == "bonus" else ("reaction" if ct == "reaction" else "action"))
        if lvl > 0:
            if self.slots <= 0:
                return self._log(f"Cast failed: no 1st-level slots left for {name}.")
            self.slots -= 1
        slot_note = "" if lvl == 0 else f" [slot used; {self.slots}/{self.max_slots} left]"
        data = SPELL_INDEX.get(name)
        if not data:
            # We don't have SRD mechanics for this spell (common for imported spells) — announce
            # the cast + slot honestly and let the DM/players resolve its effect. No made-up dice.
            return self._log(f"Casts {name} (level {lvl}, {ct} to cast). Effect resolved by the "
                             f"DM/players.{slot_note}{econ}")
        _lvl, kind, expr, note = data
        if kind == "attack":
            total, detail, n20, n1 = d20(self.cast_mod, adv)
            dmg_total, dmg_detail = roll_expr(expr)
            if n20:
                crit, crit_d = roll_expr(expr)
                dmg_total += crit
                dmg_detail += f" + CRIT {crit_d}"
            line = (f"Cast {name}: spell attack {detail}"
                    + (" — NATURAL 20, CRIT!" if n20 else (" — natural 1, miss." if n1 else ""))
                    + ((" | damage if it hits: " + (f"{dmg_detail} -> total {dmg_total}" if n20 else dmg_detail)
                        + f" ({note})") if not n1 else ""))
        elif kind == "save":
            _t, dmg_detail = roll_expr(expr) if expr else (0, "")
            line = f"Cast {name}: target makes a DC {self.spell_dc} save ({note}); damage {dmg_detail}"
        elif kind == "heal":
            heal_total, heal_detail = roll_expr(expr)
            add = (self.m(self._cast_ability) if self._cast_ability else 0)
            heal_total += add
            line = f"Cast {name}: heals {heal_detail} {fmt_mod(add)} = {heal_total} HP ({note})"
        else:
            line = f"Cast {name}: {note}" + (f" (roll: {roll_expr(expr)[1]})" if expr else "")
        return self._log(line + slot_note + econ)

    # ---- action economy (a combat turn) ----
    def start_turn(self) -> str:
        self.in_turn = True
        self.action_used = self.bonus_used = self.reaction_used = False
        self.move_left = self.speed
        return self._log(f"Turn started — Action, Bonus Action, and Reaction available; "
                         f"{self.speed} ft of movement.")

    def end_turn(self) -> str:
        self.in_turn = False
        return self._log("Turn ended.")

    def spend_move(self, feet: int) -> str:
        feet = max(0, int(feet))
        over = ""
        self.move_left -= feet
        if self.move_left < 0:
            over = f" — that's {-self.move_left} ft more than you had left!"
            self.move_left = 0
        return self._log(f"Moved {feet} ft.{over} Movement left: {self.move_left}/{self.speed} ft.")

    def use_action(self, what: str) -> str:
        note = self._spend("action")
        return self._log(f"Action: {what}.{note}")

    def use_reaction(self, what: str = "a reaction") -> str:
        if self.reaction_used:
            return self._log(f"Reaction already used — none left until your next turn.")
        self.reaction_used = True
        return self._log(f"Reaction used: {what}. (refreshes at the start of your next turn)")

    def turn_options(self) -> str:
        """What she can still do right now — the answer to 'what are my options?'."""
        lines = []
        if not self.in_turn:
            lines.append("(Not in a combat turn — say 'start my turn' to begin one.)")
        weps = " / ".join(w["name"] for w in self.weapons) or "unarmed strike"
        act_spells = [s["name"] for s in self.spell_list if cast_time(s["name"]) == "action"]
        bonus_spells = [s["name"] for s in self.spell_list if cast_time(s["name"]) == "bonus"]
        react_spells = [s["name"] for s in self.spell_list if cast_time(s["name"]) == "reaction"]
        if self.action_used:
            lines.append("ACTION: used.")
        else:
            opts = [f"Attack ({weps})"]
            if act_spells:
                opts.append("Cast a spell (" + ", ".join(act_spells) + ")")
            opts += list(BASIC_ACTIONS)
            lines.append("ACTION available — " + "; ".join(opts) + ".")
        if self.bonus_used:
            lines.append("BONUS ACTION: used.")
        elif bonus_spells:
            lines.append("BONUS ACTION available — Cast: " + ", ".join(bonus_spells) + ".")
        else:
            lines.append("BONUS ACTION available — nothing on your sheet uses it right now.")
        if self.reaction_used:
            lines.append("REACTION: used.")
        else:
            r = "REACTION available — opportunity attack"
            if react_spells:
                r += ", or cast " + ", ".join(react_spells)
            lines.append(r + ".")
        lines.append(f"MOVEMENT: {self.move_left}/{self.speed} ft left.  (free: one object interaction, speaking)")
        return "\n".join(lines)

    def take_damage(self, n: int) -> str:
        self.hp = max(0, self.hp - max(0, n))
        if self.hp == 0:
            self.unconscious = True
            return self._log(f"Took {n} damage -> 0/{self.max_hp} HP. She is DOWN (unconscious).")
        return self._log(f"Took {n} damage -> {self.hp}/{self.max_hp} HP.")

    def heal(self, n: int) -> str:
        self.hp = min(self.max_hp, self.hp + max(0, n))
        was_down = self.unconscious
        if self.hp > 0:
            self.unconscious = False
        up = " She's back on her feet!" if was_down and self.hp > 0 else ""
        return self._log(f"Healed {n} -> {self.hp}/{self.max_hp} HP.{up}")

    def long_rest(self) -> str:
        self.hp = self.max_hp
        self.slots = self.max_slots
        self.unconscious = False
        return self._log(f"Long rest: back to {self.hp}/{self.max_hp} HP, {self.slots} spell slots.")

    # ---- presentation ----
    def _spells_by_level(self) -> str:
        cants = sorted(s["name"] for s in self.spell_list if s.get("level", 0) == 0)
        leveled = sorted(s["name"] for s in self.spell_list if s.get("level", 0) > 0)
        parts = []
        if cants:
            parts.append("cantrips: " + ", ".join(cants))
        if leveled:
            parts.append("spells: " + ", ".join(leveled))
        return "   ".join(parts)

    def sheet_panel(self) -> str:
        stats = "  ".join(f"{a} {self.scores.get(a, 10)} ({fmt_mod(self.m(a))})" for a in ABILITIES)
        saves = ", ".join(sorted(self.prof_saves)) or "none"
        skills = ", ".join(sorted(s.title() for s in self.prof_skills)) or "none"
        src = "  [D&D Beyond]" if self.source == "dndbeyond" else ""
        cls = self.class_name or self.clazz
        lines = [
            "=" * 52,
            f" {self.name} — level {self.level} {self.race} {cls}{src}",
            "=" * 52,
            f" HP {self.hp}/{self.max_hp}   AC {self.ac}   speed {self.speed}ft   "
            f"prof {fmt_mod(self.prof)}   init {fmt_mod(self.m('DEX'))}"
            + ("   DOWN" if self.unconscious else ""),
            f" {stats}",
            f" Save profs: {saves}",
            f" Skill profs: {skills}",
        ]
        for w in self.weapons:
            ab = self._wep_ability(w)
            atk = self.m(ab) + (self.prof if w.get("prof", True) else 0) + w.get("bonus", 0)
            dmg_mod = self.m(ab) + w.get("bonus", 0)
            lines.append(f" Weapon: {w['name']}  {fmt_mod(atk)} to hit, "
                         f"{w['dice']}{fmt_mod(dmg_mod) if dmg_mod else ''} {w.get('type','')}".rstrip())
        if self.caster:
            lines.append(f" Spell save DC {self.spell_dc}, attack {fmt_mod(self.cast_mod)}   "
                         f"1st-lvl slots {self.slots}/{self.max_slots}")
            sb = self._spells_by_level()
            if sb:
                lines.append(" " + sb)
        if self.inventory:
            lines.append(" Items: " + ", ".join(self.inventory[:24]))
        lines.append("=" * 52)
        return "\n".join(lines)

    def short_summary(self) -> str:
        return (f"{self.name}, a level {self.level} {self.race} {self.class_name or self.clazz} "
                f"(HP {self.hp}/{self.max_hp}, AC {self.ac}"
                + (", UNCONSCIOUS" if self.unconscious else "") + ")")

    # ---- persistence ----
    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "race": self.race, "clazz": self.clazz, "level": self.level,
            "scores": self.scores, "hp": self.hp, "max_hp": self.max_hp,
            "slots": self.slots, "max_slots": self.max_slots, "unconscious": self.unconscious,
            "speed": self.speed, "weapons": self.weapons, "spell_list": self.spell_list,
            "inventory": self.inventory, "prof_skills": sorted(self.prof_skills),
            "prof_saves": sorted(self.prof_saves), "expertise": sorted(self.expertise),
            "ac_override": self.ac_override, "spell_dc_override": self.spell_dc_override,
            "spell_atk_override": self.spell_atk_override, "class_name": self.class_name,
            "source": self.source, "ddb_id": self.ddb_id,
        }, indent=2)

    @staticmethod
    def from_json(raw: str) -> "Character":
        d = json.loads(raw)
        c = Character(
            name=d["name"], race=d["race"], clazz=d["clazz"], level=d["level"],
            scores=d["scores"], hp=d["hp"], max_hp=d["max_hp"],
            slots=d["slots"], max_slots=d["max_slots"], unconscious=d.get("unconscious", False),
            speed=d.get("speed", 30), weapons=d.get("weapons", []), spell_list=d.get("spell_list", []),
            inventory=d.get("inventory", []), prof_skills=set(d.get("prof_skills", [])),
            prof_saves=set(d.get("prof_saves", [])), expertise=set(d.get("expertise", [])),
            ac_override=d.get("ac_override", 0), spell_dc_override=d.get("spell_dc_override", 0),
            spell_atk_override=d.get("spell_atk_override", 0), class_name=d.get("class_name", ""),
            source=d.get("source", "generated"), ddb_id=d.get("ddb_id", ""))
        return c


STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]     # 5e PHB standard array


def _finalize_hp_slots(c: "Character") -> None:
    """Set HP (max die at 1st, average+CON per level after) and 1st-level slots from
    class + level + CON. Called after scores are finalized."""
    die = CLASSES[c.clazz]["hit_die"]
    con = c.m("CON")
    c.max_hp = max(1, die + con + sum((die // 2 + 1) + con for _ in range(c.level - 1)))
    c.hp = c.max_hp
    c.max_slots = _slots_for(c.clazz, c.level)
    c.slots = c.max_slots


def build_character(clazz: str, race: str = "human", name: str = "Adventurer", level: int = 1,
                    scores: Optional[dict] = None, array: Optional[list] = None) -> Tuple[Character, str]:
    """Build a character with EXPLICIT control (for step-by-step creation):
      - scores: a full {ABILITY: base_score} dict the user assigned -> used as-is (pre-race).
      - array : six unassigned numbers -> assigned to abilities by the class's stat priority.
      - neither: roll 4d6-drop-lowest and assign by priority.
    Racial bonuses are then applied. Returns (char, report)."""
    clazz = clazz if clazz in CLASSES else "wizard"
    race = race if race in RACES else "human"
    level = max(1, min(20, int(level)))
    if scores and all(a in scores for a in ABILITIES):
        base = {a: int(scores[a]) for a in ABILITIES}
        method = "your assigned scores"
    else:
        if not array:
            array = roll_stat_array()
            method = f"rolled (4d6 drop lowest): {array}"
        else:
            method = f"assigned array {list(array)}"
        base = {}
        for ability, score in zip(CLASSES[clazz]["stat_priority"], array):
            base[ability] = score
    for ability, bonus in RACES[race]["bonus"].items():
        base[ability] = base.get(ability, 10) + bonus
    c = Character(name=name, race=race, clazz=clazz, level=level, scores=base)
    # sheet defaults for a generated character
    c.speed = RACES[race]["speed"]
    w = dict(CLASS_DEFAULT_WEAPON[clazz]); w.setdefault("prof", True); w.setdefault("bonus", 0)
    c.weapons = [w]
    c.spell_list = [{"name": n, "level": d[0]} for n, d in SPELLS.get(clazz, {}).items()]
    c.prof_skills = set(CLASSES[clazz]["skills"])
    c.prof_saves = set(CLASSES[clazz]["saves"])
    c.inventory = [w["name"], "explorer's pack", "50 ft rope"]
    _finalize_hp_slots(c)
    report = (f"Stats {method}, {race} bonuses applied. "
              f"HP {c.max_hp}, AC {c.ac}" + (f", {c.max_slots} 1st-level slots" if c.max_slots else ""))
    return c, report


def create_character(clazz: str = "wizard", race: str = "elf", name: str = "Melisande",
                     level: int = 3) -> Tuple[Character, str]:
    """Quick roll (legacy convenience): 4d6-drop-lowest, priority-assigned."""
    return build_character(clazz=clazz, race=race, name=name, level=level)
