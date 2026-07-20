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

    # ---- derived ----
    def m(self, ability: str) -> int:
        return mod(self.scores.get(ability, 10))

    @property
    def prof(self) -> int:
        return prof_bonus(self.level)

    @property
    def ac(self) -> int:
        return CLASSES[self.clazz]["ac"](self.m("DEX"))

    @property
    def cast_mod(self) -> int:
        ab = CLASSES[self.clazz].get("cast_ability")
        return (self.m(ab) + self.prof) if ab else 0

    @property
    def spell_dc(self) -> int:
        ab = CLASSES[self.clazz].get("cast_ability")
        return (8 + self.m(ab) + self.prof) if ab else 0

    def skill_mod(self, skill: str) -> int:
        base = self.m(SKILLS[skill])
        return base + (self.prof if skill in CLASSES[self.clazz]["skills"] else 0)

    def save_mod(self, ability: str) -> int:
        return self.m(ability) + (self.prof if ability in CLASSES[self.clazz]["saves"] else 0)

    def spells(self) -> dict:
        return SPELLS.get(self.clazz, {})

    # ---- engine actions (each returns a human-readable result line, also logged) ----
    def _log(self, line: str) -> str:
        self.log.append(line)
        del self.log[:-8]
        return line

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
        wname, dmg_expr, ability, ranged = CLASSES[self.clazz]["weapon"]
        atk_mod = self.m(ability) + self.prof
        total, detail, n20, n1 = d20(atk_mod, adv)
        dmg_bonus = self.m(ability) if wname not in ("fire bolt",) else 0
        expr = dmg_expr + (f"+{dmg_bonus}" if dmg_bonus > 0 else (str(dmg_bonus) if dmg_bonus < 0 else ""))
        dmg_total, dmg_detail = roll_expr(expr)
        if n20:
            crit_total, crit_detail = roll_expr(dmg_expr)   # crit: roll the dice again
            dmg_total += crit_total
            dmg_detail += f" + CRIT {crit_detail}"
        line = (f"Attack ({wname}): to hit {detail}"
                + (" — NATURAL 20, CRIT!" if n20 else (" — natural 1, miss." if n1 else ""))
                + ((" | damage if it hits: " + (f"{dmg_detail} -> total {dmg_total}" if n20 else dmg_detail))
                   if not n1 else ""))
        return self._log(line)

    def cast(self, spell: str, adv: int = 0) -> str:
        spell = spell.strip().lower()
        book = self.spells()
        if spell not in book:
            known = ", ".join(sorted(book)) or "none"
            return self._log(f"Cast failed: she doesn't know '{spell}'. Known: {known}.")
        lvl, kind, expr, note = book[spell]
        if lvl > 0:
            if self.slots <= 0:
                return self._log(f"Cast failed: no 1st-level slots left for {spell}.")
            self.slots -= 1
        slot_note = "" if lvl == 0 else f" [slot used; {self.slots}/{self.max_slots} left]"
        if kind == "attack":
            total, detail, n20, n1 = d20(self.cast_mod, adv)
            dmg_total, dmg_detail = roll_expr(expr)
            if n20:
                crit, crit_d = roll_expr(expr)
                dmg_total += crit
                dmg_detail += f" + CRIT {crit_d}"
            line = (f"Cast {spell}: spell attack {detail}"
                    + (" — NATURAL 20, CRIT!" if n20 else (" — natural 1, miss." if n1 else ""))
                    + ((" | damage if it hits: " + (f"{dmg_detail} -> total {dmg_total}" if n20 else dmg_detail)
                        + f" ({note})") if not n1 else ""))
        elif kind == "save":
            dmg_total, dmg_detail = roll_expr(expr) if expr else (0, "")
            line = f"Cast {spell}: target makes a DC {self.spell_dc} save ({note}); damage {dmg_detail}"
        elif kind == "heal":
            heal_total, heal_detail = roll_expr(expr)
            heal_total += self.cast_mod - self.prof   # + casting ability mod
            line = f"Cast {spell}: heals {heal_detail} {fmt_mod(self.cast_mod - self.prof)} = {heal_total} HP ({note})"
        else:
            line = f"Cast {spell}: {note}" + (f" (roll: {roll_expr(expr)[1]})" if expr else "")
        return self._log(line + slot_note)

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
    def sheet_panel(self) -> str:
        cls = CLASSES[self.clazz]
        stats = "  ".join(f"{a} {self.scores[a]} ({fmt_mod(self.m(a))})" for a in ABILITIES)
        profs = ", ".join(s.title() for s in cls["skills"])
        lines = [
            "=" * 46,
            f" {self.name} — level {self.level} {self.race} {self.clazz}",
            "=" * 46,
            f" HP {self.hp}/{self.max_hp}   AC {self.ac}   prof {fmt_mod(self.prof)}   "
            f"init {fmt_mod(self.m('DEX'))}" + ("   DOWN" if self.unconscious else ""),
            f" {stats}",
            f" Saves: {cls['saves'][0]}/{cls['saves'][1]}   Skills: {profs}",
            f" Weapon: {cls['weapon'][0]} ({cls['weapon'][1]})",
        ]
        if cls.get("caster"):
            lines.append(f" Spell DC {self.spell_dc}, attack {fmt_mod(self.cast_mod)}   "
                         f"slots {self.slots}/{self.max_slots}   "
                         f"spells: {', '.join(sorted(self.spells()))}")
        lines.append("=" * 46)
        return "\n".join(lines)

    def short_summary(self) -> str:
        return (f"{self.name}, a level {self.level} {self.race} {self.clazz} "
                f"(HP {self.hp}/{self.max_hp}, AC {self.ac}"
                + (", UNCONSCIOUS" if self.unconscious else "") + ")")

    # ---- persistence ----
    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "race": self.race, "clazz": self.clazz, "level": self.level,
            "scores": self.scores, "hp": self.hp, "max_hp": self.max_hp,
            "slots": self.slots, "max_slots": self.max_slots, "unconscious": self.unconscious,
        }, indent=2)

    @staticmethod
    def from_json(raw: str) -> "Character":
        d = json.loads(raw)
        c = Character(name=d["name"], race=d["race"], clazz=d["clazz"], level=d["level"],
                      scores=d["scores"], hp=d["hp"], max_hp=d["max_hp"],
                      slots=d["slots"], max_slots=d["max_slots"],
                      unconscious=d.get("unconscious", False))
        return c


def create_character(clazz: str = "wizard", race: str = "elf", name: str = "Melisande",
                     level: int = 3) -> Tuple[Character, str]:
    """Roll a fresh character: 4d6-drop-lowest stats auto-assigned by class priority,
    racial bonuses, HP = max die + avg per further level + CON. Returns (char, roll_report)."""
    clazz = clazz if clazz in CLASSES else "wizard"
    race = race if race in RACES else "human"
    array = roll_stat_array()
    scores = {}
    for ability, score in zip(CLASSES[clazz]["stat_priority"], array):
        scores[ability] = score
    for ability, bonus in RACES[race]["bonus"].items():
        scores[ability] = scores.get(ability, 10) + bonus
    c = Character(name=name, race=race, clazz=clazz, level=max(1, level), scores=scores)
    die = CLASSES[clazz]["hit_die"]
    con = c.m("CON")
    c.max_hp = die + con + sum((die // 2 + 1) + con for _ in range(c.level - 1))
    c.max_hp = max(1, c.max_hp)
    c.hp = c.max_hp
    c.max_slots = _slots_for(clazz, c.level)
    c.slots = c.max_slots
    report = (f"Rolled stats (4d6 drop lowest): {array} -> assigned for a {clazz}. "
              f"HP {c.max_hp}, AC {c.ac}" + (f", {c.max_slots} 1st-level slots" if c.max_slots else ""))
    return c, report
