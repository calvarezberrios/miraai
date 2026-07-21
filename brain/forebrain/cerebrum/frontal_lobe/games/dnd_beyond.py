"""
dnd_beyond.py — import a REAL character sheet from D&D Beyond (read-only).

DDB has no public WRITE API (you can't create or update a sheet from code), but a character set
to PUBLIC exposes its full sheet as JSON from the character service — the same source Avrae and
Beyond20 read. We fetch that and map it onto our engine's Character so her dice come straight from
the real sheet: ability scores, proficiencies, HP/AC, equipped weapons (with each weapon's own
to-hit + damage), and her spell list.

Endpoint:  https://character-service.dndbeyond.com/character/v5/character/{id}
Use:       parse_character_id("https://www.dndbeyond.com/characters/12345678") -> "12345678"
           import_character(url_or_id) -> (Character, notes)   (raises with a clear message on failure)

This is a best-effort mapping of a large, layered schema; import_character prints the resulting
sheet so you can compare it to your DDB page. Engine limitation: only 1st-level spell slots are
tracked (5e-lite), so higher-level slots are approximated.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import List, Optional, Tuple

from . import dnd_engine as eng

DDB_URL = "https://character-service.dndbeyond.com/character/v5/character/{id}"

# stat id (1..6) -> ability
_STAT_AB = {1: "STR", 2: "DEX", 3: "CON", 4: "INT", 5: "WIS", 6: "CHA"}
# DDB spells out ability names in modifier subTypes ("dexterity-score", "wisdom-saving-throws")
_AB_FULL = {"STR": "strength", "DEX": "dexterity", "CON": "constitution",
            "INT": "intelligence", "WIS": "wisdom", "CHA": "charisma"}
# DDB class name -> our SRD mechanics key (display keeps the real name via class_name)
_CLASS_MAP = {
    "barbarian": "fighter", "fighter": "fighter", "paladin": "fighter", "monk": "fighter",
    "rogue": "rogue", "ranger": "ranger", "wizard": "wizard", "artificer": "wizard",
    "cleric": "cleric", "druid": "cleric", "bard": "bard", "sorcerer": "bard", "warlock": "bard",
}
_FULL_CASTERS = {"bard", "cleric", "druid", "sorcerer", "wizard", "warlock"}
_HALF_CASTERS = {"paladin", "ranger"}


def parse_character_id(url_or_id: str) -> Optional[str]:
    s = (url_or_id or "").strip()
    m = re.search(r"/characters/(\d+)", s) or re.search(r"(?:character/)?(\d{5,})", s)
    return m.group(1) if m else None


def _fetch(cid: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(DDB_URL.format(id=cid),
                                 headers={"User-Agent": "Mira-DnD/1.0", "Accept": "application/json"})
    with opener.open(req, timeout=15) as r:
        raw = json.loads(r.read().decode("utf-8"))
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], dict):
        if raw.get("success") is False:
            raise RuntimeError(raw.get("message") or "DDB reported failure")
        return raw["data"]
    return raw


def _all_modifiers(data: dict) -> list:
    mods = data.get("modifiers") or {}
    out = []
    for cat in ("race", "class", "background", "item", "feat", "condition"):
        out.extend(mods.get(cat) or [])
    return out


def _score_bonus(mods: list, ability: str) -> int:
    """Sum 'bonus' modifiers to an ability score (racial/feat/item ASIs). DDB subTypes use the
    full ability name, e.g. 'dexterity-score'."""
    total = 0
    want = _AB_FULL[ability] + "-score"
    for m in mods:
        if (m.get("type") == "bonus" and (m.get("subType") == want)
                and isinstance(m.get("value"), (int, float))):
            total += int(m["value"])
    return total


def _ac_bonus(mods: list) -> int:
    total = 0
    for m in mods:
        if m.get("type") == "bonus" and m.get("subType") in ("armor-class", "unarmored-armor-class") \
                and isinstance(m.get("value"), (int, float)):
            total += int(m["value"])
    return total


def _first_level_slots(caster_level: int) -> int:
    return {0: 0, 1: 2, 2: 3}.get(caster_level, 4 if caster_level >= 3 else 0)


def to_character(data: dict, ddb_id: str = "") -> Tuple[eng.Character, List[str]]:
    notes: List[str] = []
    mods = _all_modifiers(data)

    # ----- classes / level -----
    classes = data.get("classes") or []
    total_level = sum(int(c.get("level", 0)) for c in classes) or 1
    primary = None
    for c in classes:
        if c.get("isStartingClass"):
            primary = c
    if primary is None and classes:
        primary = max(classes, key=lambda c: int(c.get("level", 0)))
    ddb_class = ((primary or {}).get("definition") or {}).get("name", "Fighter")
    class_name = ", ".join(
        f"{(c.get('definition') or {}).get('name','?')} {c.get('level','?')}" for c in classes) or ddb_class
    clazz = _CLASS_MAP.get(ddb_class.lower(), "fighter")

    # ----- ability scores: base + bonusStats + score-bonus modifiers, overrideStats wins -----
    base = {}
    for st in (data.get("stats") or []):
        ab = _STAT_AB.get(st.get("id"))
        if ab:
            base[ab] = int(st.get("value") or 10)
    for st in (data.get("bonusStats") or []):
        ab = _STAT_AB.get(st.get("id"))
        if ab and isinstance(st.get("value"), (int, float)):
            base[ab] = base.get(ab, 10) + int(st["value"])
    for ab in eng.ABILITIES:
        base[ab] = base.get(ab, 10) + _score_bonus(mods, ab)
    for st in (data.get("overrideStats") or []):
        ab = _STAT_AB.get(st.get("id"))
        if ab and isinstance(st.get("value"), (int, float)) and st["value"]:
            base[ab] = int(st["value"])

    c = eng.Character(name=data.get("name", "Adventurer").strip() or "Adventurer",
                      race=((data.get("race") or {}).get("fullName")
                            or (data.get("race") or {}).get("baseName") or "human"),
                      clazz=clazz, class_name=class_name, level=total_level, scores=base,
                      source="dndbeyond", ddb_id=str(ddb_id))
    con_mod = c.m("CON")

    # ----- HP -----
    base_hp = int(data.get("baseHitPoints") or 0)
    bonus_hp = int(data.get("bonusHitPoints") or 0)
    override_hp = data.get("overrideHitPoints")
    removed = int(data.get("removedHitPoints") or 0)
    if override_hp:
        c.max_hp = int(override_hp)
    else:
        c.max_hp = max(1, base_hp + con_mod * total_level + bonus_hp)
    c.hp = max(0, c.max_hp - removed)
    if c.hp == 0:
        c.unconscious = True

    # ----- speed -----
    try:
        c.speed = int(((data.get("race") or {}).get("weightSpeeds") or {}).get("normal", {}).get("walk") or 30)
    except Exception:
        c.speed = 30

    # ----- proficiencies (skills / saves / expertise) -----
    for m in mods:
        t, sub = m.get("type"), (m.get("subType") or "")
        if t in ("proficiency", "expertise"):
            skill = sub.replace("-", " ")
            if skill in eng.SKILLS:
                c.prof_skills.add(skill)
                if t == "expertise":
                    c.expertise.add(skill)
            m2 = re.match(r"(strength|dexterity|constitution|intelligence|wisdom|charisma)-saving-throws", sub)
            if m2:
                c.prof_saves.add(m2.group(1)[:3].upper() if False else
                                 {"strength": "STR", "dexterity": "DEX", "constitution": "CON",
                                  "intelligence": "INT", "wisdom": "WIS", "charisma": "CHA"}[m2.group(1)])

    # ----- AC (best effort from equipped armor + shield + bonuses) -----
    dex = c.m("DEX")
    base_armor = None
    armor_type = None
    shield = 0
    for it in (data.get("inventory") or []):
        d = it.get("definition") or {}
        if not it.get("equipped"):
            continue
        acv = d.get("armorClass")
        at = d.get("armorTypeId")
        if acv and at:
            if at == 4:                       # shield
                shield += int(acv)
            else:
                base_armor, armor_type = int(acv), at
    if base_armor is not None:
        if armor_type == 1:      # light: full DEX
            ac = base_armor + dex
        elif armor_type == 2:    # medium: DEX up to +2
            ac = base_armor + min(dex, 2)
        else:                    # heavy: no DEX
            ac = base_armor
    else:
        ac = 10 + dex            # unarmored
    ac += shield + _ac_bonus(mods)
    c.ac_override = ac

    # ----- weapons (from inventory) -----
    weapons = []
    for it in (data.get("inventory") or []):
        d = it.get("definition") or {}
        if d.get("filterType") != "Weapon" and not d.get("damage"):
            continue
        dmg = (d.get("damage") or {}).get("diceString")
        if not dmg:
            continue
        props = [(p or {}).get("name", "") for p in (d.get("properties") or [])]
        name = d.get("name", "weapon")
        mbonus = 0
        mm = re.match(r"^\s*\+(\d+)\b", name)
        if mm:
            mbonus = int(mm.group(1))
        weapons.append({
            "name": name,
            "dice": dmg,
            "ability": "STR",
            "finesse": ("Finesse" in props),
            "ranged": (d.get("attackType") == 2 or "Ammunition" in props),
            "prof": True,
            "bonus": mbonus,
            "type": (d.get("damageType") or ""),
        })
    # equipped first, keep it tidy
    c.weapons = weapons[:8]
    if not c.weapons:
        notes.append("No weapons found on the sheet — added an unarmed strike fallback.")
        c.weapons = [{"name": "unarmed strike", "dice": "1d1", "ability": "STR",
                      "finesse": False, "ranged": False, "prof": True, "bonus": 0, "type": "bludgeoning"}]

    # ----- inventory (item names) -----
    inv = []
    for it in (data.get("inventory") or []):
        d = it.get("definition") or {}
        nm = d.get("name")
        if nm:
            q = it.get("quantity") or 1
            inv.append(f"{nm}" + (f" x{q}" if q and q > 1 else ""))
    c.inventory = inv

    # ----- spells + spellcasting stat -----
    spells = {}
    def _add_spell(sp):
        d = sp.get("definition") or {}
        nm = (d.get("name") or "").strip().lower()
        if nm:
            spells[nm] = int(d.get("level") or 0)
    for cs in (data.get("classSpells") or []):
        for sp in (cs.get("spells") or []):
            _add_spell(sp)
    spdict = data.get("spells") or {}
    for cat in ("class", "race", "feat", "item", "background"):
        for sp in (spdict.get(cat) or []):
            _add_spell(sp)
    c.spell_list = [{"name": n, "level": lv} for n, lv in sorted(spells.items())]

    # spellcasting ability from the primary caster class -> DC/attack overrides
    cast_ab_id = None
    caster_level = 0
    for cl in classes:
        cdef = cl.get("definition") or {}
        nm = (cdef.get("name") or "").lower()
        lvl = int(cl.get("level", 0))
        if nm in _FULL_CASTERS:
            caster_level += lvl
        elif nm in _HALF_CASTERS:
            caster_level += lvl // 2
        elif nm == "artificer":
            caster_level += (lvl + 1) // 2
        if cast_ab_id is None and cdef.get("spellCastingAbilityId"):
            cast_ab_id = cdef.get("spellCastingAbilityId")
    if c.spell_list and cast_ab_id in _STAT_AB:
        ab = _STAT_AB[cast_ab_id]
        c.spell_atk_override = c.m(ab) + c.prof
        c.spell_dc_override = 8 + c.m(ab) + c.prof
    c.max_slots = _first_level_slots(caster_level if caster_level else (total_level if c.spell_list else 0))
    c.slots = c.max_slots
    if c.spell_list:
        notes.append("Only 1st-level spell slots are tracked (engine limitation); higher-level "
                     "slots aren't modeled.")

    return c, notes


def import_character(url_or_id: str) -> Tuple[eng.Character, List[str]]:
    cid = parse_character_id(url_or_id)
    if not cid:
        raise ValueError("Couldn't find a character id in that — give the D&D Beyond character URL "
                         "(https://www.dndbeyond.com/characters/NUMBERS) or just the number.")
    try:
        data = _fetch(cid)
    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            raise RuntimeError("D&D Beyond refused the request — set the character's privacy to "
                               "PUBLIC (character sheet -> gear icon -> Character Privacy -> Public), "
                               "then try again.")
        raise RuntimeError(f"D&D Beyond returned HTTP {e.code}.")
    except Exception as e:
        raise RuntimeError(f"Couldn't reach D&D Beyond ({e}).")
    return to_character(data, ddb_id=cid)
