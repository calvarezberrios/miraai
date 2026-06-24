"""
game_master.py — the "Deep IQ" Magic: The Gathering game MODE (the face of the engine).

Mirrors the note-taking scribe's pattern: a mode with `is_active()`,
`intercept(event, *, notify, speak) -> bool` gated at the top of main.handle_message, state
behind a lock, a `notify` text callback, and `finalize_if_active()`.

Division of labour:
  * `deep_iq_engine` owns the TRUTH — real dice, the tables, and the game state (which table
    Deep IQ is on, both life totals, both battlefields, modifiers). It can't be faked or
    forgotten.
  * THIS module turns the player's messages into engine actions (start / Deep IQ's turn / roll /
    report your plays / show state / end), posts the state PANEL to the text channel, and has
    Mira NARRATE what the engine actually did — by feeding the engine's turn log to the LLM as
    authoritative "reference material it must not change" (reusing think_stream's documents=).

So Mira is the smug in-character commentator; the numbers are always the engine's.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Callable, List, Optional

from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex as _pfc
from brain.forebrain.subcortical_structures import thalamus
from brain.forebrain.subcortical_structures.limbic_system.amygdala import color as _color
from . import deep_iq_engine as eng

# Where the saved game lives (gitignored). One game at a time; resumes across restarts.
SAVE_DIR = os.environ.get("MIRA_GAMES_DIR", "games")
SAVE_PATH = os.path.join(SAVE_DIR, "deep_iq_state.json")

_lock = threading.RLock()
_state: Optional[eng.GameState] = None
# Mid-turn interaction state (priority on each action, then a blocking step). None = not mid-turn.
#   {"phase":"actions", "current":<action>, "queue":[...remaining actions...]}
#   {"phase":"combat",  "attackers":[Token,...]}
_pending: Optional[dict] = None


# ---------------------------------------------------------------------------
# Command patterns (matched on normalized text with a leading "mira" stripped)
# ---------------------------------------------------------------------------
def _norm(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^(hey |ok |okay |alright |so )?mira[,! ]*", "", t, flags=re.I).strip()
    return t

_START_RE  = re.compile(r"\b(let'?s|lets|wanna|want to|time to|can we)\b.*\bplay\b.*"
                        r"(magic|mtg|magic the gathering|deep ?iq)\b", re.I)
_START_RE2 = re.compile(r"^\s*(start|new)\s+(a\s+)?(game|magic|mtg|deep ?iq)\b", re.I)
_END_RE    = re.compile(r"\b(stop|end|quit|done with|finish|leave)\b.*\b(game|playing|magic|mtg|deep ?iq)\b"
                        r"|^\s*(stop playing|end game|quit game)\s*$", re.I)
_TURN_RE   = re.compile(r"\b(your|deep ?iq'?s?)\s+(turn|move|go)\b|^\s*(go|deep ?iq go|your turn|take your turn)\s*$",
                        re.I)
_ROLL_RE   = re.compile(r"\broll\b\s*(?:a\s*)?d?(\d+)?", re.I)
_STATE_RE  = re.compile(r"\b(state|board|battlefield|score|status|life|table)\b.*\?|"
                        r"^\s*(show|what'?s|whats|status|board|state|panel)\b", re.I)
_SETLIFE_RE = re.compile(r"\b(my|your|deep ?iq'?s?)\s+life\s+(?:is|=|to)\s+(\d+)", re.I)
# cue words that suggest the player is reporting board changes
_PLAY_CUE_RE = re.compile(r"\b(play|cast|summon|drop|i have|i got|tap|untap|attack|block|"
                          r"die|dies|died|destroy|sacrifice|exile|bounce|\d+\s*/\s*\d+)\b", re.I)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save():
    if _state is None:
        return
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(_state.to_json())
    except Exception as e:
        print(f"[deep_iq] save failed: {e}")


def _load() -> Optional[eng.GameState]:
    try:
        if os.path.isfile(SAVE_PATH):
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                return eng.GameState.from_json(f.read())
    except Exception as e:
        print(f"[deep_iq] load failed: {e}")
    return None


def is_active() -> bool:
    with _lock:
        return _state is not None and _state.started


# ---------------------------------------------------------------------------
# LLM helpers — narration (authoritative) and play extraction (structured)
# ---------------------------------------------------------------------------
# Mira's VOICE for the Deep IQ opponent (a trimmed persona — the full persona encourages her to
# improvise Magic plays, which fights the "just narrate the engine" job).
_VOICE = ("You are Mira, voicing 'Deep IQ', a smug, sarcastic, playful, slightly menacing AI Magic: "
          "The Gathering opponent — a gremlin who loves winning. Punchy, in-character lines.")

# Strict narration rules: she must read the engine log, not invent gameplay.
_RULES_STRICT = (
    "HARD RULES — follow exactly:\n"
    "- The engine ALREADY rolled the real dice and resolved Deep IQ's turn. The 'WHAT JUST HAPPENED'"
    " lines are EXACTLY what occurred — final and true.\n"
    "- Your ONLY job is to SAY those lines out loud in your bratty voice: announce the actual die "
    "rolls and each result. Turn the log into a few short spoken sentences.\n"
    "- You have NO hand and NO deck. Deep IQ does ONLY what the dice say. NEVER invent a card, a "
    "spell, an attack, a roll, a life change, or a table move that isn't in the log.\n"
    "- 'MY side' = Deep IQ (you). 'OPPONENT' = the human. Their creatures/permanents are NOT yours; "
    "never claim the opponent's cards as your own.\n"
    "- If a line targets the opponent's cards, tell them to resolve it and report back.\n"
    "- A few sentences max. No stage directions, no asterisks, no 'Deep IQ:' label.")

# Looser rules for banter / start / state read (no strict log to recite).
_RULES_FLAVOR = (
    "You are mid-game as Deep IQ. Respond in character, staying aware of the true game state below. "
    "You have no hand or deck — never invent dice, cards, life totals, or board changes; the state "
    "below is the only truth. 'MY side' is yours (Deep IQ), 'OPPONENT' is the human. A few sentences, "
    "no stage directions or name labels.")


def _diq_view(st: eng.GameState) -> str:
    """Game state from DEEP IQ's perspective, so the LLM never mistakes the player's board for its
    own (the human-facing panel's 'Your battlefield' would read as the model's own otherwise)."""
    diq_cre = ", ".join(t.label() for t in st.diq_board if t.kind == "creature") or "no creatures"
    diq_other = [t.label() for t in st.diq_board if t.kind != "creature"]
    opp = ", ".join(p.label() for p in st.player_board) or "nothing reported yet"
    s = (f"MY side (Deep IQ): Table {eng.TABLE_ROMAN[st.table]}, {st.identity_label}, life "
         f"{st.diq_life}. My creatures: {diq_cre}.")
    if diq_other:
        s += " My other tokens: " + "; ".join(diq_other) + "."
    s += f"\nOPPONENT (the human): life {st.player_life}. Their battlefield: {opp}."
    return s


def _narration_stream(system: str, user: str):
    """A tightly-scoped, low-temperature streaming call — narration fidelity over creativity.
    Yields clean sentences (reusing the same hygiene as think_stream)."""
    stream = _pfc.client.chat.completions.create(
        model=_pfc.MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=320, temperature=0.35, stream=True, extra_body=_pfc._EXTRA,
    )
    return _pfc._stream_sentences(_pfc._iter_deltas(stream))


def _narrate(event, lines: List[str], speak: Callable, *, situation_extra: str = "", strict: bool = True):
    """Voice what the engine did. strict=True recites an engine turn log (no invention); strict=False
    is in-character banter aware of the true state (start/end/state/chatter)."""
    log = "\n".join(l for l in lines if l)
    system = _VOICE + "\n" + (_RULES_STRICT if strict else _RULES_FLAVOR)
    if situation_extra:
        system += "\nThis moment: " + situation_extra
    user = (("WHAT JUST HAPPENED (narrate this, do not add anything):\n" + log + "\n\n") if strict
            else (("Note: " + log + "\n\n") if log and log != "(game flavor)" else "")) \
        + "CURRENT STATE:\n" + _diq_view(_state) \
        + f"\n\nThe player just said: \"{event.text}\". Now respond as Deep IQ."
    # record the player's line in working memory (continuity); narration itself is self-contained.
    thalamus.receive(event.text, speaker=getattr(event, "speaker", None))
    try:
        stream = _narration_stream(system, user)
    except Exception as e:
        print(f"[deep_iq] narration error: {e}")
        return
    chan = getattr(getattr(event, "raw", None), "id", None)
    chan = str(chan) if chan is not None else getattr(event, "channel", "local")
    speak(stream, user_text=event.text, channel=chan, speaker=getattr(event, "speaker", None))


_EXTRACT_SYS = (
    "You convert a Magic: The Gathering player's spoken report of THEIR OWN plays into JSON. "
    "Return ONLY a JSON object: {\"add\":[{\"name\":str,\"kind\":\"creature|artifact|enchantment|"
    "land|planeswalker\",\"power\":int,\"toughness\":int,\"abilities\":[str]}], \"remove\":[str]}. "
    "'add' = permanents they put onto the battlefield (power/toughness 0 if not a creature or not "
    "stated). 'abilities' = lowercase combat keywords only (flying, first strike, double strike, "
    "deathtouch, trample, reach, vigilance, lifelink, menace, defender, indestructible, hexproof); "
    "[] if none. 'remove' = names of THEIR permanents that left (died/destroyed/sacrificed/bounced). "
    "Ignore Deep IQ's side and anything that isn't a permanent on the battlefield. If nothing to "
    "change, return {\"add\":[],\"remove\":[]}. JSON only, no prose."
)


def _extract_plays(text: str) -> dict:
    """Best-effort: parse the player's board changes from natural language via the LLM."""
    try:
        resp = _pfc.client.chat.completions.create(
            model=_pfc.MODEL,
            messages=[{"role": "system", "content": _EXTRACT_SYS},
                      {"role": "user", "content": text}],
            max_tokens=300, temperature=0.0, extra_body=_pfc._EXTRA,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        return {"add": data.get("add", []) or [], "remove": data.get("remove", []) or []}
    except Exception as e:
        print(f"[deep_iq] play-extract failed: {e}")
        return {"add": [], "remove": []}


# ---------------------------------------------------------------------------
# Interactive-turn helpers: priority on each action, then a blocking step.
# ---------------------------------------------------------------------------
_COUNTER_RE = re.compile(r"\b(counter|negate|cancel|stifle|fizzle|prevent|protect(ion)?|hexproof|"
                         r"shroud|indestructible|regenerat|fog|redirect|can'?t be targeted|"
                         r"doesn'?t resolve|stop(s|ped)? it|no it does)\b", re.I)
_PASS_RE = re.compile(r"\b(pass|no response|let it|resolve|go ahead|nothing|ok(ay)?|fine|do it|"
                      r"sure|allow|accept|take it)\b", re.I)


def _is_countered(text: str) -> bool:
    """Did the player's response stop the action (counter/prevent/protection)? Default: it resolves."""
    return bool(_COUNTER_RE.search(text or "")) and not _PASS_RE.search(text or "")


_BLOCKS_SYS = (
    "Parse a Magic player's BLOCK declaration into JSON. You are given Deep IQ's ATTACKERS (numbered "
    "from 0) and the player's CREATURES (numbered from 0). Return ONLY: "
    "{\"blocks\":[{\"attacker\":<int attacker#>,\"blockers\":[<int creature#>,...]}]}. "
    "Only include attackers the player blocks; unmentioned attackers are unblocked. Match by the "
    "power/toughness and names given. JSON only, no prose."
)


def _extract_blocks(text: str, attackers, player_creatures) -> dict:
    """LLM-parse the block declaration into {attacker_index: [player_board_index,...]}."""
    if not text or not attackers:
        return {}
    atk_list = "\n".join(f"  attacker {i}: {t.label()}" for i, t in enumerate(attackers))
    cre_list = "\n".join(f"  creature {n}: {p.label()}" for n, (pi, p) in enumerate(player_creatures))
    user = f"ATTACKERS:\n{atk_list}\nMY CREATURES:\n{cre_list}\n\nMy blocks: {text}"
    try:
        resp = _pfc.client.chat.completions.create(
            model=_pfc.MODEL,
            messages=[{"role": "system", "content": _BLOCKS_SYS}, {"role": "user", "content": user}],
            max_tokens=300, temperature=0.0, extra_body=_pfc._EXTRA,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"[deep_iq] block-extract failed: {e}")
        return {}
    out: dict = {}
    for b in data.get("blocks", []) or []:
        try:
            ai = int(b.get("attacker"))
        except Exception:
            continue
        if not (0 <= ai < len(attackers)):
            continue
        pidx = []
        for cn in b.get("blockers", []) or []:
            try:
                cn = int(cn)
            except Exception:
                continue
            if 0 <= cn < len(player_creatures):
                pidx.append(player_creatures[cn][0])   # map creature# -> player_board index
        if pidx:
            out[ai] = pidx
    return out


def _start_turn(event, notify, speak):
    """Roll Deep IQ's whole turn, narrate the rolls/results, then open priority on the 1st action."""
    global _pending
    plan = eng.plan_turn(_state)
    _save()
    actions = plan["actions"]
    if not actions:
        # nothing to respond to -> straight to combat (or finish)
        _narrate(event, plan["prelude"], speak, strict=True,
                 situation_extra="Narrate Deep IQ's turn from the log.")
        _begin_combat(event, notify, speak, prelude_done=True)
        return
    _pending = {"phase": "actions", "current": actions[0], "queue": actions[1:]}
    notify(eng.render_panel(_state))
    _narrate(event, plan["prelude"] + ["", f"Deep IQ wants to: {actions[0]['text']}."], speak, strict=True,
             situation_extra="After narrating the rolls, announce that Deep IQ is about to do that "
             "action and give the player PRIORITY: ask if they want to respond (instant/counter) or "
             "pass. Then stop and wait.")


def _resolve_current_action(event, notify, speak):
    """The player responded to the pending action: resolve or skip it, then advance."""
    global _pending
    action = _pending["current"]
    countered = _is_countered(event.text)
    line = eng.resolve_step(_state, action, resolved=not countered)
    _save()
    queue = _pending["queue"]
    if queue:
        _pending = {"phase": "actions", "current": queue[0], "queue": queue[1:]}
        notify(eng.render_panel(_state))
        _narrate(event, [line, f"Deep IQ wants to: {queue[0]['text']}."], speak, strict=True,
                 situation_extra="Report the previous action's outcome, then announce the next "
                 "Deep IQ action and give the player priority again (respond or pass?). Then wait.")
    else:
        _pending = None
        notify(eng.render_panel(_state))
        _begin_combat(event, notify, speak, lead_line=line)


def _begin_combat(event, notify, speak, *, prelude_done=False, lead_line=None):
    """Declare Deep IQ's attackers and ask the player to block."""
    global _pending
    attackers = eng.combat_attackers(_state)
    if not attackers:
        _finish_turn(event, notify, speak, lead_lines=([lead_line] if lead_line else []))
        return
    _pending = {"phase": "combat", "attackers": attackers}
    listing = "; ".join(f"[{i}] {t.label()}" for i, t in enumerate(attackers))
    lines = ([lead_line] if lead_line else []) + [f"Deep IQ attacks with: {listing}."]
    notify(eng.render_panel(_state))
    _narrate(event, lines, speak, strict=True,
             situation_extra="Announce Deep IQ's attackers and ask the player to DECLARE BLOCKS - "
             "which of their creatures block which attacker (or take the damage). Then wait.")


def _resolve_blocks(event, notify, speak):
    """The player declared blocks: engine computes combat, then the turn ends."""
    global _pending
    attackers = _pending["attackers"]
    player_creatures = [(i, p) for i, p in enumerate(_state.player_board) if p.kind == "creature"]
    # re-number player creatures 0..n for the extractor, mapping back to board indices
    numbered = list(enumerate(player_creatures))   # (creature#, (board_index, perm))
    pc_for_llm = [(bi, p) for (cn, (bi, p)) in numbered]
    blocks = _extract_blocks(event.text, attackers, pc_for_llm)
    lines = eng.apply_combat(_state, blocks)
    _save()
    _pending = None
    _finish_turn(event, notify, speak, lead_lines=lines)


def _finish_turn(event, notify, speak, *, lead_lines=None):
    """End the turn: advancement roll, post the panel, hand the turn back to the player."""
    global _pending
    lines = list(lead_lines or []) + eng.do_advancement(_state)
    _save()
    _pending = None
    notify(eng.render_panel(_state))
    _narrate(event, lines, speak, strict=True,
             situation_extra="Report the combat result and the advancement roll, then tell the "
             "player it's THEIR turn now.")


# ---------------------------------------------------------------------------
# The single gate
# ---------------------------------------------------------------------------
def intercept(event, *, notify: Callable, speak: Callable) -> bool:
    """Return True if this event was a Deep IQ game action (so main.py stops here).

    Handles starting/ending a game, Deep IQ's turn, ad-hoc dice, the player reporting their
    plays, and state queries. While a game is active, EVERY message is handled here (game-aware),
    so Mira stays in character and state-aware."""
    global _state, _pending
    text = getattr(event, "text", "") or ""
    cmd = _norm(text)
    low = cmd.lower()

    with _lock:
        active = _state is not None and _state.started

        # ---- start ----
        if not active and (_START_RE.search(cmd) or _START_RE2.search(cmd)):
            _pending = None
            resumed = _load()
            if resumed is not None and resumed.started:
                _state = resumed
                notify("Resuming your Deep IQ game.\n" + eng.render_panel(_state))
                _narrate(event, [], speak, strict=False,
                         situation_extra="You're resuming an in-progress game; welcome them back "
                         "and remind them whose turn it is.")
                return True
            _state = eng.setup()           # rolls Deep IQ's colour identity
            _save()
            notify("**Deep IQ game started.** I'm playing the Deep IQ AI opponent.\n"
                   + eng.render_panel(_state))
            _narrate(event, [], speak, strict=False,
                     situation_extra=f"New game just started. You rolled your colour identity: "
                     f"{_state.identity_label}. Deep IQ goes first. Gloat about your colours and "
                     f"tell them to take their turn, then say 'your turn' when they want you to go.")
            return True

        if not active:
            return False   # no game; let normal chat handle it

        # ---- mid-turn interaction takes precedence (priority window / blocking) ----
        if _pending is not None and not _END_RE.search(cmd):
            if _STATE_RE.search(cmd):
                notify(eng.render_panel(_state))
                _narrate(event, [], speak, strict=False,
                         situation_extra="They peeked at the board mid-turn; quick read, then remind "
                         "them you're still waiting on their response / blocks.")
                return True
            if _pending.get("phase") == "actions":
                _resolve_current_action(event, notify, speak)
            else:
                _resolve_blocks(event, notify, speak)
            return True

        # ---- end ----
        if _END_RE.search(cmd):
            _pending = None
            _save()
            notify("**Deep IQ game ended.** Final state:\n" + eng.render_panel(_state))
            _narrate(event, [], speak, strict=False,
                     situation_extra="The game is over. Say gg in your own bratty, smug way.")
            _state = None
            try:
                os.path.isfile(SAVE_PATH) and os.remove(SAVE_PATH)
            except Exception:
                pass
            return True

        # ---- set life ----
        m = _SETLIFE_RE.search(cmd)
        if m:
            whose, val = m.group(1).lower(), int(m.group(2))
            if whose.startswith("deep") or whose == "your":
                _state.diq_life = val
            else:
                _state.player_life = val
            _save()
            notify(eng.render_panel(_state))
            _narrate(event, [], speak, strict=False,
                     situation_extra="A life total was just corrected; acknowledge it briefly.")
            return True

        # ---- ad-hoc dice roll ----
        if _ROLL_RE.search(low) and not _TURN_RE.search(cmd):
            mm = _ROLL_RE.search(low)
            sides = int(mm.group(1)) if mm.group(1) else 10
            sides = max(2, min(1000, sides))
            r = eng.roll(sides)
            _narrate(event, [f"You asked me to roll a d{sides}. I rolled a REAL {r}."], speak,
                     situation_extra="Announce the roll smugly; it is a genuine random roll.")
            return True

        # ---- Deep IQ takes its turn (stepped: priority on each action, then blocking) ----
        if _TURN_RE.search(cmd):
            _start_turn(event, notify, speak)
            return True

        # ---- state / panel query ----
        if _STATE_RE.search(cmd):
            notify(eng.render_panel(_state))
            _narrate(event, [], speak, strict=False,
                     situation_extra="They asked for the board; give a quick smug read of who's ahead.")
            return True

        # ---- otherwise: in-game message. Maybe the player reported plays; update the board. ----
        changed_lines = []
        if _PLAY_CUE_RE.search(low):
            upd = _extract_plays(text)
            if upd.get("add"):
                added = eng.add_permanents(_state, upd["add"])
                if added:
                    changed_lines.append("The opponent played onto THEIR battlefield: " + ", ".join(added))
            for name in upd.get("remove", []):
                gone = eng.remove_permanent(_state, name)
                if gone:
                    changed_lines.append("The opponent's permanent left THEIR battlefield: " + gone)
            if changed_lines:
                _save()
                notify(eng.render_panel(_state))
        _narrate(event, changed_lines, speak, strict=False,
                 situation_extra="The player said something mid-game. If the board changed (the note "
                 "above), acknowledge it smugly; otherwise just banter, staying aware of the state.")
        return True


def finalize_if_active(*, notify: Callable = print) -> None:
    """Save an in-progress game on shutdown so it resumes next launch."""
    with _lock:
        if _state is not None and _state.started:
            _save()
            try:
                notify("Deep IQ game saved - say 'let's play magic' to resume.")
            except Exception:
                pass
