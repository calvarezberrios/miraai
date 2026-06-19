#!/usr/bin/env python3
"""Turn the Cynaptics/persona-chat dataset into chat-format training data for
Mira's LLM fine-tune.

Mira runs Qwen2.5-7B-Instruct (via Ollama: `qwen2.5:7b`). At runtime
`prefrontal_cortex.py` builds a system prompt from her persona and feeds the
conversation through the chat API. We want the fine-tune to sharpen the
*transferable skill* that flow depends on: stay in whatever character the system
prompt describes, and answer like a person in a conversation — not to bake one
fixed persona into the weights. So each training example uses the row's OWN
persona as the system message; Mira's persona stays injected at inference time.

Dataset shape (config `default`, split `train`, ~20k rows):
    conv_id    : str
    persona_b  : list[str]   - the speaker's character traits (this is "Persona B")
    dialogue   : list[str]   - alternating "Persona A: ..." / "Persona B: ..." lines
    reference  : str         - the gold next Persona-B reply for that context

We map  Persona A -> user,  Persona B -> assistant,  and append `reference` as the
final assistant turn. The result is written as JSONL, one object per line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}
which the Colab notebook (mira_finetune_colab.ipynb) renders with Qwen's chat
template. The notebook does this same prep itself; this script is a local/CPU
convenience for generating or inspecting the data without a GPU.

Usage:
    python prepare_data.py --out data/persona_chat.jsonl
    python prepare_data.py --out data/persona_chat.jsonl --max-rows 2000   # quick test
"""
import argparse
import json
import os
import re

from datasets import load_dataset

DATASET = "Cynaptics/persona-chat"

# "Persona A: hi" -> ("A", "hi"). Tolerant of stray spacing / case.
_SPEAKER_RE = re.compile(r"^\s*persona\s*([ab])\s*[:\-]\s*(.*)$", re.IGNORECASE | re.DOTALL)


def _persona_system(traits: list[str]) -> str:
    """Render the speaker's trait list into a system prompt that frames the task
    the same way Mira's runtime prompt does: be this character, talk naturally."""
    bullets = "\n".join(f"- {t.strip()}" for t in traits if t and t.strip())
    return (
        "You are a character in a casual conversation. Stay fully in character and "
        "reply naturally and conversationally — like a real person, not an assistant. "
        "Keep replies short unless more is genuinely warranted, and do not narrate "
        "actions or add stage directions.\n\n"
        "This is who you are:\n"
        f"{bullets}"
    )


def _parse_turn(line: str):
    """Return ('user'|'assistant', text) for a dialogue line, or None if it has no
    recognizable Persona A/B prefix (rare malformed rows)."""
    m = _SPEAKER_RE.match(line or "")
    if not m:
        return None
    who, text = m.group(1).upper(), m.group(2).strip()
    if not text:
        return None
    # Persona A is the other person talking *to* our speaker -> user.
    # Persona B is our speaker (the one whose persona we condition on) -> assistant.
    return ("user" if who == "A" else "assistant", text)


def build_example(row: dict):
    """One dataset row -> a chat example, or None if it can't yield a trainable
    (i.e. ends on an assistant turn we can learn) conversation."""
    traits = list(row.get("persona_b") or [])
    if not traits:
        return None

    turns = []
    for line in row.get("dialogue") or []:
        parsed = _parse_turn(line)
        if parsed is None:
            continue
        role, text = parsed
        # Collapse accidental consecutive same-role lines so the user/assistant
        # alternation the chat template requires stays valid.
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + text
        else:
            turns.append({"role": role, "content": text})

    # Drop a leading assistant turn — the conversation must open with the *other*
    # person (user) for the template, and so we never train on an unprompted reply.
    if turns and turns[0]["role"] == "assistant":
        turns = turns[1:]

    reference = (row.get("reference") or "").strip()
    if reference:
        if turns and turns[-1]["role"] == "user":
            turns.append({"role": "assistant", "content": reference})
        elif not turns:
            # No usable dialogue context; skip — a bare reply has nothing to ground on.
            return None
        # else: dialogue already ends on an assistant turn; `reference` would be a
        # second consecutive assistant. Keep the dialogue as-is (it still ends on a
        # learnable assistant turn) and drop the duplicate reference.

    # Need a real exchange ending on an assistant turn to have anything to learn.
    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        return None

    messages = [{"role": "system", "content": _persona_system(traits)}, *turns]
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser(description="Prepare persona-chat for fine-tuning.")
    ap.add_argument("--out", default="data/persona_chat.jsonl",
                    help="output JSONL path")
    ap.add_argument("--split", default="train", help="dataset split to use")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap rows (0 = all) — handy for a quick smoke test")
    args = ap.parse_args()

    print(f"[prepare] loading {DATASET} ({args.split}) ...")
    ds = load_dataset(DATASET, split=args.split)
    if args.max_rows and args.max_rows < len(ds):
        ds = ds.select(range(args.max_rows))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            ex = build_example(row)
            if ex is None:
                skipped += 1
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

    print(f"[prepare] wrote {written} examples to {args.out} "
          f"({skipped} rows skipped as unusable)")
    if written:
        with open(args.out, encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("[prepare] sample example:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
