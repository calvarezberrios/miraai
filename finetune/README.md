# Fine-tuning Mira's LLM for better conversational text

Fine-tunes **Qwen2.5-3B-Instruct** on the
[`Cynaptics/persona-chat`](https://huggingface.co/datasets/Cynaptics/persona-chat)
dataset, then exports a **GGUF + Ollama Modelfile** you point `MIRA_MODEL` at.
Nothing about how Mira talks to the model changes; she just gets a checkpoint
that's better at natural, in-character conversation.

We target the **3B** (rather than the 7B Mira ships with): it trains ~2× faster so
it finishes inside a free Colab session, and once it's your Ollama model it runs
comfortably on your 6 GB GTX 1660 alongside Whisper + TTS — the size the main
[README](../README.md) calls the 6 GB sweet spot.

**Training runs on a free Google Colab GPU, not your machine.** Even a 3B QLoRA
wants ~8–10 GB to train, and the 1660 Super's weak fp16 path makes local training
unreliable anyway — so we train on a free Colab T4 (16 GB) and run the result
locally. The whole thing is one notebook:
[`mira_finetune_colab.ipynb`](mira_finetune_colab.ipynb).

> **Colab rate-limited?** Use [`mira_finetune_kaggle.ipynb`](mira_finetune_kaggle.ipynb)
> instead — same job on **Kaggle Notebooks** (free T4, 30 GPU-hrs/week on a quota
> separate from Colab). Enable GPU + Internet in the session settings, Run All, and
> grab `mira-gguf.zip` from the Output panel.

## What it teaches (and what it deliberately doesn't)

`persona-chat` rows are: a list of **persona traits** (`persona_b`), a
back-and-forth **dialogue**, and the gold **reference** reply. Each row becomes a
chat example where the system prompt is *that row's* persona and the target is the
next reply.

So the fine-tune learns the **transferable skill** — "given a persona in the
system prompt, stay in character and reply like a person" — **not** a fixed
personality baked into the weights. That matters because at runtime
[`prefrontal_cortex.py`](../brain/forebrain/cerebrum/frontal_lobe/prefrontal_cortex.py)
injects Mira's own persona, grounding rules, mood, and memories as the system
message. Training on generic personas keeps that injection working and avoids
overfitting to one character. We train **on the responses only** (the prompt is
masked), so the model learns *how to reply*, not how to echo personas/questions.

## How to run it

1. Open [`mira_finetune_colab.ipynb`](mira_finetune_colab.ipynb) in
   [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. **Runtime → Change runtime type → GPU** (the free T4 is enough).
3. **Runtime → Run all.** It installs Unsloth, prepares the data, trains (well
   under an hour on a T4 for the 3B), exports the GGUF, and downloads
   `mira-gguf.zip`. (It also asks to mount Google Drive — click through it once;
   that's where checkpoints + the final model are saved.)
4. On your PC, unzip it and register the model with Ollama:
   ```powershell
   cd mira-gguf
   ollama create mira -f Modelfile
   ```
5. Point Mira at it by adding a line to your `.env` (loaded on startup by
   `env_loader.py`):
   ```
   MIRA_MODEL=mira
   ```
   then run her: `python ..\main.py`. (A real system/shell env var still overrides
   `.env` if you ever want a one-off. The default in
   [`prefrontal_cortex.py`](../brain/forebrain/cerebrum/frontal_lobe/prefrontal_cortex.py)
   is `qwen2.5:3b`.)

### Sized for free Colab (important)
A maxed-out run (7B, `MAX_SEQ_LEN=2048`, all ~20k rows) needs **~7 hours** on a free
T4 — well past the free GPU-time limit, so it gets cut off partway (this happened).
The notebook's defaults avoid that: the **3B** model, `MAX_SEQ_LEN=1024` (persona-chat
turns are short, and the shorter context stops the T4 from offloading gradients to
RAM — what makes a big run crawl), and `MAX_ROWS=8000`. That finishes in **well
under an hour**. It also checkpoints to **Google Drive** with `USE_DRIVE`, so if
Colab disconnects you just **Run all** again and it **resumes** from the last
checkpoint instead of restarting.

All knobs live in the notebook's **Config** cell: `BASE_MODEL`, `EPOCHS`,
`LORA_RANK`, `BATCH_SIZE`/`GRAD_ACCUM`, `LEARNING_RATE`, `MAX_SEQ_LEN`,
`GGUF_QUANT`, `MAX_ROWS` (set `0` for the whole dataset), and `USE_DRIVE`/`SAVE_STEPS`.

**Want the higher-quality 7B instead?** Set `BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"`
in the Config cell — it trains and exports identically, but ~2× slower (lean on the
Drive checkpoints to resume across sessions) and is a tight fit running locally on
6 GB next to Whisper + TTS.

## `prepare_data.py` (optional, local)

The notebook prepares the data itself, so you don't need this. It's kept as a
CPU-only utility to **generate or inspect** the training JSONL on your own machine:

```powershell
python -m pip install -r requirements-finetune.txt
python prepare_data.py --out data/persona_chat.jsonl --max-rows 50
```

It applies the exact same row→messages transformation the notebook uses.

## Outputs (produced in Colab, downloaded to you)

- `persona_chat.jsonl` — prepared chat-format training data.
- `mira-gguf/*.gguf` + `mira-gguf/Modelfile` — what Ollama imports. The Modelfile
  sets `temperature 0.7` / `num_predict 300` to match
  [`prefrontal_cortex.py`](../brain/forebrain/cerebrum/frontal_lobe/prefrontal_cortex.py),
  and carries a short Mira `SYSTEM` so `ollama run mira` is in-character for a
  quick check. (Mira's runtime always sends her full system prompt, which
  overrides that default — the win here is the *behaviour*, not the baked text.)

## Verifying it helped

A/B the stock vs fine-tuned model on the same line:

```powershell
ollama run qwen2.5:3b "Reply to: 'ugh I lost again'"
ollama run mira       "Reply to: 'ugh I lost again'"
```

Then run Mira both ways (`MIRA_MODEL=qwen2.5:3b` vs `MIRA_MODEL=mira`) and listen
for: more natural phrasing, fewer tacked-on "what about you?" engagement
questions, steadier persona — without new factual invention (the grounding rules
in the system prompt still do that job).

## Troubleshooting

**`ollama run mira` outputs only `@@@@@@` / gibberish.** The model is fine; the
GGUF/Ollama import is the problem. Two causes we hit:

1. **You imported the safetensors, not a GGUF.** The notebook exports the GGUF to a
   sibling folder named `mira-gguf_gguf/` (note the suffix), while `mira-gguf/`
   holds the intermediate **16-bit safetensors merge**. If your zip/`ollama create`
   picked up `mira-gguf/`, Ollama imports raw safetensors — and **Ollama's
   safetensors→GGUF importer produces gibberish for this Qwen2.5 fine-tune.** Make
   sure you ship the `.gguf` file, not the `.safetensors`.

2. **If you only have the safetensors**, don't let Ollama convert them. Convert with
   llama.cpp's canonical converter instead, then import that GGUF:
   ```powershell
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   pip install -r llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
   python llama.cpp\convert_hf_to_gguf.py <safetensors-dir> --outfile mira-f16.gguf --outtype f16
   # reuse qwen2.5's known-good template, point FROM at the gguf, quantize on import:
   ollama show qwen2.5:3b --modelfile | Where-Object { $_ -notmatch '^FROM ' } | Set-Content tmpl.txt
   Set-Content Mira.modelfile ("FROM ./mira-f16.gguf`r`n" + (Get-Content tmpl.txt -Raw))
   ollama create mira -f Mira.modelfile -q q4_K_M
   ```

**Sanity-check the weights before blaming the export.** Load the merged model
directly with transformers (`AutoModelForCausalLM.from_pretrained(<dir>)`) and
generate one reply. Coherent text = weights are good, so the problem is purely the
GGUF/Ollama step (use the llama.cpp path above). Gibberish = re-export from training.
