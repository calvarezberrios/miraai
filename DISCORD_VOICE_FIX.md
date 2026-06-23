# Discord voice — the ~5-second "stutter/cutoff" bug and the fix

**TL;DR:** Mira's Discord voice was chopping speech into ~5-second chunks and replying
mid-sentence. After ruling out endpointing, mic settings, noise suppression, and the network,
the real cause turned out to be a **bug in the pinned experimental py-cord voice-receive build**:
received audio packets were processed on the asyncio event loop, but the loop was only woken by
its ~5s voice heartbeat — so audio was handed to us in 5-second bursts. The fix is a tiny
**event-loop "pump"** in our adapter that keeps the loop awake during voice. No py-cord fork.

> **Is it safe to PR this to py-cord?** See **[the assessment at the bottom](#can-this-be-contributed-upstream-to-py-cord)**. Short version: yes to *reporting the bug* (and optionally a proper upstream patch to `SocketReader`), but our loop-pump is an app-side workaround, not the thing to submit verbatim.

---

## Symptom
- Talking continuously (no pauses), speech got split into ~5-second chunks; Mira replied to each
  fragment while the user was still talking.
- The gap was suspiciously **constant: 4.97s**, over and over.
- Crucially: **no words were lost** — audio went silent for ~5s then resumed *exactly* where it
  left off (e.g. stutter at "twelve" → 5s silence → resume at "twelve/thirteen").

That last detail is what cracked it: intact-but-late audio = a **delivery/processing batching**
problem, not packet loss, not the mic, not the network.

## Root cause (in py-cord, not our code)
Tracing the receive path from the UDP socket to our sink's `write()`:

1. **`discord/voice/state.py` — `SocketReader._do_run`** (its own thread) reads each UDP voice
   packet and schedules its processing onto the asyncio event loop:
   ```python
   task = asyncio.ensure_future(
       self.state.loop.create_task(utils.maybe_coroutine(cb, data)),
       loop=self.state.loop,
   )
   ```
   `loop.create_task(...)` called from a **non-loop thread does not wake the loop.**
2. Our sink is **non-opus** (`discord/sinks/core.py: is_opus() -> False`), so each packet is
   decrypted + opus-decoded + delivered to our `write()` *inside that scheduled task* — i.e. only
   when the event loop actually runs it.
3. On a quiet voice channel (just you + the bot, no other server traffic) the loop sleeps in
   `select()` until its next timer. The soonest timer is the **voice gateway heartbeat, capped at
   `min(interval, 5)` ≈ 5s** (`discord/voice/gateway.py`).
4. So every ~5s the heartbeat wakes the loop, it drains *all* the queued packet tasks at once, and
   ~5s of audio lands in our sink in a single burst. → the steady 4.97s gaps.

(The opus `JitterBuffer` is **not** involved — that path is only used for opus sinks; ours decodes
to PCM and takes the immediate-write branch.)

## The fix — `_loop_pump` (no py-cord changes)
`peripheral_nervous_system/discord_adapter.py` runs a tiny daemon thread that, while a voice
client exists, pokes the event loop every 10 ms with the thread-safe `call_soon_threadsafe`:

```python
def _loop_pump(self):
    def _noop():
        return None
    while self._running:
        loop = self._loop
        vc = self._voice_client
        if loop is not None and vc is not None:
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(_noop)   # wakes the loop -> queued packet tasks run now
            except Exception:
                pass
        time.sleep(0.01)
```

`call_soon_threadsafe` **does** wake the loop (unlike `create_task` from another thread), so the
already-scheduled packet tasks run in near-real-time instead of waiting for the 5s heartbeat. It's
a no-op callback, costs nothing meaningful, and only runs during voice.

**Result:** audio is delivered in real time, the `delivery gap` lines disappear, and you can talk
freely without being cut off. This was the core bug behind the whole multi-session saga.

## Supporting changes made while chasing this
All in `discord_adapter.py` unless noted:
- **Endpointing rewritten** to be robust: a turn ends only on a *transmission* gap
  (`now - last_voice`), never on silence *inside* the bursty buffer; and it **never finalizes while
  Mira is speaking** (`_brain_busy`) so her reply can't chop your continued speech into pieces.
  The whole utterance is **re-transcribed fresh at finalize** for full context.
- **`VOICE_END_SILENCE`** now `0.7s` (real-time delivery makes snappy turn-taking possible; was
  temporarily inflated to 6.5s only to bridge the batching bug). Tune via `MIRA_VOICE_END_SILENCE`.
- **Live-caption partials** transcribe only the last `MIRA_VOICE_PARTIAL_WINDOW_SEC` seconds
  (default 6) instead of the whole growing buffer, so they can't peg the GPU.
- **Diagnostics** (all off by default, handy if voice ever misbehaves again):
  - `MIRA_VOICE_DEBUG=1` — prints `delivery gap X.Xs` (a return of steady ~5s gaps = the pump isn't
    keeping up) and `finalized … (quiet X.Xs, …)`.
  - `MIRA_VOICE_DUMP=1` — writes each finalized utterance to `voice_debug/<speaker>_<ts>.wav` so you
    can *listen* to exactly what was captured (the single most useful diagnostic).
  - `MIRA_VOICE_LOG=1` — surfaces py-cord's DAVE decrypt internals (filtered), to tell a
    library-side drop from a network drop.

## How to verify
```powershell
$env:MIRA_VOICE_DEBUG=1; $env:MIRA_VOICE_DUMP=1; .\start_discord.bat
```
Join a VC, count 1→20 continuously. Expect: **no `delivery gap 4.97s` lines**, one whole turn (no
mid-word split), a continuous `voice_debug/*.wav`, and a reply ~`VOICE_END_SILENCE` after you stop.

---

## Can this be contributed upstream to py-cord?

**Reporting the bug: yes, safe and probably welcome.** The behavior (received audio delivered in
~5s heartbeat-sized bursts on a quiet connection, because `SocketReader` schedules packet tasks
without waking the loop) is a genuine, reproducible defect in the voice-receive path.

**But submit the *upstream* fix, not our workaround.** Our `_loop_pump` is an application-side
band-aid — busy-poking someone else's event loop is not something to PR. The proper fix lives in
py-cord's `SocketReader._do_run` (`discord/voice/state.py`): wake the loop when scheduling, e.g.
use `asyncio.run_coroutine_threadsafe(coro, loop)` (which signals the loop) instead of
`loop.create_task(...)` from the reader thread — or, since the receive callback chain is
synchronous, invoke sync callbacks directly on the reader thread and only bounce coroutine
callbacks to the loop.

**Caveats before opening a PR:**
1. **It's an in-progress branch.** We're pinned to commit `820460aa4` of the DAVE voice-receive PR
   (#3159), which is actively being developed. First `git pull` the latest commit on that branch
   and confirm the bug still exists — it may already be fixed or known.
2. **The real fix needs testing against py-cord**, not just our app. Changing the dispatch touches
   thread-safety and the non-voice callbacks (e.g. IP-discovery during connect, which *is* async).
   A patch should be validated end-to-end before submitting.
3. **Provide a minimal repro** in the issue/PR (bot joins a quiet VC, a `Sink.write` timestamp log
   shows ~5s delivery gaps that vanish when the loop is kept awake). Maintainers will want that.
4. **Risk to us is low** (it's a normal open-source contribution; nothing here touches our repo or
   secrets), and it would let us eventually drop both the pin *and* the `_loop_pump`.

**Recommendation:** open a GitHub **issue** on Pycord first with the repro and the root-cause
analysis above; offer the `SocketReader` fix as a PR if a maintainer confirms it isn't already
being handled on the branch. Keep `_loop_pump` locally until an upstream fix ships and is verified.
