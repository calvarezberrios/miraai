# stt_test.py — standalone Wernicke's area test (project root)
from brain.forebrain.cerebrum.temporal_lobe import wernickes_area as ears

ears.start(
    on_partial=lambda t: print(f"\r[mic] {t}{' ' * 10}", end="", flush=True),
    on_final=lambda t: print(f"\nFINAL >> {t}"),
    on_interrupt=lambda t: print(f"\n[Mira would interrupt here] {t}"),
)

input("Listening — talk, then pause ~3s to finalize. Press Enter to quit.\n")
ears.stop()