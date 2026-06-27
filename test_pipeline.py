import librosa
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from pathlib import Path
from src.pipeline import VoiceCloner

def test():
    # ── Initialize pipeline ───────────────────────────────────────────────────
    cloner = VoiceCloner()

    # ── Test 1: Basic cloning ─────────────────────────────────────────────────
    print("\n── Test 1: Voice Cloning ──")

    REFERENCE = "samples/Rudra4.wav"
    TEXT      = "This is the voice cloning pipeline speaking in your cloned voice."

    output_path = cloner.clone(
        reference_audio = REFERENCE,
        text            = TEXT,
    )

    assert output_path.exists(), "Output file was not created"
    print(f"  Output file    : {output_path} ✓")
    print(f"  File size      : {output_path.stat().st_size / 1024:.1f} KB")

    # ── Test 2: Custom output path ────────────────────────────────────────────
    print("\n── Test 2: Custom Output Path ──")

    custom_output = cloner.clone(
        reference_audio = REFERENCE,
        text            = "Testing custom output path.",
        output_path     = "outputs/custom_test.wav"
    )

    assert Path("outputs/custom_test.wav").exists()
    print(f"  Custom output  : {custom_output} ✓")

    # ── Test 3: Voice comparison ──────────────────────────────────────────────
    print("\n── Test 3: Voice Similarity (original vs cloned) ──")

    similarity = cloner.compare_voices(REFERENCE, output_path)
    print(f"  Similarity score : {similarity:.4f}")
    print(f"  (Higher = more similar. >0.70 is good for cloned voice)")

    print("\n══ All pipeline tests passed ✓ ══\n")

if __name__ == "__main__":
    test()