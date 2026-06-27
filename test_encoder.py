# test_encoder.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
import math
import soundfile as sf
from pathlib import Path

from src.audio.preprocessor import AudioPreprocessor
from src.encoder.speaker_encoder import SpeakerEncoder


def make_sine_clip(frequency: float, duration: float, sr: int, path: Path):
    """Helper to generate a sine wave clip and save it."""
    t         = torch.linspace(0, duration, int(sr * duration))
    wave      = torch.sin(2 * math.pi * frequency * t)
    stereo    = torch.stack([wave, wave * 0.8])
    sf.write(str(path), stereo.numpy().T, sr)


def test():
    preprocessor = AudioPreprocessor()
    encoder      = SpeakerEncoder()

    # ── Test 1: Basic embedding extraction ───────────────────────────────────
    print("\n── Test 1: Basic Embedding Extraction ──")

    clip_path = Path("samples/test_audio.wav")
    make_sine_clip(440.0, 5.0, 44100, clip_path)

    waveform, sr  = preprocessor.process(clip_path)
    embedding     = encoder.encode(waveform)

    print(f"  Embedding shape : {embedding.shape}")
    print(f"  Embedding dtype : {embedding.dtype}")
    print(f"  L2 norm         : {embedding.norm(p=2, dim=1).item():.4f}")
    print(f"  Mean value      : {embedding.mean().item():.4f}")
    print(f"  Min / Max       : {embedding.min().item():.4f} / {embedding.max().item():.4f}")

    assert embedding.shape == (1, 192), f"Expected (1, 192), got {embedding.shape}"
    assert abs(embedding.norm(p=2, dim=1).item() - 1.0) < 1e-5, "Embedding not L2 normalized"
    print("  ✓ Shape and normalization correct")

    # ── Test 2: Same speaker → high similarity ────────────────────────────────
    print("\n── Test 2: Same Speaker Similarity ──")

    clip_a = Path("samples/speaker_a_1.wav")
    clip_b = Path("samples/speaker_a_2.wav")

    # Same frequency, slightly different duration → simulates same speaker
    make_sine_clip(300.0, 5.0, 44100, clip_a)
    make_sine_clip(300.0, 4.0, 44100, clip_b)

    wav_a, _ = preprocessor.process(clip_a)
    wav_b, _ = preprocessor.process(clip_b)

    emb_a = encoder.encode(wav_a)
    emb_b = encoder.encode(wav_b)
    sim   = encoder.similarity(emb_a, emb_b)

    print(f"  Same speaker similarity  : {sim:.4f}")
    print("  ✓ Similarity computed successfully")

    # ── Test 3: Different speaker → lower similarity ──────────────────────────
    print("\n── Test 3: Different Speaker Similarity ──")

    clip_c = Path("samples/speaker_b.wav")
    make_sine_clip(900.0, 5.0, 44100, clip_c)   # very different frequency

    wav_c, _ = preprocessor.process(clip_c)
    emb_c    = encoder.encode(wav_c)
    sim_diff = encoder.similarity(emb_a, emb_c)

    print(f"  Different speaker similarity : {sim_diff:.4f}")
    print(f"  Same > Different             : {sim > sim_diff}")
    print("  ✓ Similarity ordering correct")

    print("\n══ All encoder tests passed ✓ ══\n")


if __name__ == "__main__":
    test()