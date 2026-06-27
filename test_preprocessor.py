# test_preprocessor.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
import math
import soundfile as sf
import numpy as np
from pathlib import Path
from src.audio.preprocessor import AudioPreprocessor

def test():
    # Generate a synthetic test clip (5 seconds of a sine wave)
    sr        = 44100
    duration  = 5.0
    frequency = 440.0
    t         = torch.linspace(0, duration, int(sr * duration))
    sine_wave = torch.sin(2 * math.pi * frequency * t)

    # Make it stereo to test mono conversion
    stereo = torch.stack([sine_wave, sine_wave * 0.8])  # (2, samples)

    # Save using soundfile directly (bypass torchaudio.save)
    test_path = Path("samples/test_audio.wav")
    audio_np  = stereo.numpy().T  # soundfile expects (samples, channels)
    sf.write(str(test_path), audio_np, sr)
    print(f"Saved synthetic test audio → {test_path}")

    # Run through preprocessor
    preprocessor     = AudioPreprocessor()
    waveform, out_sr = preprocessor.process(test_path)

    print(f"\nResults:")
    print(f"  Shape      : {waveform.shape}")
    print(f"  Sample rate: {out_sr}")
    print(f"  Duration   : {waveform.shape[1] / out_sr:.2f}s")
    print(f"  Peak amp   : {waveform.abs().max():.4f}")
    print(f"  Channels   : {waveform.shape[0]}")
    print(f"\nAll checks passed ✓")

if __name__ == "__main__":
    test()