import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
import soundfile as sf
from pathlib import Path

REFERENCE_AUDIO = "samples/Rudra4.wav"
OUTPUT_PATH     = "outputs/cloned_test.wav"
TEXT            = "Honestly, I didn't think this would actually work, but hearing it back is quite fascinating."

def run():
    from src.synthesizer.tts_engine import TTSEngine

    engine = TTSEngine()

    print("\nRunning inference on CPU — this will take a few minutes...")
    waveform, sr = engine.synthesize(
        text=TEXT,
        reference_audio_path=REFERENCE_AUDIO,
        output_path=OUTPUT_PATH,
    )

    print(f"\nDone!")
    print(f"Output shape    : {waveform.shape}")
    print(f"Duration        : {waveform.shape[1] / sr:.2f}s")
    print(f"Saved to        : {OUTPUT_PATH}")

if __name__ == "__main__":
    run()