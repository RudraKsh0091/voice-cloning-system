import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_loading():
    print("\n── Test: XTTS-v2 Model Loading ──")

    from src.synthesizer.tts_engine import TTSEngine

    engine = TTSEngine()

    assert engine.model is not None, "Model failed to load"
    print("  Model object   : ✓")

    assert hasattr(engine.model, "tts"), "tts method missing"
    print("  tts()          : ✓")

    assert hasattr(engine.model, "tts_to_file"), "tts_to_file method missing"
    print("  tts_to_file()  : ✓")

    print(f"  Device         : {engine.device}")
    print("\n── XTTS-v2 loaded successfully ✓ ──")
    print("NOTE: Run full inference on Kaggle (T4 GPU) for actual voice cloning.\n")

if __name__ == "__main__":
    test_loading()